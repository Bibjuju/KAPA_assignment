"""
High-level driver that converts a *LoadedPDF* into GitHub-flavoured markdown.

Pipeline
--------
1. **Render** each PDF page to an RGB numpy array with PyMuPDF.
2. **Detect** layout elements (tables, text, figures …) using YOLO-v10.
3. **Parse / OCR** – tables are parsed structurally, everything else is OCR-ed.
4. **Emit** one Markdown section per page (## Page N …).
"""


import concurrent.futures
import threading
from typing import List, Optional

import fitz
import numpy as np
from PIL import Image

from ..loader.types import LoadedPDF
from .base import PDFtoMarkdown
from .modules.constants import ID2NAME, CLASSES_TO_OCR
from .modules.predictors import Predictors
from .modules.table_parser import TableParser


class PymuConverter(PDFtoMarkdown):
    """
    Concrete “PDF → Markdown” converter.

    Notes
    -----
    * Uses **ThreadPoolExecutor** for page-level parallelism (I/O bound).
    * A single `Predictors` instance (YOLO, TT, OCR) is shared across threads.
      We guard calls with a mutex because neither doctr nor TT are guaranteed
      to be thread-safe.
    """

    MAX_WORKERS = 4  # How many pages to rasterise in parallel – tune per box

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        
        conf_thresh: float = 0.1,
        nms_iou_thresh: float = 0.2,
        zoom: float = 3.0,
        cell_thresh: float = 0.7,
        pad_frac: float = 0.03,
        cell_pad_frac: float = 0.03,
        overlap_thr: float = 0.8,
    ) -> None:
        super().__init__()

        # YOLO thresholds
        self.conf_thresh = conf_thresh
        self.nms_iou_thresh = nms_iou_thresh

        # PyMuPDF render scale
        self.zoom = zoom

        # Table-Transformer / OCR knobs
        self.cell_thresh = cell_thresh
        self.pad_frac = pad_frac
        self.cell_pad_frac = cell_pad_frac
        self.overlap_thr = overlap_thr

        # Heavy DL models – create once, reuse
        self._predictors = Predictors()

        # Mutex – doctr & TT call native libs that are not thread-safe
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def convert(self, doc: LoadedPDF) -> str:  # type: ignore[override]
        """
        Main entry: render every page then process them in parallel.

        Returns
        -------
        str
            The concatenated markdown of the whole document.
        """
        # ---------- 1. Render pages ----------------------------------- #
        pdf_fitz = fitz.open(stream=doc.raw_bytes, filetype="pdf")
        images = [
            self._page_to_rgb(pdf_fitz[i], zoom=self.zoom) for i in range(len(pdf_fitz))
        ]

        # ---------- 2. Process pages in parallel ---------------------- #
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as ex:
            futures = [ex.submit(self._process_page, img, idx)
                       for idx, img in enumerate(images)]

        # Gather in original order (futures already match the index list)
        pages_md: List[str] = [f.result()
                               for f in futures
                               if f.result().strip()]

        return "\n\n".join(pages_md)

    # ------------------------------------------------------------------ #
    # Per-page logic
    # ------------------------------------------------------------------ #
    def _process_page(self, img: np.ndarray, page_idx: int) -> str:
        """
        • Detect layout  
        • Parse tables or run OCR  
        • Merge everything back → Markdown
        """
        # ▸ YOLO layout detection ------------------------------------- #
        with self._lock:          # Predictor models share GPU / CPU context
            boxes, classes = self._predictors.predict_yolo_layouts(
                img,
                conf=self.conf_thresh,
                iou=self.nms_iou_thresh,
            )

        if boxes.size == 0:       # blank page
            return ""

        # Lists that preserve reading order
        tables_md: List[str] = []
        crops_for_ocr: List[Optional[np.ndarray]] = []

        # ▸ Split detections into “table” vs “text” ------------------- #
        for (x0, y0, x1, y1), cls_id in zip(boxes, classes):
            label = ID2NAME.get(int(cls_id), f"Class-{cls_id}")

            if label == "table":
                # Expand slightly so TT sees outer borders
                px0, py0, px1, py1 = self._expand_box(
                    x0, y0, x1, y1, img.shape, frac=self.pad_frac
                )
                table_crop = img[py0:py1, px0:px1]
                table_crop_no_pad = img[y0:y1, x0:x1]   # for fallback OCR

                # --- Table-Transformer → row/col boxes --------------- #
                with self._lock:
                    col_boxes, row_boxes = self._predictors.predict_table_layout(
                        Image.fromarray(table_crop),
                        cell_thresh=self.cell_thresh,
                    )

                # --- Structural parse → markdown ---------------------- #
                parser = TableParser(
                    col_boxes=col_boxes,
                    row_boxes=row_boxes,
                    overlap_thr=self.overlap_thr,
                )
                tables_md.append(
                    parser.to_markdown(
                        table_crop=table_crop,
                        cell_pad_frac=self.cell_pad_frac,
                        ocr_fn=self._predictors.ocr_batch,
                        table_crop_no_pad=table_crop_no_pad,
                    )
                )
                crops_for_ocr.append(None)          # placeholder for order

            elif label in CLASSES_TO_OCR:
                # Regular text / caption / formula – OCR later
                crops_for_ocr.append(img[y0:y1, x0:x1])

        # ▸ Batch OCR for non-table crops ----------------------------- #
        ocr_inputs = [c for c in crops_for_ocr if c is not None]
        ocr_texts: List[str] = []
        if ocr_inputs:                                # only if there is work
            with self._lock:
                ocr_texts = self._predictors.ocr_batch(ocr_inputs)

        # ▸ Merge back in original detection order -------------------- #
        md_parts: List[str] = []
        txt_idx = tbl_idx = 0
        for crop in crops_for_ocr:
            if crop is None:               # table placeholder
                md_parts.append(tables_md[tbl_idx])
                tbl_idx += 1
            else:                          # OCR result
                md_parts.append(ocr_texts[txt_idx])
                txt_idx += 1

        # Prepend page heading if anything non-empty
        page_md = "\n\n".join(md_parts)
        return f"## Page {page_idx + 1}\n{page_md}" if page_md.strip() else ""

    # ------------------------------------------------------------------ #
    # Static helpers – pure functions, no self needed
    # ------------------------------------------------------------------ #
    @staticmethod
    def _page_to_rgb(page: fitz.Page,  zoom: float = 2.0) -> np.ndarray:
        """Rasterise a single `fitz.Page` ➜ RGB numpy array."""
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        return img[..., :3] if pix.n == 4 else img  # drop alpha if present

    @staticmethod
    def _expand_box(
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        img_shape: tuple[int, int, int],
        
        frac: float = 0.10,
    ) -> tuple[int, int, int, int]:
        """
        Symmetrically pad a bounding box by `frac` × its width/height but keep
        the result inside image boundaries.
        """
        h, w = img_shape[:2]
        pad_w = int((x1 - x0) * frac)
        pad_h = int((y1 - y0) * frac)
        nx0 = max(0, x0 - pad_w)
        ny0 = max(0, y0 - pad_h)
        nx1 = min(w, x1 + pad_w)
        ny1 = min(h, y1 + pad_h)
        return nx0, ny0, nx1, ny1
