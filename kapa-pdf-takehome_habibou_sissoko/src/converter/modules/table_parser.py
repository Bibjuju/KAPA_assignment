"""
Convert the sets of **column** and **row** bounding-boxes predicted by
Table-Transformer into a 2-D grid of cells and return a GitHub-flavoured
markdown table.

The only heavy operation inside this class is OCR.  All geometry work is pure
NumPy / Python and therefore very fast.
"""

from typing import List, Callable

import numpy as np
from PIL import Image
import pytesseract


class TableParser:
    """
    Build a rectangular cell grid from two 1-D lists of boxes and emit markdown.

    Parameters
    ----------
    col_boxes, row_boxes
        Axis-aligned bounding boxes in **XYXY** order coming from the
        Table-Transformer—`col_boxes` mark column stripes, `row_boxes` mark row
        stripes.
    overlap_thr
        Fractional overlap used to de-duplicate *almost identical* stripes.
        (E.g. TT sometimes emits two columns that overlap by > 95 %.)

    Notes
    -----
    * The first row often contains **rotated header text**.  By default we
      OCR those cells twice (no rotation ➜ –90 ° rotation) and keep the better
      result; the other rows are OCR-ed once.
    * If the table degenerates to a single row or single column, we hand the
      entire crop to a higher-level `ocr_fn` helper—Doctr, paddleocr, etc.
    """

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        col_boxes: List[np.ndarray],
        row_boxes: List[np.ndarray],
        overlap_thr: float,
    ) -> None:
        # De-duplicate stripes along each axis separately
        self.col_boxes = self._dedup(col_boxes, overlap_thr, axis="x")
        self.row_boxes = self._dedup(row_boxes, overlap_thr, axis="y")

        # Reading order: left→right for columns, top→bottom for rows
        self.col_boxes.sort(key=lambda b: b[0])
        self.row_boxes.sort(key=lambda b: b[1])

        # Cartesian product → individual cell boxes
        self._synth_cells()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def to_markdown(
        self,
        table_crop: np.ndarray,
        
        cell_pad_frac: float,
        ocr_fn: Callable[[List[np.ndarray]], List[str]],
        table_crop_no_pad: np.ndarray,
    ) -> str:
        """
        Convert the detected grid into markdown.

        Parameters
        ----------
        table_crop
            RGB array containing the table **with** extra padding (helps TT).
        cell_pad_frac
            Extra padding (fraction of cell size) added around each cell before
            passing it to Tesseract—for better context.
        ocr_fn
            Batch OCR function for the *degenerate* case where TT could not
            build a true 2-D grid (e.g. only one row detected).
        table_crop_no_pad
            Same as `table_crop` but **without** the padding that was added
            earlier.  This is what `ocr_fn` will see.

        Returns
        -------
        str
            GitHub-flavoured markdown table.
        """
        # Fallback: not a real table ➜ OCR the whole thing in one go
        if not (self.nrows > 1 and self.ncols > 1):
            return ocr_fn([table_crop_no_pad])[0]

        th, tw = table_crop.shape[:2]
        cell_texts: List[str] = []

        # OCR every cell (row-major order)
        for idx, (x0, y0, x1, y1) in enumerate(
            map(lambda c: list(map(int, c)), self.cells)
        ):
            row_idx = idx // self.ncols  # zero-based row number

            # Pad the crop a little to avoid clipping characters
            pad_w = int((x1 - x0) * cell_pad_frac)
            pad_h = int((y1 - y0) * cell_pad_frac)
            sx0, sy0 = max(0, x0 - pad_w), max(0, y0 - pad_h)
            sx1, sy1 = min(tw, x1 + pad_w), min(th, y1 + pad_h)

            # Skip empty/invalid crops
            if sx1 <= sx0 or sy1 <= sy0:
                cell_texts.append("")
                continue

            crop = table_crop[sy0:sy1, sx0:sx1]
            rotate_hdr = row_idx == 0  # only first row may be rotated
            text = self._ocr(crop, rotate_test=rotate_hdr)
            cell_texts.append(text)

        # Assemble the final markdown
        rows_md = [
            "| " + " | ".join(cell_texts[r * self.ncols : (r + 1) * self.ncols]) + " |"
            for r in range(self.nrows)
        ]
        if len(rows_md) > 1:  # insert the GitHub separator after the header
            rows_md.insert(
                1, "| " + " | ".join("---" for _ in range(self.ncols)) + " |"
            )
        return "\n".join(rows_md)

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _box_area(b: np.ndarray) -> int:
        """Pixel area of a box (for sorting small→large)."""
        x0, y0, x1, y1 = b
        return max(0, x1 - x0) * max(0, y1 - y0)

    @classmethod
    def _dedup(
        cls,
        boxes: List[np.ndarray],
        thr: float,
        axis: str,
    ) -> List[np.ndarray]:
        """
        Remove nearly-duplicate stripes along one axis.

        Two boxes are considered duplicates if they overlap by `thr` or more
        **along the given axis** (`"x"` for columns, `"y"` for rows).
        """
        def overlap(a: np.ndarray, b: np.ndarray) -> float:
            if axis == "x":  # horizontal overlap
                left, right = max(a[0], b[0]), min(a[2], b[2])
                inter = max(0, right - left)
                base = min(a[2] - a[0], b[2] - b[0])
            else:            # vertical overlap
                top, bottom = max(a[1], b[1]), min(a[3], b[3])
                inter = max(0, bottom - top)
                base = min(a[3] - a[1], b[3] - b[1])
            return inter / base if base else 0.0

        keep: List[np.ndarray] = []
        for bx in sorted(boxes, key=cls._box_area):
            if all(overlap(bx, k) < thr for k in keep):
                keep.append(bx)
        return keep

    def _synth_cells(self) -> None:
        """Cartesian product of row & column stripes → list of cell boxes."""
        self.cells: List[List[int]] = []
        for r in self.row_boxes:
            for c in self.col_boxes:
                x0, y0 = max(r[0], c[0]), max(r[1], c[1])
                x1, y1 = min(r[2], c[2]), min(r[3], c[3])
                if x1 > x0 and y1 > y0:  # non-empty intersection
                    self.cells.append([x0, y0, x1, y1])

        self.nrows, self.ncols = len(self.row_boxes), len(self.col_boxes)

    # ------------------------------------------------------------------ #
    # OCR helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean(txt: str) -> str:
        """Strip Tesseract artefacts (pipes, newlines, trailing spaces)."""
        return txt.replace("|", "").replace("\n", "").strip()

    @classmethod
    def _alphanum_ratio(cls, txt: str) -> float:
        """Heuristic: proportion of alpha-numeric characters in `txt`."""
        cleaned = cls._clean(txt)
        return sum(ch.isalnum() for ch in cleaned) / len(cleaned) if cleaned else 0.0

    @classmethod
    def _ocr(cls, img: np.ndarray,  rotate_test: bool = False) -> str:
        """
        Run Tesseract on `img`.

        If `rotate_test=True`, we OCR **twice** (no rotation, then –90 °) and
        keep the text with the higher “alpha-numeric ratio”.
        """
        pil = Image.fromarray(img)

        # 1️⃣  straight orientation
        text0 = pytesseract.image_to_string(pil, config="--psm 6")
        if not rotate_test:
            return cls._clean(text0)

        # 2️⃣  header heuristic
        ratio0 = cls._alphanum_ratio(text0)
        if ratio0 >= 0.82:  # good enough, skip second pass
            return cls._clean(text0)

        text1 = pytesseract.image_to_string(pil.rotate(-90, expand=True), config="--psm 6")
        return (
            cls._clean(text1)
            if cls._alphanum_ratio(text1) > ratio0
            else cls._clean(text0)
        )
