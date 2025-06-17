"""
Light-weight wrappers around all deep-learning inference models used by the
pipeline:

* **YOLO-v10** – page-level object / layout detection.
* **Table-Transformer** – recognise row / column stripes inside a table crop.
* **doctr** – high-accuracy OCR.

Putting every heavy model behind the `Predictors` façade has a few benefits:

1.  **Single place to tweak hyper-params** (e.g. confidence thresholds).
2.  **Lazy, on-demand initialisation** so command-line tools start instantly.
3.  **Easy mocking** in unit tests (swap the whole object for a stub).
"""


from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

from doclayout_yolo import YOLOv10  # third-party package


class Predictors:
    """
    Central access point for all DL models.

    Parameters
    ----------
    model_weights
        Filesystem path to the YOLOv10 checkpoint.  The default is the model
        fine-tuned on **DocStructBench** at 1024 px resolution.
    """

    # A shared doctr instance – created once, reused across *all* `Predictors`
    # objects.  This avoids loading large weights multiple times.
    _DOCTR_MODEL = None

    # --------------------------------------------------------------------- #
    # Construction & initialisation
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        model_weights: str = "./doclayout_yolo_docstructbench_imgsz1024.pt",
    ) -> None:
        # Auto-select the fastest device available
        self.device: torch.device = self._select_device()

        # ---------- 1. Object-layout detector (YOLO-v10) ---------------- #
        # Note: YOLOv10 will transparently send tensors to the *first*
        # available GPU, so we don't need to call .to(self.device) here.
        self._detector = YOLOv10(model_weights)

        # ---------- 2. Table-Transformer -------------------------------- #
        self._tt_processor = DetrImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        self._tt_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        ).to(self.device)

        # ---------- 3. OCR (doctr) -------------------------------------- #
        if Predictors._DOCTR_MODEL is None:
            # Lazy import so users can run ‘--help’ without loading GPUs
            from doctr.models import ocr_predictor  # pylint: disable=import-error

            Predictors._DOCTR_MODEL = (
                ocr_predictor(detect_orientation=True, pretrained=True)
                .to(self.device)
                .eval()
            )

    # ------------------------------------------------------------------ #
    # Device utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_device() -> torch.device:
        """
        Pick the best available compute backend.

        Priority: **CUDA ➜ Apple MPS ➜ CPU**.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------ #
    # 1. Page-level layout detection (YOLO-v10)
    # ------------------------------------------------------------------ #
    def predict_yolo_layouts(
        self,
        img: np.ndarray,
        
        conf: float,
        iou: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect layout objects on a single page image.

        Parameters
        ----------
        img
            RGB numpy array returned by PyMuPDF.
        conf, iou
            Confidence threshold and NMS IoU threshold forwarded to YOLO.

        Returns
        -------
        boxes, classes
            • **boxes** – *(N, 4)* int array in *XYXY* order  
            • **classes** – *(N,)* int array of class-ids

        Both arrays are empty (shape (0, …)) if nothing is detected.
        """
        yolo_res = self._detector.predict(img, conf=conf, iou=iou)
        if not yolo_res or not hasattr(yolo_res[0], "boxes"):
            return np.empty((0, 4), dtype=int), np.empty((0,), dtype=int)

        det = yolo_res[0]
        boxes = det.boxes.xyxy.cpu().numpy().astype(int)
        classes = det.boxes.cls.cpu().numpy().astype(int)
        return boxes, classes

    # ------------------------------------------------------------------ #
    # 2. Table structure recognition (Table-Transformer)
    # ------------------------------------------------------------------ #
    def predict_table_layout(
        self,
        pil_table: Image.Image,
        
        cell_thresh: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Predict row & column stripes for a *single* table crop.

        Returns two parallel lists: ``col_boxes`` and ``row_boxes``.
        """
        # Pre-tokenise & send to the correct device
        encoding = self._tt_processor(pil_table, return_tensors="pt").to(self.device)

        # Forward pass with gradients disabled (inference only)
        with torch.no_grad():
            outputs = self._tt_model(**encoding)

        # Convert raw logits to bounding boxes
        dets = self._tt_processor.post_process_object_detection(
            outputs,
            threshold=cell_thresh,
            target_sizes=[pil_table.size[::-1]],
        )[0]

        # Split the predictions into columns vs rows
        boxes_tt: np.ndarray = (
            dets["boxes"].cpu().numpy()
            if hasattr(dets["boxes"], "cpu")
            else dets["boxes"].numpy()
        )
        labels_tt: np.ndarray = (
            dets["labels"].cpu().numpy()
            if hasattr(dets["labels"], "cpu")
            else dets["labels"].numpy()
        )

        col_boxes = [boxes_tt[i] for i, lbl in enumerate(labels_tt) if lbl in (1, 3)]
        row_boxes = [boxes_tt[i] for i, lbl in enumerate(labels_tt) if lbl in (2, 4)]
        return col_boxes, row_boxes

    # ------------------------------------------------------------------ #
    # 3. OCR (doctr)
    # ------------------------------------------------------------------ #
    def ocr_batch(self, imgs: List[np.ndarray]) -> List[str]:
        """
        Run doctr OCR over a *batch* of cropped regions.

        The order of `imgs` is preserved in the returned list.
        """
        if not imgs:
            return []

        # doctr requires a list of numpy arrays in RGB; enforce that
        img_list = [
            np.asarray(img).copy() if isinstance(img, Image.Image) else img
            for img in imgs
        ]
        preds = Predictors._DOCTR_MODEL(img_list)

        texts: List[str] = []
        for page in preds.pages:
            exported = page.export()
            text = " ".join(
                w["value"]
                for blk in exported["blocks"]
                for line in blk["lines"]
                for w in line["words"]
            )
            texts.append(text.replace("|", "").strip())

        return texts
