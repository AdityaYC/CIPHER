"""
YOLOv8 single-class crack segmentation detector (ONNX).

Produces the same detection dict format as backend.perception.YOLODetector:
    {"class": "crack", "confidence": float, "bbox": [x1,y1,x2,y2], "center": [cx,cy]}

The ONNX export of OpenSistemas/YOLOv8-crack-seg outputs:
    output0 — (1, 37, 8400):  4 bbox + 1 class-conf + 32 mask-coeff per anchor
    output1 — (1, 32, 160, 160): prototype masks (unused here; kept for future overlay)

This detector only uses output0 for bounding-box detections.  Detection parsing is
fully vectorized (numpy) so it runs efficiently even on ARM devices without SIMD.

All heavy imports (numpy, cv2, onnxruntime) are deferred to method scope so the
module can be imported for config/constants without requiring those packages.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CRACK_CLASS_NAME = "crack"


class YOLOCrackSegDetector:
    """
    ONNX Runtime inference for the 1-class YOLOv8 crack-seg model.

    Compatible with QNN (Qualcomm NPU) when qnn_dll_path is provided,
    otherwise falls back to CPUExecutionProvider (works on ARM and x86).
    """

    def __init__(
        self,
        model_path: str,
        qnn_dll_path: str | None = None,
        confidence_threshold: float = 0.35,
        nms_iou_threshold: float = 0.5,
        input_size: int = 640,
        class_name: str = CRACK_CLASS_NAME,
    ):
        import onnxruntime as ort

        self._last_latency_ms: float = 0.0
        self._input_size = input_size
        self._conf_threshold = confidence_threshold
        self._nms_iou = nms_iou_threshold
        self._class_name = class_name

        resolved = self._resolve_model_path(model_path)

        providers = []
        if qnn_dll_path and Path(qnn_dll_path).exists():
            providers.append(("QNNExecutionProvider", {"backend_path": qnn_dll_path}))
        providers.append("CPUExecutionProvider")

        try:
            self._session = ort.InferenceSession(resolved, providers=providers)
        except Exception as exc:
            logger.warning("QNN load failed (%s), falling back to CPU", exc)
            self._session = ort.InferenceSession(
                resolved, providers=["CPUExecutionProvider"]
            )

        active = self._session.get_providers()
        if "QNNExecutionProvider" not in active:
            logger.info("Crack-seg running on CPU (%s)", active[0])
        else:
            logger.info("Crack-seg running on NPU (QNN)")

        self._input_name = self._session.get_inputs()[0].name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame) -> list[dict]:
        """Run inference on a BGR numpy frame.

        Returns a list of detection dicts compatible with the main YOLO pipeline:
            [{"class": "crack", "confidence": float, "bbox": [x1,y1,x2,y2], "center": [cx,cy]}, ...]
        """
        import time

        h_orig, w_orig = frame.shape[:2]
        blob = self._preprocess(frame)

        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: blob})
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0

        # output0: (1, C, N) where C = 4 + num_classes + 32(mask coeffs)
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]  # (C, N)

        return self._parse_detections(raw, w_orig, h_orig)

    def get_provider(self) -> str:
        return self._session.get_providers()[0]

    def get_last_latency(self) -> float:
        return self._last_latency_ms

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        p = Path(model_path)
        if p.is_file():
            return str(p)
        if not p.is_absolute():
            for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
                candidate = base / model_path
                if candidate.is_file():
                    return str(candidate)
        raise FileNotFoundError(f"Crack-seg ONNX not found: {model_path}")

    def _preprocess(self, frame):
        import cv2
        import numpy as np

        img = cv2.resize(frame, (self._input_size, self._input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    def _parse_detections(self, raw, w_orig: int, h_orig: int) -> list[dict]:
        """Vectorized extraction of bounding boxes from the (C, N) tensor."""
        import numpy as np

        # raw shape: (C, N) where N=8400 anchors, C = 4 + num_cls + 32
        num_features, num_anchors = raw.shape
        # For 1-class seg: C=37 -> num_classes = 37 - 4 - 32 = 1
        # For 1-class det (no masks): C=5 -> num_classes = 1
        num_mask_coeffs = 32 if num_features > 36 else 0
        num_classes = num_features - 4 - num_mask_coeffs
        if num_classes < 1:
            num_classes = 1
            num_mask_coeffs = num_features - 5

        # Transpose to (N, C) for easier row indexing
        data = raw.T  # (N, C)

        # Extract class confidence (columns 4 .. 4+num_classes), take max per anchor
        class_scores = data[:, 4 : 4 + num_classes]
        if num_classes == 1:
            confidences = class_scores[:, 0]
        else:
            confidences = class_scores.max(axis=1)

        # Threshold filter (vectorized)
        mask = confidences >= self._conf_threshold
        if not np.any(mask):
            return []

        filtered = data[mask]
        confs = confidences[mask]

        # Bounding boxes: cx, cy, w, h -> x1, y1, x2, y2 in original frame coords
        scale_x = w_orig / self._input_size
        scale_y = h_orig / self._input_size

        cx = filtered[:, 0] * scale_x
        cy = filtered[:, 1] * scale_y
        bw = filtered[:, 2] * scale_x
        bh = filtered[:, 3] * scale_y

        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        # Build detection list
        detections = []
        for i in range(len(confs)):
            detections.append(
                {
                    "class": self._class_name,
                    "confidence": float(confs[i]),
                    "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                    "center": [float(cx[i]), float(cy[i])],
                }
            )

        return self._nms(detections, self._nms_iou)

    @staticmethod
    def _nms(detections: list[dict], iou_threshold: float) -> list[dict]:
        """Non-maximum suppression on bbox IoU."""
        import numpy as np

        if not detections:
            return []
        boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        scores = np.array([d["confidence"] for d in detections], dtype=np.float32)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
            order = rest[iou <= iou_threshold]

        return [detections[j] for j in keep]
