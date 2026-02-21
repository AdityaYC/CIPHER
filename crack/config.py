"""
Crack detection configuration.

All paths are relative to the repository root unless absolute.
Override via environment variables when needed.
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Path to the exported ONNX model (relative to repo root).
CRACK_ONNX_PATH = os.environ.get(
    "CRACK_ONNX_PATH",
    str(_REPO_ROOT / "models" / "yolov8_crack_seg.onnx"),
)

# Enable/disable crack detection at startup.
CRACK_ENABLED = os.environ.get("CRACK_ENABLED", "1").strip().lower() in ("1", "true", "yes")

# Confidence threshold — lower than standard YOLO (0.45) because cracks
# are subtle and the model is trained on a narrow domain.
CRACK_CONFIDENCE_THRESHOLD = float(os.environ.get("CRACK_CONFIDENCE_THRESHOLD", "0.35"))

# NMS IoU threshold.
CRACK_NMS_IOU_THRESHOLD = float(os.environ.get("CRACK_NMS_IOU_THRESHOLD", "0.5"))

# Input resolution expected by the model.
CRACK_INPUT_SIZE = 640

# HuggingFace repo and available variants.
HF_REPO = "OpenSistemas/YOLOv8-crack-seg"
VARIANT_FILES = {
    "yolov8n": "yolov8n/weights/best.pt",
    "yolov8s": "yolov8s/weights/best.pt",
    "yolov8m": "yolov8m/weights/best.pt",
    "yolov8l": "yolov8l/weights/best.pt",
    "yolov8x": "yolov8x/weights/best.pt",
}

# Visualization color for crack detections (BGR): orange-red to distinguish
# from standard YOLO green (0, 255, 102).
CRACK_VIS_COLOR_BGR = (0, 80, 255)
