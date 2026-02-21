"""
Crack detection package — YOLOv8 crack segmentation for drone environmental detection.

Model: OpenSistemas/YOLOv8-crack-seg (Hugging Face).
Single-class ONNX segmentation detector with the same bbox API as backend YOLO detectors.

Usage:
    from crack.detector import YOLOCrackSegDetector
    from crack.config import CRACK_ONNX_PATH, CRACK_CONFIDENCE_THRESHOLD

    det = YOLOCrackSegDetector(CRACK_ONNX_PATH, confidence_threshold=CRACK_CONFIDENCE_THRESHOLD)
    detections = det.detect(frame)
"""

from crack.detector import CRACK_CLASS_NAME, YOLOCrackSegDetector
from crack.config import (
    CRACK_CONFIDENCE_THRESHOLD,
    CRACK_ENABLED,
    CRACK_ONNX_PATH,
)

__all__ = [
    "CRACK_CLASS_NAME",
    "CRACK_CONFIDENCE_THRESHOLD",
    "CRACK_ENABLED",
    "CRACK_ONNX_PATH",
    "YOLOCrackSegDetector",
]
