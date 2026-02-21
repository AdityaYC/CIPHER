#!/usr/bin/env python3
"""
Download YOLOv8-crack-seg from HuggingFace and export to ONNX.

Thin wrapper around crack.download_model — run from repo root:
    python scripts/download_model_crack_seg.py [--variant yolov8n]

Requires: pip install ultralytics huggingface_hub
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `crack` package resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from crack.download_model import main

if __name__ == "__main__":
    sys.exit(main())
