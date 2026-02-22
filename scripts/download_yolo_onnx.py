"""
Download a pre-exported YOLOv8n ONNX model (no Hugging Face, no ultralytics needed).
Saves to repo models/yolov8_det.onnx for use with backend.perception.YOLODetector.
Run from repo root: python scripts/download_yolo_onnx.py
"""

import os
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(REPO_ROOT, "models")
OUTPUT_PATH = os.path.join(MODELS_DIR, "yolov8_det.onnx")

# Direct URLs for YOLOv8n ONNX (no auth). First that works is used.
YOLO_ONNX_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
]


def download(url: str, path: str) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                return False
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(resp.read())
        return True
    except Exception as e:
        print(f"  {url}: {e}")
        return False


def main():
    if os.path.isfile(OUTPUT_PATH):
        print(f"ONNX model already exists: {OUTPUT_PATH}")
        return 0

    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Downloading YOLOv8n ONNX model...")

    for url in YOLO_ONNX_URLS:
        print(f"  Trying {url}")
        if download(url, OUTPUT_PATH):
            print(f"Saved to {OUTPUT_PATH}")
            return 0
        if os.path.isfile(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)

    print("")
    print("Could not download ONNX from direct URLs.")
    print("To build the model yourself (requires Windows Long Paths enabled):")
    print("  1. Enable Long Paths: run PowerShell as Administrator:")
    print('     New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force')
    print("  2. Reboot, then: python -m pip install ultralytics")
    print("  3. python scripts/download_model.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
