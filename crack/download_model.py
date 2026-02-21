"""
Download OpenSistemas/YOLOv8-crack-seg from HuggingFace and export to ONNX.

Output: models/yolov8_crack_seg.onnx (relative to repo root).

Usage (from repo root):
    python -m crack.download_model              # default yolov8n
    python -m crack.download_model --variant yolov8s
"""

import argparse
import shutil
import sys
from pathlib import Path

from crack.config import HF_REPO, VARIANT_FILES

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _REPO_ROOT / "models"
_OUTPUT_ONNX = _MODELS_DIR / "yolov8_crack_seg.onnx"


def _ensure_deps():
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "ultralytics", "huggingface_hub", "-q"]
        )


def download_and_export(variant: str = "yolov8n") -> Path:
    """Download the .pt weights from HuggingFace and export to ONNX.

    Returns the path to the exported ONNX file.
    Raises RuntimeError on failure.
    """
    _ensure_deps()
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    hf_filename = VARIANT_FILES.get(variant, VARIANT_FILES["yolov8n"])
    print(f"Downloading {HF_REPO} ({variant}) ...")
    pt_path = Path(
        hf_hub_download(repo_id=HF_REPO, filename=hf_filename)
    )

    print(f"Exporting to ONNX (1-class crack-seg, {640}x{640}) ...")
    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", imgsz=640, dynamic=False, half=False)

    # Ultralytics returns the export path as a string or Path
    exported_path = Path(str(exported)) if exported else None

    if exported_path and exported_path.is_file():
        if exported_path.resolve() != _OUTPUT_ONNX.resolve():
            shutil.copy2(exported_path, _OUTPUT_ONNX)
    else:
        # Fallback: look for common export names
        for name in ("best.onnx", f"{variant}-seg.onnx"):
            candidate = pt_path.parent / name
            if candidate.is_file():
                shutil.copy2(candidate, _OUTPUT_ONNX)
                break

    if not _OUTPUT_ONNX.is_file():
        raise RuntimeError(
            f"Export completed but ONNX file not found at {_OUTPUT_ONNX}. "
            "Check ultralytics output."
        )
    return _OUTPUT_ONNX


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download YOLOv8-crack-seg and export to ONNX"
    )
    parser.add_argument(
        "--variant",
        default="yolov8n",
        choices=list(VARIANT_FILES),
        help="Model size variant (default: yolov8n, fastest for edge/ARM)",
    )
    args = parser.parse_args()

    print(f"YOLOv8 Crack Segmentation — {HF_REPO}")
    print(f"Output: {_OUTPUT_ONNX}\n")

    try:
        path = download_and_export(variant=args.variant)
        print(f"\nModel saved to {path}")
        return 0
    except Exception as exc:
        print(f"\nFailed: {exc}")
        print("Ensure: pip install ultralytics huggingface_hub")
        return 1


if __name__ == "__main__":
    sys.exit(main())
