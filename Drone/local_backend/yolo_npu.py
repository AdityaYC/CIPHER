"""YOLO inference optimized for Qualcomm Snapdragon X Elite NPU.

Uses Qualcomm AI Hub Models for hardware-accelerated inference on the
Hexagon NPU (45 TOPS on Snapdragon X Elite/Plus).

Installation:
    pip install qai_hub_models
    pip install "qai_hub_models[yolov8]"
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import time

# Resolve ONNX model relative to project root
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_DEFAULT_ONNX_PATH = _REPO_ROOT / "models" / "yolov8_det.onnx"


def _find_qnn_dll() -> Optional[str]:
    """Locate QnnHtp.dll from onnxruntime install or Qualcomm SDK."""
    try:
        import onnxruntime as ort
        candidate = Path(ort.__file__).parent / "capi" / "QnnHtp.dll"
        if candidate.exists():
            return str(candidate)
    except ImportError:
        pass
    # Qualcomm QAIRT SDK
    import os
    for sdk_root in [Path(r"C:\Qualcomm\AIStack\QAIRT"), Path.home() / "Qualcomm" / "AIStack" / "QAIRT"]:
        if sdk_root.is_dir():
            for dll in sorted(sdk_root.rglob("QnnHtp.dll"), reverse=True):
                return str(dll)
    env = os.environ.get("QNN_DLL_PATH", "").strip()
    if env and Path(env).exists():
        return env
    return None


class YOLONPUInference:
    """YOLO inference using Qualcomm NPU acceleration.

    Supports multiple backends:
    1. Qualcomm AI Hub (NPU) - Fastest on Snapdragon X Elite
    2. ONNX Runtime with QNN - NPU acceleration via ONNX
    3. ONNX Runtime CPU (optimized) - Threaded CPU fallback
    4. PyTorch (fallback) - Standard Ultralytics
    """

    def __init__(self, model_name: str = "yolov8_det", use_npu: bool = True,
                 onnx_path: Optional[str] = None):
        """Initialize YOLO with NPU acceleration.

        Args:
            model_name: Model variant (yolov8_det, yolov8n_det, yolov8s_det)
            use_npu: Try to use NPU if available
            onnx_path: Explicit path to ONNX model file (auto-detected if None)
        """
        self.model_name = model_name
        self.use_npu = use_npu
        self.model = None
        self.backend = None
        self._session = None
        self._last_latency_ms: float = 0.0
        self.class_names = self._get_coco_names()
        self._onnx_path = onnx_path or str(_DEFAULT_ONNX_PATH)

        self._load_model()

    def _load_model(self):
        """Load YOLO model with best available backend."""

        # Try Qualcomm AI Hub first (best for NPU)
        if self.use_npu:
            try:
                print(f"  YOLO NPU: loading {self.model_name} with Qualcomm AI Hub...")
                from qai_hub_models.models.yolov8_det import Model as YOLOv8Model

                self.model = YOLOv8Model.from_pretrained()
                self.backend = "qualcomm_npu"
                print(f"  YOLO NPU: Qualcomm NPU (Hexagon) active")
                return
            except ImportError:
                print("  YOLO NPU: qai_hub_models not installed (pip install qai_hub_models)")
            except Exception as e:
                print(f"  YOLO NPU: Qualcomm AI Hub unavailable: {e}")

        # Try ONNX Runtime with QNN execution provider
        if self.use_npu:
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                if 'QNNExecutionProvider' in available_providers:
                    if self._load_onnx_qnn():
                        return
                else:
                    print(f"  YOLO NPU: QNN provider not in available: {available_providers}")
            except ImportError:
                print("  YOLO NPU: onnxruntime not installed")
            except Exception as e:
                print(f"  YOLO NPU: ONNX+QNN failed: {e}")

        # Try ONNX Runtime CPU (optimized with threading)
        try:
            if self._load_onnx_cpu():
                return
        except Exception as e:
            print(f"  YOLO NPU: ONNX CPU failed: {e}")

        # Fallback to standard Ultralytics YOLO (CPU/GPU)
        print("  YOLO NPU: falling back to Ultralytics YOLO (CPU/GPU)...")
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            self.backend = "pytorch_cpu"
            print("  YOLO NPU: loaded on CPU/GPU (no NPU acceleration)")
        except Exception as e:
            raise RuntimeError(f"Failed to load any YOLO backend: {e}")

    def _find_onnx_model(self) -> Optional[str]:
        """Resolve ONNX model path."""
        path = Path(self._onnx_path)
        if path.is_file():
            return str(path)
        # Search common locations
        for candidate in [
            _REPO_ROOT / "models" / "yolov8_det.onnx",
            _REPO_ROOT / "models" / "yolov8n.onnx",
            Path.cwd() / "models" / "yolov8_det.onnx",
            Path.cwd() / "yolov8n.onnx",
        ]:
            if candidate.is_file():
                return str(candidate)
        return None

    def _load_onnx_qnn(self) -> bool:
        """Load YOLO via ONNX Runtime with QNN execution provider."""
        import onnxruntime as ort

        onnx_path = self._find_onnx_model()
        if onnx_path is None:
            print("  YOLO NPU: no ONNX model file found")
            return False

        qnn_dll = _find_qnn_dll()
        if qnn_dll is None:
            print("  YOLO NPU: QnnHtp.dll not found")
            return False

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [
            ('QNNExecutionProvider', {
                'backend_path': qnn_dll,
                'profiling_level': 'basic',
            }),
            'CPUExecutionProvider'
        ]

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=providers,
        )

        active = self._session.get_providers()
        if "QNNExecutionProvider" in active:
            self.backend = "onnx_qnn"
            print(f"  YOLO NPU: ONNX + QNN (NPU) active — {onnx_path}")
            return True
        else:
            print(f"  YOLO NPU: QNN requested but got {active} — falling through")
            self._session = None
            return False

    def _load_onnx_cpu(self) -> bool:
        """Load YOLO via ONNX Runtime with optimized CPU threading."""
        import onnxruntime as ort
        import os

        onnx_path = self._find_onnx_model()
        if onnx_path is None:
            return False

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Use all available cores for CPU inference
        cpu_count = os.cpu_count() or 4
        session_options.intra_op_num_threads = cpu_count
        session_options.inter_op_num_threads = max(1, cpu_count // 2)
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider'],
        )
        self.backend = "onnx_cpu"
        print(f"  YOLO NPU: ONNX CPU ({cpu_count} threads) — {onnx_path}")
        return True

    def predict(self, image: Image.Image, conf_threshold: float = 0.25) -> List[Dict]:
        """Run inference on a PIL Image."""
        start_time = time.time()

        if self.backend == "qualcomm_npu":
            detections = self._predict_qualcomm(image, conf_threshold)
        elif self.backend in ("onnx_qnn", "onnx_cpu"):
            detections = self._predict_onnx(image, conf_threshold)
        else:
            detections = self._predict_pytorch(image, conf_threshold)

        self._last_latency_ms = (time.time() - start_time) * 1000

        if detections:
            detections[0]["inference_time_ms"] = round(self._last_latency_ms, 2)

        return detections

    def detect(self, frame) -> List[Dict]:
        """Run inference on a BGR numpy frame (same interface as perception.YOLODetector).

        Returns list of dicts: {class, confidence, bbox: [x1,y1,x2,y2], center: [cx,cy]}
        """
        start_time = time.time()
        h_orig, w_orig = frame.shape[:2]

        if self.backend in ("onnx_qnn", "onnx_cpu") and self._session is not None:
            detections = self._detect_onnx_raw(frame, 0.25)
        elif self.backend == "qualcomm_npu":
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            detections = self._predict_qualcomm(image, 0.25)
            # Convert normalized bbox to pixel coords
            for d in detections:
                b = d["bbox"]
                d["bbox"] = [b[0] * w_orig, b[1] * h_orig, b[2] * w_orig, b[3] * h_orig]
                d["center"] = [(b[0] + b[2]) / 2 * w_orig, (b[1] + b[3]) / 2 * h_orig]
                d["class"] = d.pop("class_name", d.get("class", "object"))
        else:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            raw = self._predict_pytorch(image, 0.25)
            detections = []
            for d in raw:
                b = d["bbox"]
                # Ultralytics xyxyn -> pixel coords
                bbox = [b[0] * w_orig, b[1] * h_orig, b[2] * w_orig, b[3] * h_orig]
                detections.append({
                    "class": d.get("class_name", d.get("class", "object")),
                    "confidence": d["confidence"],
                    "bbox": bbox,
                    "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                })

        self._last_latency_ms = (time.time() - start_time) * 1000
        return detections

    def _detect_onnx_raw(self, frame, conf_threshold: float) -> List[Dict]:
        """ONNX inference directly on BGR numpy frame, returns pixel-coord detections."""
        import cv2
        h_orig, w_orig = frame.shape[:2]
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        inp = np.expand_dims(img, axis=0)

        input_name = self._session.get_inputs()[0].name
        out = self._session.run(None, {input_name: inp})

        raw = out[0]
        if raw.ndim == 3:
            raw = raw[0]
        # YOLOv8 output: (84, 8400) or (8400, 84)
        if raw.shape[0] == 84 and raw.shape[1] == 8400:
            raw = np.transpose(raw, (1, 0))

        scale_x = w_orig / 640
        scale_y = h_orig / 640
        detections = []
        for i in range(raw.shape[0]):
            cx, cy, w, h = raw[i, :4]
            probs = raw[i, 4:84]
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            if confidence < conf_threshold:
                continue
            cx_s, cy_s = cx * scale_x, cy * scale_y
            w_s, h_s = w * scale_x, h * scale_y
            x1, y1 = cx_s - w_s / 2, cy_s - h_s / 2
            x2, y2 = cx_s + w_s / 2, cy_s + h_s / 2
            detections.append({
                "class": self.class_names[class_id] if class_id < len(self.class_names) else "object",
                "confidence": confidence,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": [float(cx_s), float(cy_s)],
            })

        return self._nms(detections, iou_threshold=0.5)

    def _predict_qualcomm(self, image: Image.Image, conf_threshold: float) -> List[Dict]:
        """Predict using Qualcomm AI Hub model."""
        import torch

        img_resized = image.resize((640, 640))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        input_tensor = torch.from_numpy(img_array)

        with torch.no_grad():
            output = self.model(input_tensor)

        detections = []

        if isinstance(output, (list, tuple)):
            if len(output) >= 3:
                boxes = output[0].cpu().numpy() if isinstance(output[0], torch.Tensor) else output[0]
                scores = output[1].cpu().numpy() if isinstance(output[1], torch.Tensor) else output[1]
                classes = output[2].cpu().numpy() if isinstance(output[2], torch.Tensor) else output[2]

                for i, score in enumerate(scores.flatten()):
                    if score >= conf_threshold:
                        box = boxes[i] if boxes.ndim > 1 else boxes
                        cls = int(classes[i]) if classes.ndim > 0 else int(classes)
                        if len(box) == 4:
                            x1, y1, x2, y2 = box
                            detections.append({
                                "class_name": self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}",
                                "confidence": float(score),
                                "bbox": [float(x1/640), float(y1/640), float(x2/640), float(y2/640)],
                            })
        else:
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()
            if output.ndim >= 2:
                for detection in output[0] if output.ndim == 3 else output:
                    if len(detection) >= 6:
                        x, y, w, h = detection[:4]
                        conf = detection[4]
                        class_scores = detection[5:]
                        if conf >= conf_threshold:
                            cls = np.argmax(class_scores)
                            detections.append({
                                "class_name": self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}",
                                "confidence": float(conf * class_scores[cls]),
                                "bbox": [float((x-w/2)/640), float((y-h/2)/640), float((x+w/2)/640), float((y+h/2)/640)],
                            })

        return detections

    def _predict_onnx(self, image: Image.Image, conf_threshold: float) -> List[Dict]:
        """Predict using ONNX Runtime (QNN or CPU)."""
        input_array = self._preprocess_image(image)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_array})
        return self._postprocess_output(outputs[0], conf_threshold, image.size)

    def _predict_pytorch(self, image: Image.Image, conf_threshold: float) -> List[Dict]:
        """Predict using standard Ultralytics YOLO."""
        results = self.model(image, conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_name": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxyn[0].tolist(),
                })
        return detections

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for YOLO input."""
        img_resized = image.resize((640, 640))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        return np.expand_dims(img_array, axis=0)

    def _postprocess_output(self, output: np.ndarray, conf_threshold: float,
                            original_size: Tuple[int, int]) -> List[Dict]:
        """Post-process YOLO output to detections."""
        detections = []
        if len(output.shape) == 3:
            output = output[0]

        # YOLOv8: (8400, 84) — 4 box + 80 class scores (no objectness)
        if output.shape[-1] == 84 and output.shape[0] == 8400:
            for detection in output:
                class_scores = detection[4:]
                class_id = np.argmax(class_scores)
                confidence = float(class_scores[class_id])
                if confidence < conf_threshold:
                    continue
                cx, cy, w, h = detection[:4]
                detections.append({
                    "class_name": self.class_names[class_id],
                    "confidence": confidence,
                    "bbox": [float((cx-w/2)/640), float((cy-h/2)/640),
                             float((cx+w/2)/640), float((cy+h/2)/640)],
                })
        else:
            # Legacy YOLOv5 format: 4 box + objectness + 80 classes
            for detection in output:
                objectness = detection[4]
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                confidence = objectness * class_scores[class_id]
                if confidence < conf_threshold:
                    continue
                cx, cy, w, h = detection[:4]
                detections.append({
                    "class_name": self.class_names[class_id],
                    "confidence": float(confidence),
                    "bbox": [float((cx-w/2)/640), float((cy-h/2)/640),
                             float((cx+w/2)/640), float((cy+h/2)/640)],
                })

        return detections

    @staticmethod
    def _nms(detections: list, iou_threshold: float = 0.45) -> list:
        """Non-max suppression by bbox IoU."""
        if not detections:
            return []
        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        area = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (area[i] + area[order[1:]] - inter + 1e-8)
            order = order[1:][iou <= iou_threshold]
        return [detections[j] for j in keep]

    @staticmethod
    def _get_coco_names() -> List[str]:
        """Get COCO class names."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def get_backend_info(self) -> Dict:
        """Get information about the current backend."""
        return {
            "backend": self.backend,
            "model": self.model_name,
            "npu_enabled": self.backend in ["qualcomm_npu", "onnx_qnn"],
            "device": "Qualcomm Hexagon NPU" if self.backend == "qualcomm_npu" else
                     "NPU via ONNX/QNN" if self.backend == "onnx_qnn" else
                     "ONNX CPU (optimized)" if self.backend == "onnx_cpu" else "CPU/GPU",
        }

    def get_provider(self) -> str:
        """Return the primary execution provider in use (compat with perception.YOLODetector)."""
        if self._session is not None:
            return self._session.get_providers()[0]
        if self.backend == "qualcomm_npu":
            return "QualcommNPU"
        return "CPUExecutionProvider"

    def get_last_latency(self) -> float:
        """Return last inference time in milliseconds."""
        return self._last_latency_ms


def create_yolo_npu(use_npu: bool = True) -> YOLONPUInference:
    """Create YOLO instance with NPU acceleration if available."""
    return YOLONPUInference(use_npu=use_npu)


if __name__ == "__main__":
    print("=" * 70)
    print("Testing YOLO NPU Inference")
    print("=" * 70)

    yolo = create_yolo_npu(use_npu=True)

    info = yolo.get_backend_info()
    print(f"\n  Backend: {info['backend']}")
    print(f"  Device: {info['device']}")
    print(f"  NPU Enabled: {info['npu_enabled']}")
    print(f"  Provider: {yolo.get_provider()}")

    print("\n  Running test inference...")
    test_image = Image.new('RGB', (640, 640), color='red')
    detections = yolo.predict(test_image)

    print(f"  Inference complete: {len(detections)} detections")
    print(f"  Latency: {yolo.get_last_latency():.1f}ms")

    print("\n" + "=" * 70)
