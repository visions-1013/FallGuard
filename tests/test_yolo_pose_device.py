from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import numpy as np

from fallguard.pose import yolo_pose
from fallguard.pose.yolo_pose import PoseDetector, select_inference_device


class FakeCuda:
    def __init__(self, available: bool, name: str = "Test GPU") -> None:
        self.available = available
        self.name = name

    def is_available(self) -> bool:
        return self.available

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return self.name


class BrokenCuda:
    def is_available(self) -> bool:
        raise RuntimeError("cuda probe failed")


def install_fake_torch(monkeypatch, cuda: object) -> None:
    torch = ModuleType("torch")
    torch.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_select_inference_device_uses_cuda_when_available(monkeypatch):
    install_fake_torch(monkeypatch, FakeCuda(available=True))

    assert select_inference_device() == "cuda:0"


def test_select_inference_device_uses_cpu_when_cuda_is_unavailable(monkeypatch):
    install_fake_torch(monkeypatch, FakeCuda(available=False))

    assert select_inference_device() == "cpu"


def test_select_inference_device_falls_back_to_cpu_when_cuda_probe_fails(monkeypatch):
    install_fake_torch(monkeypatch, BrokenCuda())

    assert select_inference_device() == "cpu"


def test_select_inference_device_falls_back_to_cpu_when_torch_is_missing(monkeypatch):
    real_import_module = yolo_pose.importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "torch":
            raise ImportError("torch unavailable")
        return real_import_module(name, package)

    monkeypatch.setattr(yolo_pose.importlib, "import_module", fake_import_module)

    assert select_inference_device() == "cpu"


def test_pose_detector_prints_selected_cpu_device(monkeypatch, capsys):
    install_fake_torch(monkeypatch, FakeCuda(available=False))
    install_fake_ultralytics(monkeypatch, FakeYolo)

    detector = PoseDetector()

    assert detector.device == "cpu"
    assert "FallGuard 推理设备: CPU cpu" in capsys.readouterr().out


def test_pose_detector_prints_selected_gpu_device(monkeypatch, capsys):
    install_fake_torch(monkeypatch, FakeCuda(available=True, name="Test GPU"))
    install_fake_ultralytics(monkeypatch, FakeYolo)

    detector = PoseDetector()

    assert detector.device == "cuda:0"
    assert "FallGuard 推理设备: GPU cuda:0 (Test GPU)" in capsys.readouterr().out


def test_pose_detector_passes_selected_device_to_yolo(monkeypatch):
    install_fake_torch(monkeypatch, FakeCuda(available=False))
    install_fake_ultralytics(monkeypatch, FakeYolo)
    detector = PoseDetector(device="cuda:0")

    detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

    assert detector._model.calls[-1]["device"] == "cuda:0"


def install_fake_ultralytics(monkeypatch, yolo_class: type) -> None:
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = yolo_class
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)


class FakeYolo:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[dict[str, object]] = []

    def __call__(self, frame: np.ndarray, *, verbose: bool, device: str):
        self.calls.append({"frame": frame, "verbose": verbose, "device": device})
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=FakeTensor(np.array([[1.0, 2.0, 3.0, 4.0]])),
                    conf=FakeTensor(np.array([0.9])),
                ),
                keypoints=SimpleNamespace(
                    xy=FakeTensor(np.zeros((1, 17, 2), dtype=float)),
                    conf=FakeTensor(np.ones((1, 17), dtype=float)),
                ),
            )
        ]


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values
