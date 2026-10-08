"""Enlight stage using the EnlightenGAN ONNX model.

Preprocessing matches arsenyinfo/EnlightenGAN-inference: pad to a multiple of 16,
scale to [0, 1], then map the network output from [-1, 1] back to uint8.
https://github.com/arsenyinfo/EnlightenGAN-inference
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

WEIGHTS_URL = (
    "https://github.com/arsenyinfo/EnlightenGAN-inference/raw/main/enlighten_inference/enlighten.onnx"
)
WEIGHTS_NAME = "enlighten.onnx"
MIN_BYTES = 30_000_000


class EnlightenStage:
    name = "enlight"

    def __init__(self, device: str = "auto", weights_dir: Path | None = None) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir) if weights_dir is not None else _default_weights_dir()
        self._session = None

    def load(self) -> None:
        if self._session is not None:
            return
        _register_nvidia_dlls()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Thiếu onnxruntime. Cài onnxruntime-gpu để chạy EnlightenGAN.") from exc
        model_path = ensure_weights(self.weights_dir)
        available = set(ort.get_available_providers())
        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif self.device == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDA không có trong onnxruntime. Cài onnxruntime-gpu hoặc chọn thiết bị CPU."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(model_path), providers=providers)

    def process(self, frame_rgb: np.ndarray) -> np.ndarray:
        self.load()
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        enlightened = _predict(self._session, np.ascontiguousarray(bgr))
        return cv2.cvtColor(np.ascontiguousarray(enlightened), cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._session = None


def ensure_weights(weights_dir: Path) -> Path:
    destination = weights_dir / WEIGHTS_NAME
    if destination.is_file() and destination.stat().st_size >= MIN_BYTES:
        return destination
    weights_dir.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "video-enhance/0.1"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        if partial.stat().st_size < MIN_BYTES:
            raise RuntimeError(f"File EnlightenGAN tải về quá nhỏ: {partial}")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _predict(session, image_bgr: np.ndarray) -> np.ndarray:
    padded, (height, width) = _pad(image_bgr)
    image_numpy = session.run(["output"], {"input": _preprocess(padded)})[0]
    image_numpy = (np.transpose(image_numpy[0], (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype("uint8")[:height, :width, :]


def _pad(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width, _ = image.shape
    block = 16
    padded_height = (height // block + 1) * block
    padded_width = (width // block + 1) * block
    padded = np.pad(
        image,
        ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return padded, (height, width)


def _preprocess(image: np.ndarray) -> np.ndarray:
    if len(image.shape) != 3:
        raise ValueError(f"Incorrect shape: expected 3, got {len(image.shape)}")
    return np.expand_dims(np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0, 0)


def _default_weights_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "weights"


def _register_nvidia_dlls() -> None:
    if sys.platform != "win32":
        return
    site = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not site.is_dir():
        return
    for folder in site.glob("*/bin"):
        if not folder.is_dir():
            continue
        try:
            os.add_dll_directory(str(folder))
        except OSError:
            continue
        os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
