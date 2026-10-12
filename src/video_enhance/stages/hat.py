"""HAT x4 super-resolution using weights stored in this project.

The network lives in hat_arch.py. HAT-L is the largest official classical model.
Real HAT GAN is the official real-world GAN model. Both upscale by 4.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from video_enhance.stages.hat_arch import HAT

SCALE = 4
WINDOW = 16
TILE_PAD = 32
WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "weights" / "hat"

_SHARED = dict(
    upscale=SCALE,
    in_chans=3,
    img_size=64,
    window_size=WINDOW,
    compress_ratio=3,
    squeeze_factor=30,
    conv_scale=0.01,
    overlap_ratio=0.5,
    img_range=1.0,
    embed_dim=180,
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv",
)


@dataclass(frozen=True)
class HatModel:
    key: str
    label: str
    weights_name: str
    url: str
    sha256: str
    min_bytes: int
    depths: tuple[int, ...]
    num_heads: tuple[int, ...]


# Official configs from XPixelGroup/HAT options/test.
MODELS: tuple[HatModel, ...] = (
    HatModel(
        "hat-l",
        "HAT-L x4",
        "HAT-L_SRx4_ImageNet-pretrain.pth",
        "https://huggingface.co/jaideepsingh/upscale_models/resolve/main/HAT/HAT-L_SRx4_ImageNet-pretrain.pth",
        "5992bd38522f2b8faf11ea4bd8ee08de92465bb66892166576999afc36d60043",
        150_000_000,
        (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    ),
    HatModel(
        "real-hat-gan",
        "Real HAT GAN x4",
        "Real_HAT_GAN_SRx4.pth",
        "https://huggingface.co/jaideepsingh/upscale_models/resolve/main/HAT/Real_HAT_GAN_SRx4.pth",
        "f5b1e3bbbb05147ca2beefcc715279cb647d7976cbda67d62ea7e6e20d5ffcc7",
        150_000_000,
        (6, 6, 6, 6, 6, 6),
        (6, 6, 6, 6, 6, 6),
    ),
)


class HatStage:
    def __init__(self, model: HatModel, device: str = "auto", tile: int = 512, fp16: bool = False) -> None:
        self.model = model
        self.name = model.key
        self.scale = SCALE
        self.device_name = device
        self.tile = int(tile)
        self.fp16 = bool(fp16)
        self._dtype = torch.float32
        self._network = None
        self._device = torch.device("cpu")

    def load(self) -> None:
        if self._network is not None:
            return
        weights = ensure_weights(self.model)
        self._device = _resolve_device(self.device_name)
        network = build_network(self.model)
        _load_weights(network, weights)
        network.eval()
        # FP16 overflows HAT (activations exceed 65504). BF16 keeps the range and the memory saving.
        self._dtype = torch.float32
        if self.fp16 and self._device.type == "cuda" and torch.cuda.is_bf16_supported():
            network.bfloat16()
            self._dtype = torch.bfloat16
        self._network = network.to(self._device)
        self.fp16 = self._dtype == torch.bfloat16

    def process(self, frame_rgb: np.ndarray) -> np.ndarray:
        self.load()
        tensor = torch.from_numpy(np.ascontiguousarray(frame_rgb)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self._device, dtype=self._dtype).div_(255)
        with torch.inference_mode():
            restored = _tiled_forward(self._network, tensor, self.tile)
        restored = restored.clamp(0, 1)
        array = restored.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return np.clip(np.round(array * 255), 0, 255).astype(np.uint8)

    def close(self) -> None:
        self._network = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def model_by_key(key: str) -> HatModel:
    for model in MODELS:
        if model.key == key:
            return model
    labels = ", ".join(model.label for model in MODELS)
    raise ValueError(f"Model siêu phân giải phải là một trong: {labels}.")


def build_network(model: HatModel) -> HAT:
    return HAT(depths=model.depths, num_heads=model.num_heads, **_SHARED)


def ensure_weights(model: HatModel, weights_dir: Path | None = None) -> Path:
    folder = Path(weights_dir) if weights_dir is not None else WEIGHTS_DIR
    destination = folder / model.weights_name
    if destination.is_file() and destination.stat().st_size >= model.min_bytes:
        return destination
    folder.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(model.url, headers={"User-Agent": "video-enhance/0.1"})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != model.sha256 or partial.stat().st_size < model.min_bytes:
            raise RuntimeError(f"File weight {model.label} tải về không đúng.")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _load_weights(network: HAT, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                state = nested
                break
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    missing, unexpected = network.load_state_dict(cleaned, strict=False)
    missing = [name for name in missing if not name.startswith("relative_position_index_")]
    if missing or unexpected:
        raise RuntimeError(
            f"Weight {path.name} không khớp mạng HAT. Thiếu {missing[:6]}, thừa {list(unexpected)[:6]}."
        )


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không dùng được. Chọn thiết bị CPU.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiled_forward(network: HAT, tensor: torch.Tensor, tile: int) -> torch.Tensor:
    _, _, height, width = tensor.shape
    if tile <= 0 or (tile >= height and tile >= width):
        return _forward_padded(network, tensor)
    tile = max(WINDOW, int(round(min(tile, height, width) / WINDOW) * WINDOW))
    tile = min(tile, max(WINDOW, height), max(WINDOW, width))
    output = torch.empty(
        (tensor.shape[0], tensor.shape[1], height * SCALE, width * SCALE),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    for top in range(0, height, tile):
        for left in range(0, width, tile):
            bottom = min(top + tile, height)
            right = min(left + tile, width)
            top_pad = max(top - TILE_PAD, 0)
            left_pad = max(left - TILE_PAD, 0)
            bottom_pad = min(bottom + TILE_PAD, height)
            right_pad = min(right + TILE_PAD, width)
            patch = tensor[:, :, top_pad:bottom_pad, left_pad:right_pad]
            restored = _forward_padded(network, patch)
            y0 = (top - top_pad) * SCALE
            x0 = (left - left_pad) * SCALE
            y1 = y0 + (bottom - top) * SCALE
            x1 = x0 + (right - left) * SCALE
            output[:, :, top * SCALE : bottom * SCALE, left * SCALE : right * SCALE] = restored[:, :, y0:y1, x0:x1]
    return output


def _forward_padded(network: HAT, tensor: torch.Tensor) -> torch.Tensor:
    _, _, height, width = tensor.shape
    pad_h = (WINDOW - height % WINDOW) % WINDOW
    pad_w = (WINDOW - width % WINDOW) % WINDOW
    working = tensor
    if pad_h or pad_w:
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        working = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    output = network(working)
    return output[:, :, : height * SCALE, : width * SCALE]
