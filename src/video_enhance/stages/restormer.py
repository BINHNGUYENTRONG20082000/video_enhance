"""Restormer inference using weights stored in this project.

Network code is restormer_arch.py. Checkpoints live in weights/restormer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from video_enhance.stages.restormer_arch import Restormer

WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "weights" / "restormer"
DENOISE_WEIGHTS = WEIGHTS_DIR / "real_denoising.pth"
DERAIN_WEIGHTS = WEIGHTS_DIR / "deraining.pth"
DEBLUR_WEIGHTS = WEIGHTS_DIR / "motion_deblurring.pth"
DEFOCUS_WEIGHTS = WEIGHTS_DIR / "single_image_defocus_deblurring.pth"


@dataclass(frozen=True)
class RestormerTask:
    key: str
    label: str
    weights: Path
    layer_norm: str


# Order is the order they run when several are enabled.
TASKS: tuple[RestormerTask, ...] = (
    RestormerTask("denoise", "Real denoising", DENOISE_WEIGHTS, "BiasFree"),
    RestormerTask("derain", "Deraining", DERAIN_WEIGHTS, "WithBias"),
    RestormerTask("deblur", "Motion deblurring", DEBLUR_WEIGHTS, "WithBias"),
    RestormerTask("defocus", "Defocus deblurring", DEFOCUS_WEIGHTS, "WithBias"),
)


class RestormerStage:
    def __init__(
        self,
        name: str,
        weights: Path,
        layer_norm: str,
        device: str = "auto",
        tile: int = 512,
        fp16: bool = False,
    ) -> None:
        self.name = name
        self.weights = Path(weights)
        self.layer_norm = layer_norm
        self.device_name = device
        self.tile = int(tile)
        self.fp16 = bool(fp16)
        self._model = None
        self._device = torch.device("cpu")

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.weights.is_file():
            raise RuntimeError(f"Không thấy weight Restormer: {self.weights}")
        self._device = _resolve_device(self.device_name)
        model = Restormer(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type=self.layer_norm,
            dual_pixel_task=False,
        )
        checkpoint = torch.load(self.weights, map_location="cpu", weights_only=False)
        state = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
        model.load_state_dict(state)
        model.eval()
        use_half = self.fp16 and self._device.type == "cuda"
        if use_half:
            model.half()
        self._model = model.to(self._device)
        self.fp16 = use_half

    def process(self, frame_rgb: np.ndarray) -> np.ndarray:
        self.load()
        tensor = torch.from_numpy(np.ascontiguousarray(frame_rgb)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self._device, dtype=torch.float16 if self.fp16 else torch.float32).div_(255)
        height, width = tensor.shape[2], tensor.shape[3]
        tensor = _pad_multiple(tensor, 8)
        with torch.inference_mode():
            restored = _tiled_forward(self._model, tensor, self.tile)
        restored = restored[:, :, :height, :width].clamp(0, 1)
        array = restored.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return np.clip(np.round(array * 255), 0, 255).astype(np.uint8)

    def close(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không dùng được. Chọn thiết bị CPU.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pad_multiple(tensor: torch.Tensor, multiple: int) -> torch.Tensor:
    height, width = tensor.shape[2], tensor.shape[3]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")


def _tiled_forward(model: torch.nn.Module, tensor: torch.Tensor, tile: int, overlap: int = 32) -> torch.Tensor:
    _, _, height, width = tensor.shape
    if tile <= 0 or (tile >= height and tile >= width):
        return model(tensor)
    tile = max(8, int(np.ceil(min(tile, height, width) / 8) * 8))
    tile = min(tile, height, width)
    overlap = min(overlap, tile // 4)
    stride = tile - overlap
    rows = list(range(0, height - tile, stride)) + [height - tile]
    cols = list(range(0, width - tile, stride)) + [width - tile]
    accumulator = torch.zeros_like(tensor)
    weights = torch.zeros_like(tensor)
    for top in rows:
        for left in cols:
            patch = tensor[..., top : top + tile, left : left + tile]
            output = model(patch)
            accumulator[..., top : top + tile, left : left + tile].add_(output)
            weights[..., top : top + tile, left : left + tile].add_(1)
    return accumulator.div_(weights)
