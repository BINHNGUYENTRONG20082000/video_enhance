"""Public entry point for embedding the models in another program."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from video_enhance.paths import resolve_weights_dir
from video_enhance.pipeline import ProgressCallback, build_pipeline
from video_enhance.stages.hat import model_by_key

_DEVICES = {"auto", "cuda", "cpu"}


class Enhancer:
    """Run one or more restoration models on an image or a video.

    Each selected model is loaded, used, and released before the next one starts.
    ``device``, ``tile`` and ``fp16`` stay on the object. The tools are chosen per call.
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        tile: int = 512,
        fp16: bool = True,
        weights_dir: str | Path | None = None,
    ) -> None:
        if device not in _DEVICES:
            raise ValueError("Thiết bị phải là auto, cuda hoặc cpu.")
        if int(tile) < 0:
            raise ValueError("Tile phải lớn hơn hoặc bằng 0.")
        self.device = device
        self.tile = int(tile)
        self.fp16 = bool(fp16)
        self.weights_dir = resolve_weights_dir(weights_dir)

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        denoise: bool = False,
        derain: bool = False,
        deblur: bool = False,
        defocus: bool = False,
        enlight: bool = False,
        superres: bool = False,
        sr_model: str = "hat-l",
    ) -> np.ndarray:
        _require_rgb(frame)
        pipeline = self._pipeline(
            denoise=denoise,
            derain=derain,
            deblur=deblur,
            defocus=defocus,
            enlight=enlight,
            superres=superres,
            sr_model=sr_model,
        )
        try:
            return pipeline.process_frame(frame)
        finally:
            pipeline.close()

    def preview(
        self,
        source: str | Path,
        frame: int = 0,
        *,
        denoise: bool = False,
        derain: bool = False,
        deblur: bool = False,
        defocus: bool = False,
        enlight: bool = False,
        superres: bool = False,
        sr_model: str = "hat-l",
    ) -> np.ndarray:
        pipeline = self._pipeline(
            denoise=denoise,
            derain=derain,
            deblur=deblur,
            defocus=defocus,
            enlight=enlight,
            superres=superres,
            sr_model=sr_model,
        )
        try:
            return pipeline.preview_frame(Path(source), frame)
        finally:
            pipeline.close()

    def run(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        denoise: bool = False,
        derain: bool = False,
        deblur: bool = False,
        defocus: bool = False,
        enlight: bool = False,
        superres: bool = False,
        sr_model: str = "hat-l",
        progress: ProgressCallback | None = None,
    ) -> None:
        pipeline = self._pipeline(
            denoise=denoise,
            derain=derain,
            deblur=deblur,
            defocus=defocus,
            enlight=enlight,
            superres=superres,
            sr_model=sr_model,
        )
        try:
            pipeline.run(Path(source), Path(destination), progress=progress)
        finally:
            pipeline.close()

    def __enter__(self) -> Enhancer:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    def _pipeline(self, **tools: bool | str):
        sr_model = str(tools.pop("sr_model"))
        if bool(tools.get("superres")):
            model_by_key(sr_model)
        return build_pipeline(
            denoise=bool(tools["denoise"]),
            derain=bool(tools["derain"]),
            deblur=bool(tools["deblur"]),
            defocus=bool(tools["defocus"]),
            enlight=bool(tools["enlight"]),
            superres=bool(tools["superres"]),
            sr_model=sr_model,
            device=self.device,
            tile=self.tile,
            fp16=self.fp16,
            weights_dir=self.weights_dir,
        )


def _require_rgb(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Ảnh phải là mảng RGB uint8 dạng (cao, rộng, 3).")
