"""Frame pipeline shared by the CLI and Gradio app.

Restormer supplies real denoising, deraining, motion deblurring, and defocus deblurring.
Enlight uses EnlightenGAN. Super resolution uses HAT-L x4 or Real HAT GAN x4.
"""

from __future__ import annotations

import gc
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from video_enhance.io_ffmpeg import (
    FFmpegError,
    drain_stderr,
    open_decoder,
    open_encoder,
    pad_even,
    probe,
    raise_if_failed,
    read_frame_at,
    read_raw_frame,
)
from video_enhance.io_image import read_image, write_image
from video_enhance.media import MediaTypeError, media_kind
from video_enhance.stages.base import Stage
from video_enhance.stages.enlighten import EnlightenStage
from video_enhance.stages.hat import HatStage, model_by_key

ProgressCallback = Callable[[int, int | None], None]


class Pipeline:
    def __init__(self, stages: list[Stage] | None = None) -> None:
        self.stages = list(stages or [])

    def load(self) -> None:
        """Stages load one at a time inside processing, so VRAM holds a single model."""

    def close(self) -> None:
        for stage in self.stages:
            stage.close()
        _release_device()

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        current = frame_rgb
        for stage in self.stages:
            stage.load()
            try:
                current = stage.process(current)
            finally:
                stage.close()
                _release_device()
        return current

    def preview_frame(self, source: Path, frame_index: int = 0) -> np.ndarray:
        source = Path(source)
        kind = media_kind(source)
        if kind == "image":
            if frame_index != 0:
                raise MediaTypeError("Images only have frame 0.")
            return self.process_frame(read_image(source))
        return self.process_frame(read_frame_at(source, frame_index))

    def run_image(self, source: Path, destination: Path) -> None:
        source = Path(source)
        destination = Path(destination)
        if media_kind(source) != "image" or media_kind(destination) != "image":
            raise MediaTypeError("Image input and output must both be image files.")
        if source.resolve() == destination.resolve():
            raise OSError("Output path must be different from the input path")
        write_image(destination, self.process_frame(read_image(source)))

    def run_video(self, source: Path, destination: Path, progress: ProgressCallback | None = None) -> None:
        source = Path(source)
        destination = Path(destination)
        if media_kind(source) != "video" or media_kind(destination) != "video":
            raise MediaTypeError("Video input and output must both be video files.")
        if source.resolve() == destination.resolve():
            raise FFmpegError("Output path must be different from the input path")
        info = probe(source)
        passes: list[Stage | None] = list(self.stages) or [None]
        done = 0
        total = info.frame_count * len(passes) if info.frame_count is not None else None
        current = source
        temps: list[Path] = []

        def _tick() -> None:
            nonlocal done
            done += 1
            if progress is not None:
                progress(done, total)

        try:
            for index, stage in enumerate(passes):
                last = index == len(passes) - 1
                target = destination
                if not last:
                    target = destination.with_name(f".{destination.stem}.pass{index}{destination.suffix}")
                    temps.append(target)
                self._write_pass(
                    current,
                    target,
                    audio_from=source,
                    copy_audio=last and info.has_audio,
                    lossless=not last,
                    stage=stage,
                    on_frame=_tick,
                )
                current = target
        finally:
            for path in temps:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _write_pass(
        self,
        source: Path,
        destination: Path,
        audio_from: Path,
        copy_audio: bool,
        lossless: bool,
        stage: Stage | None,
        on_frame: Callable[[], None],
    ) -> None:
        decoder = None
        decoder_err: list[bytes] = []
        decoder_thread: threading.Thread | None = None
        encoder = None
        encoder_err: list[bytes] = []
        encoder_thread: threading.Thread | None = None
        written = 0
        abort_decoder = False
        try:
            info = probe(source)
            if stage is not None:
                stage.load()
            decoder = open_decoder(source)
            decoder_err, decoder_thread = drain_stderr(decoder)
            assert decoder.stdout is not None
            while True:
                frame = read_raw_frame(decoder.stdout, info.width, info.height)
                if frame is None:
                    break
                processed = frame if stage is None else stage.process(frame)
                processed = pad_even(processed)
                if encoder is None:
                    out_h, out_w = processed.shape[:2]
                    encoder = open_encoder(
                        destination,
                        audio_from,
                        out_w,
                        out_h,
                        info.fps_expr,
                        copy_audio,
                        lossless=lossless,
                    )
                    encoder_err, encoder_thread = drain_stderr(encoder)
                assert encoder.stdin is not None
                try:
                    encoder.stdin.write(np.ascontiguousarray(processed).tobytes())
                except BrokenPipeError:
                    abort_decoder = True
                    break
                written += 1
                on_frame()
        except Exception:
            abort_decoder = True
            raise
        finally:
            if encoder is not None and encoder.stdin is not None:
                try:
                    encoder.stdin.close()
                except BrokenPipeError:
                    pass
            if decoder is not None:
                if (abort_decoder or sys.exc_info()[0] is not None) and decoder.poll() is None:
                    decoder.kill()
                decoder.wait()
                if decoder_thread is not None:
                    decoder_thread.join()
            if encoder is not None:
                encoder.wait()
                if encoder_thread is not None:
                    encoder_thread.join()
            if stage is not None:
                stage.close()
            _release_device()
        if encoder is not None:
            raise_if_failed(encoder, encoder_err, "ffmpeg encoder")
        if decoder is not None and not abort_decoder:
            raise_if_failed(decoder, decoder_err, "ffmpeg decoder")
        if written == 0:
            raise FFmpegError(f"No frames decoded from {source}")

    def run(self, source: Path, destination: Path, progress: ProgressCallback | None = None) -> None:
        kind = media_kind(Path(source))
        if kind == "image":
            self.run_image(source, destination)
            return
        self.run_video(source, destination, progress=progress)


def build_pipeline(
    *,
    denoise: bool = False,
    derain: bool = False,
    deblur: bool = False,
    defocus: bool = False,
    enlight: bool = False,
    superres: bool = False,
    sr_model: str = "hat-l",
    device: str = "auto",
    tile: int = 512,
    fp16: bool = False,
) -> Pipeline:
    selected = {"denoise": denoise, "derain": derain, "deblur": deblur, "defocus": defocus}
    stages: list[Stage] = []
    if any(selected.values()):
        from video_enhance.stages.restormer import TASKS, RestormerStage

        for task in TASKS:
            if selected[task.key]:
                stages.append(
                    RestormerStage(task.key, task.weights, task.layer_norm, device, tile, fp16)
                )
    if enlight:
        stages.append(EnlightenStage(device))
    if superres:
        stages.append(HatStage(model_by_key(sr_model), device, tile, fp16))
    return Pipeline(stages)


def _release_device() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
