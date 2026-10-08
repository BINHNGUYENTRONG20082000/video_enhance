"""Frame pipeline shared by the CLI and Gradio app.

Restormer supplies real denoising, deraining, motion deblurring, and defocus deblurring.
Enlight uses EnlightenGAN. Super resolution stays empty.
"""

from __future__ import annotations

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

ProgressCallback = Callable[[int, int | None], None]


class Pipeline:
    def __init__(self, stages: list[Stage] | None = None) -> None:
        self.stages = list(stages or [])
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        for stage in self.stages:
            stage.load()
        self._loaded = True

    def close(self) -> None:
        for stage in self.stages:
            stage.close()
        self._loaded = False

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        self.load()
        current = frame_rgb
        for stage in self.stages:
            current = stage.process(current)
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
        self.load()
        decoder = open_decoder(source)
        decoder_err, decoder_thread = drain_stderr(decoder)
        encoder = None
        encoder_err: list[bytes] = []
        encoder_thread: threading.Thread | None = None
        written = 0
        abort_decoder = False
        try:
            assert decoder.stdout is not None
            while True:
                frame = read_raw_frame(decoder.stdout, info.width, info.height)
                if frame is None:
                    break
                processed = pad_even(self.process_frame(frame))
                if encoder is None:
                    out_h, out_w = processed.shape[:2]
                    encoder = open_encoder(
                        destination,
                        source,
                        out_w,
                        out_h,
                        info.fps_expr,
                        info.has_audio,
                    )
                    encoder_err, encoder_thread = drain_stderr(encoder)
                assert encoder.stdin is not None
                try:
                    encoder.stdin.write(np.ascontiguousarray(processed).tobytes())
                except BrokenPipeError:
                    abort_decoder = True
                    break
                written += 1
                if progress is not None:
                    progress(written, info.frame_count)
        except Exception:
            abort_decoder = True
            raise
        finally:
            if encoder is not None and encoder.stdin is not None:
                try:
                    encoder.stdin.close()
                except BrokenPipeError:
                    pass
            if (abort_decoder or sys.exc_info()[0] is not None) and decoder.poll() is None:
                decoder.kill()
            decoder.wait()
            decoder_thread.join()
            if encoder is not None:
                encoder.wait()
                if encoder_thread is not None:
                    encoder_thread.join()
        if encoder is not None:
            raise_if_failed(encoder, encoder_err, "ffmpeg encoder")
        if not abort_decoder:
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
    return Pipeline(stages)
