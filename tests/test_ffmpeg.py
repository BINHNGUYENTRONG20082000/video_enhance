import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from video_enhance.io_ffmpeg import pad_even, probe, read_frame_at
from video_enhance.pipeline import Pipeline

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not on PATH",
)


def _make_clip(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10:duration=0.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=0.4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def test_round_trip_keeps_size_audio_and_frames(tmp_path: Path):
    source = tmp_path / "in.mp4"
    destination = tmp_path / "out.mp4"
    _make_clip(source)
    before = probe(source)
    seen: list[tuple[int, int | None]] = []
    pipeline = Pipeline()
    pipeline.run_video(source, destination, progress=lambda done, total: seen.append((done, total)))
    after = probe(destination)
    assert after.width == before.width == 64
    assert after.height == before.height == 48
    assert after.has_audio
    assert before.frame_count is not None
    assert seen[-1][0] == before.frame_count
    original = read_frame_at(source, 0)
    copied = read_frame_at(destination, 0)
    assert copied.shape == original.shape
    assert float(np.abs(original.astype(np.int16) - copied.astype(np.int16)).mean()) < 20


def test_video_releases_one_stage_before_the_next(tmp_path: Path):
    source = tmp_path / "in.mp4"
    destination = tmp_path / "out.mp4"
    _make_clip(source)
    log: list[str] = []

    class _Stage:
        def __init__(self, name: str) -> None:
            self.name = name

        def load(self) -> None:
            log.append(f"load:{self.name}")

        def process(self, frame: np.ndarray) -> np.ndarray:
            log.append(self.name)
            return frame

        def close(self) -> None:
            log.append(f"close:{self.name}")

    Pipeline([_Stage("first"), _Stage("second")]).run_video(source, destination)
    assert log.index("close:first") < log.index("load:second")
    assert log.count("first") == log.count("second") > 0
    assert not list(tmp_path.glob(".*.pass*.mp4"))
    assert destination.is_file()


def test_preview_frame_and_out_of_range(tmp_path: Path):
    source = tmp_path / "in.mp4"
    _make_clip(source)
    pipeline = Pipeline()
    frame = pipeline.preview_frame(source, 0)
    assert frame.shape == (48, 64, 3)
    info = probe(source)
    assert info.frame_count is not None
    with pytest.raises(Exception, match="outside"):
        pipeline.preview_frame(source, info.frame_count + 5)


def test_pad_even_adds_a_row_and_column():
    frame = np.zeros((3, 5, 3), dtype=np.uint8)
    padded = pad_even(frame)
    assert padded.shape == (4, 6, 3)
    assert pad_even(padded).shape == padded.shape
