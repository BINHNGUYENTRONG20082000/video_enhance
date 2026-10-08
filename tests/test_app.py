import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gradio")

import app as gradio_app
from video_enhance.io_image import read_image, write_image

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
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
            "color=c=black:size=64x48:rate=10:duration=0.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_gradio_preview_image_keeps_pixels(tmp_path: Path):
    source = tmp_path / "in.png"
    original = np.full((8, 8, 3), 15, dtype=np.uint8)
    write_image(source, original)
    summary = gradio_app.describe_video(None, str(source), True, 2)
    assert "Đầu ra dự kiến: 16×16." in summary
    before, after, status = gradio_app.preview_video(
        None,
        str(source),
        False,
        False,
        False,
        False,
        False,
        True,
        2,
        512,
        "cpu",
        True,
    )
    assert np.array_equal(before, read_image(source))
    assert np.array_equal(after, before)
    assert "ảnh" in status


def test_gradio_preview_and_render_video(tmp_path: Path):
    source = tmp_path / "clip.mp4"
    _make_clip(source)
    summary = gradio_app.describe_video(str(source), None, False, 2)
    assert "64×48" in summary
    assert "64×48." in summary.split("Đầu ra dự kiến: ")[-1]
    original, enhanced, status = gradio_app.preview_video(
        str(source),
        None,
        False,
        False,
        False,
        False,
        False,
        False,
        2,
        512,
        "cpu",
        False,
    )
    assert original.shape == enhanced.shape == (48, 64, 3)
    assert "frame 0" in status
    output, done = gradio_app.render_video(
        str(source),
        False,
        False,
        False,
        False,
        False,
        False,
        2,
        512,
        "cpu",
        False,
        progress=None,
    )
    rendered = gradio_app.probe(Path(output))
    assert (rendered.width, rendered.height) == (64, 48)
    assert done.startswith("Đã ghi")


def test_missing_media_is_rejected():
    with pytest.raises(Exception, match="Hãy chọn một video hoặc một ảnh"):
        gradio_app.preview_video(None, None, False, False, False, False, False, False, 2, 512, "cpu", False)
