import asyncio
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
    summary = gradio_app.describe_video(None, str(source), True, "hat-l")
    assert "Đầu ra dự kiến: 32×32." in summary
    assert "HAT-L x4" in summary
    gan_summary = gradio_app.describe_video(None, str(source), True, "real-hat-gan")
    assert "Real HAT GAN x4" in gan_summary
    before, after, status = gradio_app.preview_video(
        None,
        str(source),
        False,
        False,
        False,
        False,
        False,
        False,
        "hat-l",
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
    summary = gradio_app.describe_video(str(source), None, False, "hat-l")
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
        "hat-l",
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
        "hat-l",
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
        gradio_app.preview_video(None, None, False, False, False, False, False, False, "hat-l", 512, "cpu", False)


def test_upload_is_published_before_the_response():
    seen = {}

    async def _original(*args, **kwargs):
        del args
        seen["force_move"] = kwargs.get("force_move")
        return ("ready",)

    upload = gradio_app._publishing_upload(_original)
    assert asyncio.run(upload(force_move=False)) == ("ready",)
    assert seen["force_move"] is True


def test_gradio_temp_read_waits_out_a_windows_lock():
    async def _open(path, mode="rb"):
        del path, mode
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "Permission denied")
        return "opened"

    calls = {"n": 0}
    sleeps = []

    async def _sleep(delay):
        sleeps.append(delay)

    gradio_path = r"C:\Users\binhn\AppData\Local\Temp\gradio\abc\clip.mp4"
    opened = asyncio.run(
        gradio_app._call_with_permission_retry(_open, gradio_path, mode="rb", sleep=_sleep)
    )
    assert opened == "opened"
    assert calls["n"] == 3
    waited = len(sleeps)

    async def _denied(path, mode="rb"):
        del path, mode
        raise PermissionError(13, "Permission denied")

    with pytest.raises(PermissionError):
        asyncio.run(
            gradio_app._call_with_permission_retry(
                _denied,
                r"C:\Users\binhn\Videos\clip.mp4",
                mode="rb",
                sleep=_sleep,
            )
        )
    assert len(sleeps) == waited

    with pytest.raises(PermissionError):
        asyncio.run(
            gradio_app._call_with_permission_retry(
                _denied,
                gradio_path,
                attempts=2,
                sleep=_sleep,
            )
        )
    assert len(sleeps) == waited + 1


def test_aiofiles_open_retries_without_becoming_a_coroutine():
    class _File:
        def __init__(self):
            self.closed = False

    class _Manager:
        def __init__(self, result):
            self._result = result

        def __await__(self):
            async def _ready():
                if isinstance(self._result, Exception):
                    raise self._result
                return self._result

            return _ready().__await__()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            self._result.closed = True

    calls = {"n": 0}

    def _open(path, mode="rb"):
        del path, mode
        calls["n"] += 1
        if calls["n"] < 2:
            return _Manager(PermissionError(13, "Permission denied"))
        return _Manager(_File())

    sleeps = []

    async def _sleep(delay):
        sleeps.append(delay)

    wrapped = gradio_app._wrap_aiofiles_open(_open)
    opener = wrapped(r"C:\Users\binhn\AppData\Local\Temp\gradio\abc\clip.mp4", mode="rb")
    opener._sleep = _sleep
    assert not asyncio.iscoroutine(opener)

    async def _use():
        async with opener as handle:
            assert isinstance(handle, _File)
            return handle

    handle = asyncio.run(_use())
    assert calls["n"] == 2
    assert sleeps
    assert handle.closed
