import os
from pathlib import Path

import numpy as np
import pytest

from video_enhance import Enhancer
from video_enhance.io_image import read_image, write_image
from video_enhance.paths import WEIGHTS_ENV, resolve_weights_dir


def test_default_weights_dir_is_the_checkout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    root = Path(__file__).resolve().parents[1]
    assert resolve_weights_dir() == root / "weights"


def test_weights_dir_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path))
    assert resolve_weights_dir() == tmp_path
    assert Enhancer().weights_dir == tmp_path


def test_explicit_weights_dir_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "env"))
    chosen = tmp_path / "chosen"
    assert Enhancer(weights_dir=chosen).weights_dir == chosen


def test_process_frame_without_tools_returns_the_same_pixels():
    frame = np.full((6, 8, 3), 20, dtype=np.uint8)
    with Enhancer(device="cpu", fp16=False) as enhancer:
        output = enhancer.process_frame(frame)
    assert np.array_equal(output, frame)


def test_frame_must_be_rgb_uint8():
    enhancer = Enhancer(device="cpu")
    with pytest.raises(ValueError, match="RGB"):
        enhancer.process_frame(np.zeros((4, 4), dtype=np.uint8))


def test_unknown_device_and_model_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="Thiết bị"):
        Enhancer(device="mps")
    enhancer = Enhancer(device="cpu", weights_dir=tmp_path)
    with pytest.raises(ValueError, match="siêu phân giải"):
        enhancer.process_frame(np.zeros((4, 4, 3), dtype=np.uint8), superres=True, sr_model="swinir")


def test_run_uses_the_selected_weights_directory(tmp_path: Path):
    source = tmp_path / "in.png"
    destination = tmp_path / "out.png"
    original = np.full((5, 5, 3), 9, dtype=np.uint8)
    write_image(source, original)
    Enhancer(device="cpu", weights_dir=tmp_path / "missing").run(source, destination)
    assert np.array_equal(read_image(destination), original)
    with pytest.raises(RuntimeError, match="Restormer"):
        Enhancer(device="cpu", weights_dir=tmp_path / "missing").run(source, destination, denoise=True)


def test_environment_is_not_required_for_the_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    assert os.environ.get(WEIGHTS_ENV) is None
    assert resolve_weights_dir().name == "weights"
