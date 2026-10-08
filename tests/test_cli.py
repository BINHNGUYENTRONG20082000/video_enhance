from pathlib import Path

import numpy as np

from video_enhance.cli import main
from video_enhance.io_image import read_image, write_image


def test_help_exits_zero():
    try:
        main(["--help"])
    except SystemExit as caught:
        assert caught.code == 0
    else:
        raise AssertionError("help should exit")


def test_missing_input_returns_one(tmp_path: Path):
    code = main(["run", str(tmp_path / "missing.png"), "-o", str(tmp_path / "out.png")])
    assert code == 1


def test_run_image(tmp_path: Path):
    source = tmp_path / "in.png"
    destination = tmp_path / "out.png"
    original = np.full((6, 6, 3), 40, dtype=np.uint8)
    write_image(source, original)
    assert main(["run", str(source), "-o", str(destination), "--no-denoise", "--no-deblur", "--no-enlight"]) == 0
    assert np.array_equal(read_image(destination), original)


def test_default_config_is_accepted():
    root = Path(__file__).resolve().parents[1]
    code = main(
        [
            "run",
            "missing.png",
            "-o",
            "out.png",
            "--config",
            str(root / "configs" / "default.yaml"),
        ]
    )
    assert code == 1
