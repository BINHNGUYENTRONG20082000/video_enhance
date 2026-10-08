from pathlib import Path

import numpy as np

from video_enhance.io_image import read_image
from video_enhance.pipeline import Pipeline, build_pipeline


class _RecordingStage:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def load(self) -> None:
        self.log.append(f"load:{self.name}")

    def process(self, frame_rgb: np.ndarray) -> np.ndarray:
        self.log.append(self.name)
        updated = frame_rgb.copy()
        updated[0, 0, 0] = np.uint8(int(updated[0, 0, 0]) + 1)
        return updated

    def close(self) -> None:
        self.log.append(f"close:{self.name}")


def test_build_pipeline_has_no_models():
    pipeline = build_pipeline()
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert pipeline.stages == []
    assert np.array_equal(pipeline.process_frame(frame), frame)


def test_stages_run_in_order():
    log: list[str] = []
    stages = [_RecordingStage(name, log) for name in ("denoise", "superres")]
    pipeline = Pipeline(stages)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    output = pipeline.process_frame(frame)
    assert log == ["load:denoise", "load:superres", "denoise", "superres"]
    assert int(output[0, 0, 0]) == 2
    pipeline.close()
    assert log[-2:] == ["close:denoise", "close:superres"]


def test_image_round_trip_keeps_pixels(tmp_path: Path):
    source = tmp_path / "in.png"
    destination = tmp_path / "out.png"
    original = np.zeros((8, 10, 3), dtype=np.uint8)
    original[1, 2] = (10, 20, 30)
    from video_enhance.io_image import write_image

    write_image(source, original)
    build_pipeline().run(source, destination)
    assert np.array_equal(read_image(destination), original)
