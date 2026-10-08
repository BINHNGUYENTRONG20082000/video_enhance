import numpy as np
import pytest

from video_enhance.pipeline import build_pipeline
from video_enhance.stages.enlighten import EnlightenStage


def test_enlighten_brightens_a_dark_frame():
    pytest.importorskip("onnxruntime")
    frame = np.full((32, 48, 3), 8, dtype=np.uint8)
    stage = EnlightenStage(device="cpu")
    try:
        output = stage.process(frame)
    finally:
        stage.close()
    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert float(output.mean()) > float(frame.mean())


def test_pipeline_skips_enlighten_when_disabled():
    frame = np.full((8, 8, 3), 8, dtype=np.uint8)
    output = build_pipeline(enlight=False).process_frame(frame)
    assert np.array_equal(output, frame)
