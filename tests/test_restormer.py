import numpy as np
import pytest

from video_enhance.stages.restormer import TASKS, RestormerStage

pytestmark = pytest.mark.skipif(
    any(not task.weights.is_file() for task in TASKS),
    reason="Restormer weights are not in weights/restormer",
)


def _frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 80, size=(32, 48, 3), dtype=np.uint8)


@pytest.mark.parametrize("task", TASKS, ids=[task.key for task in TASKS])
def test_restormer_keeps_shape(task):
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = _frame()
    stage = RestormerStage(task.key, task.weights, task.layer_norm, device=device, tile=32, fp16=device == "cuda")
    try:
        output = stage.process(frame)
    finally:
        stage.close()
    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert not np.array_equal(output, frame)
