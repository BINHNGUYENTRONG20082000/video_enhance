import numpy as np
import pytest
import torch

from video_enhance.pipeline import build_pipeline
from video_enhance.stages.hat import MODELS, HatStage, WEIGHTS_DIR, _tiled_forward, build_network, model_by_key
from video_enhance.stages.hat_arch import HAT


def test_hat_l_is_the_larger_official_network():
    large = model_by_key("hat-l")
    gan = model_by_key("real-hat-gan")
    assert large.label == "HAT-L x4"
    assert gan.label == "Real HAT GAN x4"
    assert sum(large.depths) > sum(gan.depths)
    assert len(large.depths) == 12
    assert len(gan.depths) == 6
    assert {model.key for model in MODELS} == {"hat-l", "real-hat-gan"}


def test_unknown_sr_model_is_rejected():
    with pytest.raises(ValueError):
        build_pipeline(superres=True, sr_model="swinir")


def test_superres_stage_is_last_and_scales_by_four():
    pipeline = build_pipeline(denoise=True, enlight=True, superres=True, sr_model="real-hat-gan")
    assert [stage.name for stage in pipeline.stages] == ["denoise", "enlight", "real-hat-gan"]
    assert pipeline.stages[-1].scale == 4


def test_tiny_network_upscales_and_tiles_cover_the_frame():
    network = HAT(
        upscale=4,
        in_chans=3,
        img_size=16,
        window_size=16,
        compress_ratio=2,
        squeeze_factor=4,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=(1,),
        embed_dim=32,
        num_heads=(4,),
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        drop_path_rate=0.0,
    )
    network.eval()
    tensor = torch.rand(1, 3, 32, 40)
    with torch.inference_mode():
        restored = _tiled_forward(network, tensor, tile=16)
    assert restored.shape == (1, 3, 128, 160)
    assert torch.isfinite(restored).all()


@pytest.mark.parametrize("model", MODELS, ids=[model.key for model in MODELS])
def test_official_weights_upscale(model):
    weights = WEIGHTS_DIR / model.weights_name
    if not weights.is_file():
        pytest.skip("HAT weights are not in weights/hat")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = np.full((16, 32, 3), 48, dtype=np.uint8)
    frame[4:12, 8:24] = 180
    stage = HatStage(model, device=device, tile=16, fp16=device == "cuda")
    try:
        output = stage.process(frame)
    finally:
        stage.close()
    assert output.shape == (64, 128, 3)
    assert output.dtype == np.uint8
    assert int(output.max()) > 0
    assert 20 < float(output.mean()) < 220


def test_hat_l_has_more_parameters_than_real_hat_gan():
    large = build_network(model_by_key("hat-l"))
    gan = build_network(model_by_key("real-hat-gan"))
    assert sum(parameter.numel() for parameter in large.parameters()) > sum(
        parameter.numel() for parameter in gan.parameters()
    )
