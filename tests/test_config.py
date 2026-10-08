from pathlib import Path

import pytest

from video_enhance.config import ConfigError, PipelineConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_empty_config():
    assert load_config(ROOT / "configs" / "default.yaml") == PipelineConfig()


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="Unknown config keys"):
        PipelineConfig.from_dict({"denoise": True})


def test_round_trip_dict():
    assert PipelineConfig.from_dict(PipelineConfig().to_dict()) == PipelineConfig()
