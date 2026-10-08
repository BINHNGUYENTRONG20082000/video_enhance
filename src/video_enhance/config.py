"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is invalid."""


@dataclass
class PipelineConfig:
    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PipelineConfig:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError("Config root must be a mapping")
        if data:
            raise ConfigError(f"Unknown config keys: {sorted(data)}")
        return cls()


def load_config(path: Path) -> PipelineConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PipelineConfig.from_dict(payload)
