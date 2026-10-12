"""Find the weights directory from an explicit path, the environment, or the checkout."""

from __future__ import annotations

import os
from pathlib import Path

WEIGHTS_ENV = "VIDEO_ENHANCE_WEIGHTS"


def resolve_weights_dir(explicit: str | Path | None = None) -> Path:
    if explicit is not None and str(explicit) != "":
        return Path(explicit)
    env = os.environ.get(WEIGHTS_ENV)
    if env:
        return Path(env)
    package_dir = Path(__file__).resolve().parent
    candidates = (
        package_dir.parents[1] / "weights",
        package_dir.parent / "weights",
        Path.cwd() / "weights",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "weights"
