"""Stage interface for future models."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Stage(Protocol):
    name: str

    def load(self) -> None: ...

    def process(self, frame_rgb: np.ndarray) -> np.ndarray: ...

    def close(self) -> None: ...
