"""Classify an input as an image or a video."""

from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


class MediaTypeError(ValueError):
    """Raised when a path is not a supported image or video."""


def media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    raise MediaTypeError(
        f"Unsupported file type '{path.suffix}'. Use an image ({', '.join(sorted(IMAGE_SUFFIXES))}) "
        f"or a video ({', '.join(sorted(VIDEO_SUFFIXES))})."
    )
