"""Decode and encode video through ffmpeg while copying the audio stream."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FFmpegError(RuntimeError):
    """Raised when ffmpeg or ffprobe is missing or returns an error."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    fps_expr: str
    frame_count: int | None
    has_audio: bool


def require_ffmpeg() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise FFmpegError(
            "Required executable not on PATH: " + ", ".join(missing) + ". Install ffmpeg and try again."
        )


def probe(path: Path) -> VideoInfo:
    require_ffmpeg()
    if not path.is_file():
        raise FFmpegError(f"Input video not found: {path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise FFmpegError(completed.stderr.strip() or f"ffprobe failed for {path}")
    payload = json.loads(completed.stdout or "{}")
    video = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError(f"No video stream in {path}")
    width = int(video["width"])
    height = int(video["height"])
    fps, fps_expr = _frame_rate(video)
    has_audio = any(stream.get("codec_type") == "audio" for stream in payload.get("streams", []))
    frame_count = _frame_count(video.get("nb_frames"), payload.get("format", {}).get("duration"), fps)
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        fps_expr=fps_expr,
        frame_count=frame_count,
        has_audio=has_audio,
    )


def read_frame_at(path: Path, index: int) -> np.ndarray:
    if index < 0:
        raise FFmpegError("Frame index must be >= 0")
    info = probe(path)
    if info.frame_count is not None and index >= info.frame_count:
        raise FFmpegError(f"Frame {index} is outside the video ({info.frame_count} frames)")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select=eq(n\\,{index})",
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    frame = _decode_single(completed.stdout, info.width, info.height)
    if frame is None:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FFmpegError(detail or f"Could not read frame {index} from {path}")
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FFmpegError(detail or f"ffmpeg failed while reading frame {index}")
    return frame


def open_decoder(path: Path) -> subprocess.Popen[bytes]:
    require_ffmpeg()
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def open_encoder(
    destination: Path,
    source: Path,
    width: int,
    height: int,
    fps_expr: str,
    has_audio: bool,
) -> subprocess.Popen[bytes]:
    require_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        fps_expr,
        "-i",
        "pipe:0",
        "-i",
        str(source),
        "-map",
        "0:v:0",
    ]
    if has_audio:
        command.extend(["-map", "1:a:0", "-c:a", "copy"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def read_raw_frame(stream, width: int, height: int) -> np.ndarray | None:
    needed = width * height * 3
    chunks: list[bytes] = []
    got = 0
    while got < needed:
        piece = stream.read(needed - got)
        if not piece:
            if got == 0:
                return None
            raise FFmpegError(f"ffmpeg closed the frame stream after {got} of {needed} bytes")
        chunks.append(piece)
        got += len(piece)
    raw = chunks[0] if len(chunks) == 1 else b"".join(chunks)
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    return frame.copy()


def pad_even(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def drain_stderr(process: subprocess.Popen[bytes]) -> tuple[list[bytes], threading.Thread]:
    bucket: list[bytes] = []

    def _read() -> None:
        if process.stderr is not None:
            bucket.append(process.stderr.read())

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    return bucket, thread


def raise_if_failed(process: subprocess.Popen[bytes], stderr_parts: list[bytes], label: str) -> None:
    if process.returncode not in (0, None):
        detail = b"".join(stderr_parts).decode("utf-8", errors="replace").strip()
        raise FFmpegError(detail or f"{label} exited with status {process.returncode}")


def _frame_rate(video: dict) -> tuple[float, str]:
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = str(video.get(key) or "")
        fps = _parse_rate(raw)
        if fps > 0:
            return fps, raw
    raise FFmpegError("Could not determine the video frame rate")


def _parse_rate(text: str) -> float:
    if not text or text == "0/0":
        return 0.0
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        den = float(denominator)
        if den == 0:
            return 0.0
        return float(numerator) / den
    return float(text)


def _frame_count(raw: object, duration: object, fps: float) -> int | None:
    if raw not in (None, "", "N/A"):
        count = int(str(raw))
        if count > 0:
            return count
    if duration not in (None, "", "N/A") and fps > 0:
        estimated = int(round(float(str(duration)) * fps))
        if estimated > 0:
            return estimated
    return None


def _decode_single(raw: bytes, width: int, height: int) -> np.ndarray | None:
    needed = width * height * 3
    if len(raw) < needed:
        return None
    frame = np.frombuffer(raw[:needed], dtype=np.uint8).reshape(height, width, 3)
    return frame.copy()
