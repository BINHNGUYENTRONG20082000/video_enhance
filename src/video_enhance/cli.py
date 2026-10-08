"""Command line interface for image and video processing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from video_enhance.config import ConfigError, load_config
from video_enhance.io_ffmpeg import FFmpegError
from video_enhance.io_image import write_image
from video_enhance.media import MediaTypeError
from video_enhance.pipeline import build_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.config:
            load_config(args.config)
        pipeline = build_pipeline(
            denoise=args.denoise,
            derain=args.derain,
            deblur=args.deblur,
            defocus=args.defocus,
            enlight=args.enlight,
            device=args.device,
            tile=args.tile,
            fp16=args.fp16,
        )
        try:
            if args.command == "preview":
                frame = pipeline.preview_frame(args.input, args.frame)
                write_image(args.output, frame)
                print(f"wrote {args.output}")
                return 0
            pipeline.run(args.input, args.output, progress=_print_progress)
            if _is_video(args.input):
                print()
            print(f"wrote {args.output}")
            return 0
        finally:
            pipeline.close()
    except (FFmpegError, ConfigError, MediaTypeError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-enhance")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Process an image or a video")
    _add_input(run)
    run.add_argument("-o", "--output", type=Path, required=True)

    preview = subparsers.add_parser("preview", help="Process one frame, or an image, and save a picture")
    _add_input(preview)
    preview.add_argument("--frame", type=int, default=0)
    preview.add_argument("-o", "--output", type=Path, required=True)
    return parser


def _add_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--derain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deblur", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--defocus", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enlight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)


def _print_progress(done: int, total: int | None) -> None:
    if total:
        print(f"\rframes {done}/{total}", end="", flush=True)
    else:
        print(f"\rframes {done}", end="", flush=True)


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


if __name__ == "__main__":
    raise SystemExit(main())
