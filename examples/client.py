"""Gọi API với một ảnh nguồn, lưu một ảnh kết quả. Cần httpx."""
import argparse
import json
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description="Gửi ảnh tới Photo Filters API")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--settings", type=Path, help="File JSON chứa tham số")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/process")
    args = parser.parse_args()
    try:
        settings = {} if args.settings is None else json.loads(args.settings.read_text(encoding="utf-8"))
        output_format = "jpeg" if args.output.suffix.lower() in {".jpg", ".jpeg"} else "png"
        with args.input.open("rb") as source:
            response = httpx.post(args.url, files={"image": (args.input.name, source)},
                                  data={"settings": json.dumps(settings), "output_format": output_format},
                                  timeout=180)
        if response.is_error:
            parser.exit(1, f"HTTP {response.status_code}: {response.text}\n")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(response.content)
        print(f"Saved: {args.output} ({response.headers.get('x-image-width')} x {response.headers.get('x-image-height')})")
        print(f"Processing: {response.headers.get('x-processing-time-ms')} ms")
    except (OSError, ValueError, httpx.HTTPError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
