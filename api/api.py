"""HTTP API xử lý ảnh; chạy: python -m uvicorn api:app --port 8000."""
import base64
import io
import json
import os
import threading
import time
from typing import Annotated, Literal

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from src.processing import control_schema, discover, process_image, validate_settings

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_INPUT_PIXELS = 24_000_000
# Một lượt đồng thời: giới hạn RAM và tránh matplotlib chạy nhiều thread.
PROCESSING_SLOT = threading.BoundedSemaphore(1)
MODULES, SKIPPED = discover()
router = APIRouter(prefix="/api/v1", tags=["Image processing"])


@router.get("/health")
def health():
    required = {"crop_rotate", "filter", "histogram"} | {
        "color", "gamma_levels", "lighting", "clahe", "clarity", "texture",
        "sharpen", "deblur", "dehaze", "vignette", "up_down"}
    missing = sorted(required - MODULES.keys())
    return {"status": "ok" if not missing else "degraded", "api_version": "1.0",
            "missing_modules": missing,
            "guided_available": hasattr(getattr(cv2, "ximgproc", None), "guidedFilter")}


@router.get("/filters")
def filters():
    return {"api_version": "1.0", "groups": control_schema(),
            "capabilities": {"guided_available": health()["guided_available"]},
            "limits": {"upload_bytes": MAX_UPLOAD_BYTES,
                       "input_pixels": MAX_INPUT_PIXELS, "output_pixels": 32_000_000},
            "input_formats": ["png", "jpeg", "webp"], "output_formats": ["png", "jpeg"]}


def parse_settings(raw):
    if len(raw.encode("utf-8")) > 32_768:
        raise HTTPException(422, "settings vượt 32 KiB")
    def reject_constant(value):
        raise ValueError(f"Không chấp nhận {value}")
    try:
        return validate_settings(json.loads(raw, parse_constant=reject_constant))
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise HTTPException(422, f"settings không hợp lệ: {exc}") from exc


def decode_image(payload):
    if not payload:
        raise HTTPException(400, "File ảnh rỗng")
    try:
        with Image.open(io.BytesIO(payload)) as header:
            if header.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Chỉ hỗ trợ JPEG, PNG và WebP")
            if getattr(header, "is_animated", False):
                raise HTTPException(415, "Chỉ hỗ trợ ảnh tĩnh")
            if header.width * header.height > MAX_INPUT_PIXELS:
                raise HTTPException(413, "Ảnh nguồn vượt 24 triệu pixel")
            header.verify()
        # getexif có thể load PNG; đọc bằng handle riêng sau verify.
        with Image.open(io.BytesIO(payload)) as exif_header:
            orientation = exif_header.getexif().get(274, 1)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(413, "Ảnh nguồn vượt giới hạn kích thước") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(415, "File không phải ảnh hợp lệ hoặc đã bị hỏng") from exc
    try:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise HTTPException(415, "Không giải mã được ảnh") from exc
    if image is None:
        raise HTTPException(415, "Không giải mã được ảnh")
    if image.dtype != np.uint8:
        raise HTTPException(422, "Chỉ hỗ trợ ảnh 8-bit; hãy đổi ảnh 16-bit sang 8-bit")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise HTTPException(415, "Số kênh ảnh không được hỗ trợ")
    # Chuẩn hóa EXIF để khung crop trên web khớp hướng ảnh người dùng thấy.
    if orientation == 2:
        image = cv2.flip(image, 1)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 0)
    elif orientation == 5:
        image = cv2.transpose(image)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(cv2.transpose(image), -1)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def encode_image(image, output_format, jpeg_quality=90):
    suffix = ".png" if output_format == "png" else ".jpg"
    options = [] if output_format == "png" else [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    success, data = cv2.imencode(suffix, image, options)
    if not success:
        raise RuntimeError("Không mã hóa được ảnh đầu ra")
    return data.tobytes(), "image/png" if output_format == "png" else "image/jpeg"


def image_json(image, payload, mime):
    return {"mime_type": mime, "width": image.shape[1], "height": image.shape[0],
            "base64": base64.b64encode(payload).decode("ascii")}


@router.post("/process", responses={200: {"description": "Ảnh PNG/JPEG hoặc JSON theo response_type"},
                                   413: {"description": "Ảnh quá lớn"},
                                   415: {"description": "Định dạng ảnh không hỗ trợ"},
                                   422: {"description": "Tham số không hợp lệ"},
                                   503: {"description": "Đang bận hoặc thiếu bộ lọc"}})
def process(
    image: Annotated[UploadFile, File(description="Ảnh gốc PNG/JPEG/WebP, tối đa 20 MiB")],
    settings: Annotated[str, Form(description="JSON object chứa các tham số từ GET /filters")] = "{}",
    response_type: Annotated[Literal["image", "json"], Form()] = "image",
    output_format: Annotated[Literal["png", "jpeg"], Form()] = "png",
    jpeg_quality: Annotated[int, Form(ge=1, le=100)] = 90,
):
    """Gửi ảnh gốc + toàn bộ chỉnh sửa; không tích lũy hiệu ứng giữa các request."""
    values = parse_settings(settings)
    if values.get("guided_enabled") and not health()["guided_available"]:
        raise HTTPException(503, "Thiếu ximgproc.guidedFilter; cài opencv-contrib-python")
    if health()["missing_modules"]:
        raise HTTPException(503, "Thiếu module xử lý: " + ", ".join(health()["missing_modules"]))
    if not PROCESSING_SLOT.acquire(blocking=False):
        raise HTTPException(503, "Đang xử lý ảnh khác; hãy thử lại", headers={"Retry-After": "1"})
    started = time.perf_counter()
    try:
        payload = image.file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File ảnh vượt 20 MiB")
        source = decode_image(payload)
        try:
            result, chart, metadata = process_image(source, values, mods=MODULES)
        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except cv2.error as exc:
            raise HTTPException(422, "OpenCV không xử lý được ảnh với thiết lập này") from exc
        encoded, mime = encode_image(result, output_format, jpeg_quality)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        if response_type == "json":
            histogram = None
            if chart is not None:
                histogram_bytes, histogram_mime = encode_image(chart, "png")
                histogram = image_json(chart, histogram_bytes, histogram_mime)
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            return {"image": image_json(result, encoded, mime), "histogram": histogram,
                    "processing_ms": elapsed, "metadata": metadata}
        return Response(encoded, media_type=mime, headers={
            "X-Image-Width": str(result.shape[1]), "X-Image-Height": str(result.shape[0]),
            "X-Processing-Time-Ms": str(elapsed), "Cache-Control": "no-store",
        })
    finally:
        PROCESSING_SLOT.release()


app = FastAPI(title="Photo Filters API", version="1.0.0",
              description="API xử lý ảnh bằng OpenCV, dùng chung bộ lọc với Gradio.")
origins = [item.strip() for item in os.getenv("IMAGE_API_CORS_ORIGINS", "").split(",") if item.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["*"],
                       expose_headers=["X-Image-Width", "X-Image-Height", "X-Processing-Time-Ms"])
app.include_router(router)
