"""Client HTTP dùng cho Gradio; không import engine hoặc module bộ lọc."""
import base64
import io
import json
import os

import httpx
import numpy as np
from PIL import Image


class ImageApiError(RuntimeError):
    """Lỗi API có thể hiển thị trực tiếp trên giao diện."""


def editor_image(value):
    """Chuẩn hóa ảnh nguồn RGB từ upload, không xử lý hiệu ứng."""
    if isinstance(value, dict):
        value = value.get("composite") if value.get("composite") is not None else value.get("background")
    if value is None:
        return None
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or not value.size:
        raise ValueError("Cần ảnh uint8 hợp lệ")
    if value.ndim == 2:
        value = np.repeat(value[:, :, None], 3, axis=2)
    if value.ndim != 3 or value.shape[2] not in (3, 4):
        raise ValueError("Cần ảnh RGB hoặc RGBA")
    return np.ascontiguousarray(value[:, :, :3])


class ImageApiClient:
    def __init__(self, base_url=None, *, transport=None):
        self.base_url = (base_url or os.getenv("IMAGE_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = httpx.Timeout(180.0, connect=5.0)
        self.transport = transport

    def _request(self, method, path, **kwargs):
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.request(method, self.base_url + "/api/v1" + path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ImageApiError("API hết thời gian chờ; hãy thử lại với ảnh hoặc thiết lập nhẹ hơn") from exc
        except httpx.RequestError as exc:
            raise ImageApiError(f"Không kết nối được API tại {self.base_url}. Chạy: python -m uvicorn api:app --host 127.0.0.1 --port 8000") from exc
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
                message = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
            except (ValueError, AttributeError):
                message = response.text[:1000] or "API không trả thông báo lỗi"
            if response.status_code == 503:
                message += " — hãy thử lại sau khi API sẵn sàng"
            raise ImageApiError(f"HTTP {response.status_code}: {message}")
        try:
            body = response.json()
            if not isinstance(body, dict):
                raise ImageApiError("API phải trả JSON object")
            return body
        except ValueError as exc:
            raise ImageApiError("API trả dữ liệu JSON không hợp lệ") from exc

    def health(self):
        return self._request("GET", "/health")

    def filters(self):
        return self._request("GET", "/filters")

    def configuration(self):
        health = self.health()
        if health.get("status") != "ok":
            raise ImageApiError("API thiếu module xử lý: " + ", ".join(health.get("missing_modules", [])))
        schema = self.filters()
        if not isinstance(schema.get("groups"), list) or not schema["groups"]:
            raise ImageApiError("API không trả danh sách bộ lọc hợp lệ")
        return schema

    @staticmethod
    def _decode(value):
        try:
            raw = base64.b64decode(value["base64"], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                return np.array(image.convert("RGB"))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ImageApiError("API trả ảnh không hợp lệ") from exc

    def process(self, image, settings=None):
        source = editor_image(image)
        if source is None:
            return None, None, {"steps": []}
        payload = io.BytesIO()
        Image.fromarray(source).save(payload, format="PNG")
        body = self._request("POST", "/process", files={"image": ("source.png", payload.getvalue(), "image/png")},
                             data={"settings": json.dumps({} if settings is None else settings, allow_nan=False),
                                   "response_type": "json", "output_format": "png"})
        try:
            result = self._decode(body["image"])
            histogram = self._decode(body["histogram"]) if body.get("histogram") is not None else None
            metadata = dict(body["metadata"])
            metadata["processing_ms"] = body["processing_ms"]
            return result, histogram, metadata
        except (KeyError, TypeError, ValueError) as exc:
            raise ImageApiError("API trả kết quả thiếu ảnh hoặc metadata") from exc
