"""Bốn điều chỉnh màu trên ảnh BGR uint8: nhiệt độ, sắc thái, rực màu, bão hòa.

Temperature/tint dùng gain tương đối, không phải nhiệt độ Kelvin thật.
Vibrance điều chỉnh S theo (1 - S), ưu tiên màu ít bão hòa; không có mặt nạ da.
Thứ tự: cân bằng trắng → vibrance → saturation. Không đọc/ghi ảnh khi import.
"""
import cv2
import numpy as np

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_module_utils import apply_mask, check_image, check_params, roi_mask
from modules.white_balance import compute_gains

MODULE_ID = "color"
VERSION = "1.0.0"
SUPPORTS_ROI = True
PARAMS_SCHEMA = {
    "temperature": {"type": "float", "min": -100.0, "max": 100.0, "default": 0.0,
                    "ui": "slider", "step": 1.0, "label": "Nhiệt độ",
                    "desc": "Kéo trái cho màu lạnh hơn, kéo phải cho màu ấm hơn; 0 giữ nguyên."},
    "tint": {"type": "float", "min": -100.0, "max": 100.0, "default": 0.0,
             "ui": "slider", "step": 1.0, "label": "Sắc thái",
             "desc": "Kéo trái thêm xanh lá, kéo phải thêm hồng tím; 0 giữ nguyên."},
    "vibrance": {"type": "float", "min": -100.0, "max": 100.0, "default": 0.0,
                 "ui": "slider", "step": 1.0, "label": "Độ rực màu",
                 "desc": "Ưu tiên điều chỉnh những màu nhạt; kéo phải tăng, kéo trái giảm; 0 giữ nguyên."},
    "saturation": {"type": "float", "min": -100.0, "max": 100.0, "default": 0.0,
                   "ui": "slider", "step": 1.0, "label": "Độ bão hòa",
                   "desc": "Điều chỉnh độ đậm của tất cả màu; -100 thành đen trắng, 0 giữ nguyên."},
}


def _fill_and_check(params):
    return check_params(MODULE_ID, PARAMS_SCHEMA, params)


def process(image, params=None, roi=None):
    check_image(MODULE_ID, image)
    if image.ndim != 3:
        raise ValueError(f"{MODULE_ID}: cần ảnh màu BGR 3 kênh")
    p = _fill_and_check(params)
    mask = roi_mask(MODULE_ID, roi, image.shape)
    gains = np.asarray(compute_gains(p["temperature"], p["tint"]), dtype=np.float32)
    scaled = image.astype(np.float32) * gains
    clipped_pct = float((scaled > 255).any(axis=2).mean() * 100)
    scaled = np.clip(scaled, 0, 255)
    if p["vibrance"] != 0 or p["saturation"] != 0:
        # Float HSV avoids quantizing hue/saturation between adjustments.
        hsv = cv2.cvtColor(scaled / 255.0, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        saturation = np.clip(saturation * (1 + p["vibrance"] / 100.0 * (1 - saturation)), 0, 1)
        hsv[:, :, 1] = np.clip(saturation * (1 + p["saturation"] / 100.0), 0, 1)
        scaled = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) * 255.0
    result = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    return apply_mask(image, result, mask), {
        "effective_params": p, "gains_bgr": [float(g) for g in gains],
        "white_balance_clipped_pixels_pct": round(clipped_pct, 4),
        "roi_applied": mask is not None, "color_order": "BGR",
        "adjustment_order": ["white_balance", "vibrance", "saturation"],
        "saturation_space": "HSV", "vibrance_method": "saturation_weighted",
    }


if __name__ == "__main__":
    from src.image_module_utils import demo
    demo(process, "Màu sắc")
