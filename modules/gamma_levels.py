"""Levels rồi Gamma trên uint8. Gamma < 1 làm sáng, > 1 làm tối.

Áp dụng cùng một LUT cho từng kênh BGR; module không ghi log hay lưu ảnh.
"""
import cv2
import numpy as np
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_module_utils import apply_mask, check_image, check_params, roi_mask

MODULE_ID = "gamma_levels"
VERSION = "1.0.1"
SUPPORTS_ROI = False

PARAMS_SCHEMA = {
    "black": {"type": "int", "min": 0, "max": 254, "default": 0,
              "ui": "number", "desc": "Pixel <= black về đen. Phải nhỏ hơn white."},
    "white": {"type": "int", "min": 1, "max": 255, "default": 255,
              "ui": "number", "desc": "Pixel >= white về trắng. Phải lớn hơn black."},
    "gamma": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0,
              "ui": "slider", "step": 0.01,
              "desc": "< 1 làm sáng, > 1 làm tối, = 1 giữ nguyên gamma."},
}


def _fill_and_check(params):
    p = check_params(MODULE_ID, PARAMS_SCHEMA, params)
    if p["black"] >= p["white"]:
        raise ValueError(f"{MODULE_ID}: cần 0 <= black < white <= 255")
    return p


def process(image, params=None, roi=None):
    check_image(MODULE_ID, image)
    p = _fill_and_check(params)
    mask = roi_mask(MODULE_ID, roi, image.shape)
    values = np.arange(256, dtype=np.float32)
    values = np.clip((values - p["black"]) / (p["white"] - p["black"]), 0, 1)
    lut = np.rint(255 * values ** p["gamma"]).astype(np.uint8)
    result = cv2.LUT(image, lut)
    return apply_mask(image, result, mask), {
        "black": p["black"], "white": p["white"], "gamma": p["gamma"],
        "gamma_convention": "output = normalized_input ** gamma",
        "roi_applied": mask is not None,
        "color_order": "GRAY" if image.ndim == 2 else "BGR",
    }


def adjust_levels_gamma(image, black=0, white=255, gamma=1.0):
    """Giữ cách gọi cũ: trả ảnh thay vì tuple (ảnh, metadata)."""
    return process(image, {"black": black, "white": white, "gamma": gamma})[0]


if __name__ == "__main__":
    from src.image_module_utils import demo
    demo(process, "Levels / Gamma")
