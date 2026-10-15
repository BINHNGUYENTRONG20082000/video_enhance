"""Median, Bilateral, NLM và Guided Filter cho ảnh uint8.

process() trả (ảnh BGR/grayscale, metadata); không đọc ảnh hay mở GUI khi import.
"""
from collections.abc import Mapping
import cv2

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_module_utils import check_image, check_params

MODULE_ID = "filter"
VERSION = "1.0.0"
SUPPORTS_ROI = False
FILTER_PARAMS = {
    "median": ("kernel_size",),
    "bilateral": ("diameter", "sigma_color", "sigma_space"),
    "nlm": ("h", "template_window_size", "search_window_size"),
    "guided": ("radius", "eps"),
}
PARAMS_SCHEMA = {
    "filter": {"type": "enum", "values": list(FILTER_PARAMS), "default": "median",
               "ui": "dropdown", "desc": "Chọn bộ lọc khử nhiễu."},
    "kernel_size": {"type": "int", "min": 3, "max": 31, "default": 3,
                    "ui": "number", "desc": "Kích thước kernel Median, số lẻ."},
    "diameter": {"type": "int", "min": 3, "max": 31, "default": 9,
                 "ui": "number", "desc": "Đường kính vùng lọc Bilateral, số lẻ."},
    "sigma_color": {"type": "float", "min": 0.1, "max": 255.0, "default": 75.0,
                    "ui": "slider", "step": 0.1, "desc": "Độ làm mượt theo màu."},
    "sigma_space": {"type": "float", "min": 0.1, "max": 255.0, "default": 75.0,
                    "ui": "slider", "step": 0.1, "desc": "Độ làm mượt theo khoảng cách."},
    "h": {"type": "float", "min": 0.0, "max": 50.0, "default": 10.0,
          "ui": "slider", "step": 0.1, "desc": "Cường độ khử nhiễu NLM; 0 giữ nguyên ảnh."},
    "template_window_size": {"type": "int", "min": 3, "max": 15, "default": 7,
                             "ui": "number", "desc": "Kích thước patch NLM, số lẻ."},
    "search_window_size": {"type": "int", "min": 3, "max": 63, "default": 15,
                           "ui": "number", "desc": "Vùng tìm kiếm NLM, số lẻ và >= patch."},
    "radius": {"type": "int", "min": 1, "max": 64, "default": 8,
               "ui": "number", "desc": "Bán kính Guided Filter."},
    "eps": {"type": "float", "min": 0.000001, "max": 65025.0, "default": 0.1,
            "ui": "number", "desc": "Tham số điều chỉnh Guided trên thang pixel 0–255."},
}


def _fill_and_check(params):
    params = {} if params is None else params
    if not isinstance(params, Mapping):
        raise ValueError(f"{MODULE_ID}: params phải là dict")
    unknown = set(params) - set(PARAMS_SCHEMA)
    if unknown:
        raise ValueError(f"{MODULE_ID}: tham số không hợp lệ: {list(unknown)}")
    method = params.get("filter", "median")
    if not isinstance(method, str) or method not in FILTER_PARAMS:
        raise ValueError(f"{MODULE_ID}: bộ lọc phải là median, bilateral, nlm hoặc guided")
    active = FILTER_PARAMS[method]
    schema = {name: PARAMS_SCHEMA[name] for name in active}
    p = check_params(MODULE_ID, schema, {name: params[name] for name in active if name in params})
    for name in ("kernel_size", "diameter", "template_window_size", "search_window_size"):
        if name in p and p[name] % 2 == 0:
            raise ValueError(f"{MODULE_ID}.{name}: cần số lẻ")
    if method == "nlm" and p["search_window_size"] < p["template_window_size"]:
        raise ValueError(f"{MODULE_ID}: search_window_size phải >= template_window_size")
    return {"filter": method, **p}


def process(image, params=None, roi=None):
    if roi is not None:
        raise ValueError(f"{MODULE_ID}: không hỗ trợ ROI; bộ lọc xử lý toàn ảnh")
    check_image(MODULE_ID, image)
    p = _fill_and_check(params)
    method = p["filter"]
    try:
        if method == "median":
            result = cv2.medianBlur(image, p["kernel_size"])
        elif method == "bilateral":
            result = cv2.bilateralFilter(image, p["diameter"], p["sigma_color"], p["sigma_space"])
        elif method == "nlm":
            if p["h"] == 0:
                result = image.copy()
            elif image.ndim == 2:
                result = cv2.fastNlMeansDenoising(image, None, p["h"],
                                                p["template_window_size"], p["search_window_size"])
            else:
                result = cv2.fastNlMeansDenoisingColored(image, None, p["h"], p["h"],
                                                       p["template_window_size"], p["search_window_size"])
        else:
            guided = getattr(getattr(cv2, "ximgproc", None), "guidedFilter", None)
            if not callable(guided):
                raise ValueError(f"{MODULE_ID}: Guided Filter cần OpenCV có ximgproc.guidedFilter "
                                 "(opencv-contrib-python)")
            result = guided(image, image, p["radius"], p["eps"])
    except cv2.error as exc:
        raise ValueError(f"{MODULE_ID}: bộ lọc {method} không xử lý được ảnh: {exc}") from exc
    return result, {
        "filter": method, "effective_params": {k: v for k, v in p.items() if k != "filter"},
        "roi_applied": False, "color_order": "GRAY" if image.ndim == 2 else "BGR",
    }


def median_filter(image, kernel_size=3):
    return process(image, {"filter": "median", "kernel_size": kernel_size})[0]


def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    return process(image, {"filter": "bilateral", "diameter": diameter,
                           "sigma_color": sigma_color, "sigma_space": sigma_space})[0]


def gaussian_nlm_filter(image, h=10, template_window_size=7, search_window_size=15):
    return process(image, {"filter": "nlm", "h": h,
                           "template_window_size": template_window_size,
                           "search_window_size": search_window_size})[0]


def guided_filter(image, radius=8, eps=0.1):
    return process(image, {"filter": "guided", "radius": radius, "eps": eps})[0]


if __name__ == "__main__":
    from src.image_module_utils import demo
    demo(process, "Filter")
