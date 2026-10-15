"""CLAHE trên kênh L của BGR uint8, hoặc trực tiếp trên ảnh xám.

ROI giới hạn vùng ghi kết quả; histogram CLAHE được tính trên toàn ảnh.
Module không ghi log, lưu ảnh hay mở cửa sổ khi import/process.
"""
import cv2
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_module_utils import apply_mask, check_image, check_params, roi_mask

MODULE_ID = "clahe"
VERSION = "1.0.0"
SUPPORTS_ROI =  False

PARAMS_SCHEMA = {
    "clip_limit": {"type": "float", "min": 0.1, "max": 40.0, "default": 2.0,
                   "ui": "slider", "step": 0.1, "desc": "Giới hạn tăng tương phản."},
    "tile_grid_x": {"type": "int", "min": 1, "max": 64, "default": 8,
                    "ui": "number", "desc": "Số ô chia theo chiều ngang."},
    "tile_grid_y": {"type": "int", "min": 1, "max": 64, "default": 8,
                    "ui": "number", "desc": "Số ô chia theo chiều dọc."},
}


def _fill_and_check(params):
    return check_params(MODULE_ID, PARAMS_SCHEMA, params)


def process(image, params=None, roi=None):
    check_image(MODULE_ID, image)
    p = _fill_and_check(params)
    mask = roi_mask(MODULE_ID, roi, image.shape)
    clahe = cv2.createCLAHE(clipLimit=p["clip_limit"],
                           tileGridSize=(p["tile_grid_x"], p["tile_grid_y"]))
    if image.ndim == 2:
        result = clahe.apply(image)
    else:
        l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
        result = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    return apply_mask(image, result, mask), {
        "clip_limit": p["clip_limit"],
        "tile_grid_size": [p["tile_grid_x"], p["tile_grid_y"]],
        "processed_channel": "GRAY" if image.ndim == 2 else "L",
        "roi_applied": mask is not None, "histogram_scope": "full_image",
        "color_order": "GRAY" if image.ndim == 2 else "BGR",
    }


if __name__ == "__main__":
    from src.image_module_utils import demo
    demo(process, "CLAHE")
