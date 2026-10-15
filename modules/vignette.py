"""Viền sáng/tối hình ellipse, độ rộng và độ mềm cố định theo kích thước ảnh."""
import cv2
import numpy as np

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_module_utils import apply_mask, check_image, check_params, roi_mask

MODULE_ID = "vignette"
VERSION = "1.1.0"
SUPPORTS_ROI = True
PARAMS_SCHEMA = {
    "vignette": {"type": "float", "min": -100, "max": 100, "default": 0, "ui": "slider", "step": 1,
                 "label": "Tối / Sáng viền", "desc": "Kéo trái làm tối mép ảnh, kéo phải làm sáng; 0 giữ nguyên."},
}


def process(image, params=None, roi=None):
    check_image(MODULE_ID, image)
    p = check_params(MODULE_ID, PARAMS_SCHEMA, params)
    mask = roi_mask(MODULE_ID, roi, image.shape)
    if p["vignette"] == 0:
        return image.copy(), {"effective_params": p, "roi_applied": mask is not None}
    h, w = image.shape[:2]
    x = (np.arange(w, dtype=np.float32) - (w - 1) / 2) / max((w - 1) / 2, 1)
    y = (np.arange(h, dtype=np.float32) - (h - 1) / 2) / max((h - 1) / 2, 1)
    radius = (y[:, None] ** 2 + x[None, :] ** 2) / 2
    weight = np.clip((radius - 0.2) / 0.8, 0, 1)
    weight = weight * weight * (3 - 2 * weight)
    if image.ndim == 3:
        weight = weight[:, :, None]
    amount = p["vignette"] / 100.0
    work = image.astype(np.float32) / 255.0
    result = work + (1 - work) * 0.8 * amount * weight if amount > 0 else work * (1 + 0.8 * amount * weight)
    result = np.clip(np.rint(result * 255), 0, 255).astype(np.uint8)
    return apply_mask(image, result, mask), {
        "effective_params": p, "roi_applied": mask is not None,
        "shape": "ellipse", "feather": "smoothstep",
    }


if __name__ == "__main__":
    from src.image_module_utils import demo
    demo(process, "Tối / Sáng viền")
