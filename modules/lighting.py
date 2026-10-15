"""Chỉnh vùng sáng/tối, tương phản tổng thể và điểm trắng/đen trên luminance."""
import cv2
import numpy as np

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_module_utils import apply_mask, check_image, check_params, roi_mask

MODULE_ID = "lighting"
VERSION = "1.1.0"
SUPPORTS_ROI = True
PARAMS_SCHEMA = {
    "highlights": {"type": "float", "min": -200, "max": 200, "default": 0, "ui": "slider", "step": 1,
                   "label": "Vùng sáng", "desc": "Kéo trái giảm, kéo phải tăng sáng ở vùng sáng; ngoài ±100 mạnh hơn; 0 giữ nguyên."},
    "shadows": {"type": "float", "min": -200, "max": 200, "default": 0, "ui": "slider", "step": 1,
                "label": "Vùng tối", "desc": "Kéo phải nâng vùng tối, kéo trái làm tối hơn; ngoài ±100 mạnh hơn; 0 giữ nguyên."},
    "contrast": {"type": "float", "min": -100, "max": 100, "default": 0, "ui": "slider", "step": 1,
                 "label": "Tương phản tổng thể", "desc": "Kéo trái giảm tương phản, kéo phải tăng tương phản toàn ảnh."},
    "whites": {"type": "float", "min": -100, "max": 100, "default": 0, "ui": "slider", "step": 1,
               "label": "Điểm trắng", "desc": "Kéo phải tăng độ sáng điểm trắng, kéo trái giảm; mức cao có thể cháy sáng."},
    "blacks": {"type": "float", "min": -100, "max": 100, "default": 0, "ui": "slider", "step": 1,
               "label": "Điểm đen", "desc": "Kéo trái tăng độ sâu màu đen, kéo phải nâng vùng đen."},
}


def _adjust_region(y, value, shadows):
    """Ghép các bước tối đa ±100 để tăng lực mà giữ đường cong đơn điệu."""
    remaining = abs(value)
    direction = 1 if value >= 0 else -1
    while remaining > 0:
        step = min(remaining, 100.0)
        strength = 0.8 * direction * step / 100.0
        detail = y * (1 - y) ** 2 if shadows else y ** 2 * (1 - y)
        y = np.clip(y + strength * detail, 0, 1)
        remaining -= step
    return y


def process(image, params=None, roi=None):
    check_image(MODULE_ID, image)
    p = check_params(MODULE_ID, PARAMS_SCHEMA, params)
    mask = roi_mask(MODULE_ID, roi, image.shape)
    if not any(p.values()):
        return image.copy(), {"effective_params": p, "roi_applied": mask is not None}
    work = image.astype(np.float32) / 255.0
    ycc = cv2.cvtColor(work, cv2.COLOR_BGR2YCrCb) if image.ndim == 3 else None
    y = ycc[:, :, 0].copy() if ycc is not None else work
    black = -0.15 * p["blacks"] / 100.0
    white = 1.0 - 0.25 * p["whites"] / 100.0
    y = np.clip((y - black) / (white - black), 0, 1)
    y = _adjust_region(y, p["shadows"], shadows=True)
    y = _adjust_region(y, p["highlights"], shadows=False)
    if p["contrast"]:
        exponent = 2.0 ** (p["contrast"] / 100.0)
        a, b = y ** exponent, (1 - y) ** exponent
        y = a / (a + b)
    if ycc is not None:
        ycc[:, :, 0] = y
        work = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    else:
        work = y
    result = np.clip(np.rint(work * 255), 0, 255).astype(np.uint8)
    return apply_mask(image, result, mask), {
        "effective_params": p, "roi_applied": mask is not None,
        "processed_channel": "Y" if ycc is not None else "GRAY",
        "adjustment_order": ["levels", "shadows", "highlights", "contrast"],
    }


if __name__ == "__main__":
    from src.image_module_utils import demo
    demo(process, "Ánh sáng")
