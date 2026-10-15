"""Unsharp mask trên luminance, cố định bán kính và ngưỡng hạn chế tăng nhiễu."""
import cv2
import numpy as np

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_module_utils import apply_mask, check_image, check_params, roi_mask

MODULE_ID = "sharpen"
VERSION = "1.1.0"
SUPPORTS_ROI = True
RADIUS = 1.0
THRESHOLD = 2.0 / 255.0
PARAMS_SCHEMA = {
    "sharpness": {"type": "float", "min": 0, "max": 200, "default": 0, "ui": "slider", "step": 1,
                  "label": "Độ nét", "desc": "0 giữ nguyên; 100 là mức mạnh trước đây; 100–200 tăng nét hơn, có thể nổi nhiễu hoặc viền sáng."},
}


def process(image, params=None, roi=None):
    check_image(MODULE_ID, image)
    p = check_params(MODULE_ID, PARAMS_SCHEMA, params)
    mask = roi_mask(MODULE_ID, roi, image.shape)
    if p["sharpness"] == 0:
        return image.copy(), {"effective_params": p, "roi_applied": mask is not None}
    work = image.astype(np.float32) / 255.0
    ycc = cv2.cvtColor(work, cv2.COLOR_BGR2YCrCb) if image.ndim == 3 else None
    y = ycc[:, :, 0] if ycc is not None else work
    blur = cv2.GaussianBlur(y, (0, 0), RADIUS, borderType=cv2.BORDER_REFLECT_101)
    detail = y - blur
    detail = np.where(np.abs(detail) >= THRESHOLD, detail, 0)
    amount = 2.0 * p["sharpness"] / 100.0
    enhanced = np.clip(y + amount * detail, 0, 1)
    if ycc is not None:
        ycc[:, :, 0] = enhanced
        work = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    else:
        work = enhanced
    result = np.clip(np.rint(work * 255), 0, 255).astype(np.uint8)
    return apply_mask(image, result, mask), {
        "effective_params": p, "amount": amount, "radius": RADIUS, "threshold": THRESHOLD * 255,
        "roi_applied": mask is not None, "method": "unsharp_mask_luminance",
    }


if __name__ == "__main__":
    from src.image_module_utils import demo
    demo(process, "Độ nét")
