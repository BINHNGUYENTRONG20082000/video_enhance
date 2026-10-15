"""Khử sương bằng dark channel prior và tinh chỉnh transmission theo ảnh dẫn xám.

Ước lượng từ một ảnh; không tái tạo được chi tiết đã mất. Không cần ximgproc.
"""
import cv2
import numpy as np

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_module_utils import apply_mask, check_image, check_params, roi_mask

MODULE_ID = "dehaze"
VERSION = "1.0.0"
SUPPORTS_ROI = True
PATCH_SIZE = 15
TRANSMISSION_FLOOR = 0.15
PARAMS_SCHEMA = {
    "dehaze": {"type": "float", "min": 0, "max": 100, "default": 0, "ui": "slider", "step": 1,
               "label": "Khử sương", "desc": "Kéo phải giảm lớp sương mờ; mức cao có thể làm ảnh tối hoặc tăng nhiễu."},
}


def _dark_channel(image):
    return cv2.erode(image.min(axis=2), np.ones((PATCH_SIZE, PATCH_SIZE), np.uint8),
                     borderType=cv2.BORDER_REPLICATE)


def _refine_transmission(guide, transmission):
    box = lambda x: cv2.boxFilter(x, -1, (41, 41), borderType=cv2.BORDER_REFLECT_101)
    mean_i, mean_t = box(guide), box(transmission)
    variance = np.maximum(box(guide * guide) - mean_i * mean_i, 0)
    covariance = box(guide * transmission) - mean_i * mean_t
    a = covariance / (variance + 0.001)
    b = mean_t - a * mean_i
    return box(a) * guide + box(b)


def process(image, params=None, roi=None):
    check_image(MODULE_ID, image)
    p = check_params(MODULE_ID, PARAMS_SCHEMA, params)
    mask = roi_mask(MODULE_ID, roi, image.shape)
    metadata = {"effective_params": p, "roi_applied": mask is not None,
                "method": "dark_channel_prior", "transmission_floor": TRANSMISSION_FLOOR}
    if p["dehaze"] == 0:
        return image.copy(), metadata
    work = image.astype(np.float32) / 255.0
    work = work[:, :, None] if image.ndim == 2 else work
    pixels = work.reshape(-1, work.shape[2])
    if np.max(np.ptp(pixels, axis=0)) < 1.0 / 255:
        return image.copy(), {**metadata, "skipped": "uniform_image"}
    dark = _dark_channel(work)
    count = max(1, int(dark.size * 0.001))
    candidates = np.argpartition(dark.ravel(), dark.size - count)[-count:]
    air = pixels[candidates[np.argmax(pixels[candidates].sum(axis=1))]]
    air = np.clip(air, 0.05, 1.0)
    transmission = 1 - 0.95 * p["dehaze"] / 100.0 * _dark_channel(work / air)
    guide = work[:, :, 0] if image.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    transmission = np.clip(_refine_transmission(guide, transmission), TRANSMISSION_FLOOR, 1)
    restored = (work - air) / transmission[:, :, None] + air
    result = np.clip(np.rint(restored * 255), 0, 255).astype(np.uint8)
    if image.ndim == 2:
        result = result[:, :, 0]
    return apply_mask(image, result, mask), {
        **metadata, "atmospheric_light": air.tolist(),
        "transmission_min": float(transmission.min()),
    }


if __name__ == "__main__":
    from src.image_module_utils import demo
    demo(process, "Khử sương")
