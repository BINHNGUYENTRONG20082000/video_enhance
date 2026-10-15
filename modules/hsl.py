"""HSL từng dải màu, chuyển tiếp mềm và chọn vùng theo màu nguồn."""
import sys
from pathlib import Path
import cv2
import numpy as np
if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.image_module_utils import check_image, check_params, roi_mask, apply_mask

MODULE_ID = 'hsl'
VERSION = '1.0.0'
SUPPORTS_ROI = True
COLOR_BANDS = {
    'red': ('Đỏ', 0), 'orange': ('Cam', 30), 'yellow': ('Vàng', 60),
    'green': ('Xanh lá', 120), 'aqua': ('Xanh ngọc', 180),
    'blue': ('Xanh dương', 240), 'purple': ('Tím', 270), 'magenta': ('Hồng', 300),
}
PARAMS_SCHEMA = {}
for color, (label, center) in COLOR_BANDS.items():
    for adjustment, title, description in (
        ('hue', 'Tông màu', 'Đổi tông của dải màu đã chọn; mức ±100 dịch tối đa ±30 độ.'),
        ('saturation', 'Độ bão hòa', 'Kéo trái giảm màu, kéo phải làm màu đã chọn đậm hơn.'),
        ('lightness', 'Độ sáng', 'Kéo trái làm tối, kéo phải làm sáng dải màu đã chọn.')):
        PARAMS_SCHEMA[f'hsl_{color}_{adjustment}'] = {
            'type': 'float', 'min': -100, 'max': 100, 'default': 0, 'ui': 'slider', 'step': 1,
            'label': f'{title} — {label}', 'desc': description + ' 0 giữ nguyên.',
        }


def _weight(hue, index):
    centers = [center for _, center in COLOR_BANDS.values()]
    center = centers[index]
    left = (center-centers[index-1]) % 360
    right = (centers[(index+1) % len(centers)]-center) % 360
    delta = (hue-center+180) % 360 - 180
    distance = np.abs(delta) / np.where(delta < 0, left, right)
    return np.where(distance < 1, 0.5*(1+np.cos(np.pi*np.minimum(distance, 1))), 0).astype(np.float32)


def process(image, params=None, roi=None):
    check_image(MODULE_ID, image)
    p = check_params(MODULE_ID, PARAMS_SCHEMA, params)
    mask = roi_mask(MODULE_ID, roi, image.shape)
    metadata = {'effective_params': p, 'roi_applied': mask is not None,
                'method': 'selective_hsl', 'selection': 'original_hue', 'color_bands': list(COLOR_BANDS)}
    if not any(p.values()) or image.ndim == 2:
        return image.copy(), metadata
    source = image.astype(np.float32) / 255.0
    hls = cv2.cvtColor(source, cv2.COLOR_BGR2HLS)
    hue, light, saturation = cv2.split(hls)
    hue_delta = np.zeros_like(hue)
    sat_delta = np.zeros_like(hue)
    light_delta = np.zeros_like(hue)
    # Avoid assigning neutral gray/white/black pixels to the red band (undefined hue).
    chromatic = np.clip(saturation/0.1, 0, 1)
    for index, color in enumerate(COLOR_BANDS):
        if not any(p[f'hsl_{color}_{name}'] for name in ('hue', 'saturation', 'lightness')):
            continue
        weight = _weight(hue, index)
        hue_delta += weight*p[f'hsl_{color}_hue']*0.3
        sat_delta += weight*p[f'hsl_{color}_saturation']/100.0
        light_delta += weight*p[f'hsl_{color}_lightness']/100.0
    hls[:, :, 0] = (hue + hue_delta*chromatic) % 360
    hls[:, :, 2] = np.clip(saturation*(1+sat_delta), 0, 1)
    light_delta = np.clip(light_delta*chromatic, -1, 1)
    hls[:, :, 1] = np.clip(light + np.where(light_delta >= 0, 1-light, light)*light_delta, 0, 1)
    changed = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    result = np.clip(np.rint(changed*255), 0, 255).astype(np.uint8)
    # Preserve unaffected pixels exactly, including conversion rounding at band boundaries.
    active = (hue_delta*chromatic != 0) | (sat_delta*saturation != 0) | (light_delta != 0)
    result[~active] = image[~active]
    return apply_mask(image, result, mask), metadata


if __name__ == '__main__':
    from src.image_module_utils import demo
    demo(process, 'Chỉnh từng màu HSL')
