"""Shared luminance band-pass processing for Texture and Clarity."""
import cv2
import numpy as np
from src.image_module_utils import check_image, check_params, roi_mask, apply_mask


def process_band(image, params, roi, module_id, schema, field, small, large, limit, gain, midtones=False):
    check_image(module_id, image)
    p = check_params(module_id, schema, params)
    mask = roi_mask(module_id, roi, image.shape)
    meta = {'effective_params': p, 'roi_applied': mask is not None,
            'processed_channel': 'Y' if image.ndim == 3 else 'GRAY',
            'scales': [small, large], 'method': 'luminance_bandpass'}
    if p[field] == 0:
        return image.copy(), meta
    work = image.astype(np.float32) / 255.0
    ycc = cv2.cvtColor(work, cv2.COLOR_BGR2YCrCb) if image.ndim == 3 else None
    y = ycc[:, :, 0].copy() if ycc is not None else work
    fine = cv2.GaussianBlur(y, (0, 0), small, borderType=cv2.BORDER_REFLECT_101)
    coarse = cv2.GaussianBlur(y, (0, 0), large, borderType=cv2.BORDER_REFLECT_101)
    band = fine - coarse
    band = np.where(np.abs(band) >= 1/255.0, band, 0)
    band = np.clip(band, -limit, limit)
    if midtones:
        band *= 4*y*(1-y)
    effective_gain = gain if p[field] > 0 else 1.0
    enhanced = np.clip(y + effective_gain*p[field]/100.0*band, 0, 1)
    if ycc is None:
        result = enhanced
    else:
        ycc[:, :, 0] = enhanced
        result = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    result = np.clip(np.rint(result*255), 0, 255).astype(np.uint8)
    return apply_mask(image, result, mask), meta
