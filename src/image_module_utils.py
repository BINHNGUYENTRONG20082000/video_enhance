"""Kiểm tra đầu vào dùng chung cho các module uint8."""
from collections.abc import Mapping
import numpy as np


def check_params(module_id, schema, params):
    params = {} if params is None else params
    if not isinstance(params, Mapping):
        raise ValueError(f"{module_id}: params phải là dict")
    unknown = set(params) - set(schema)
    if unknown:
        raise ValueError(f"{module_id}: tham số không hợp lệ: {list(unknown)}")
    out = {}
    for name, spec in schema.items():
        try:
            value = float(params.get(name, spec["default"]))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{module_id}.{name}: cần giá trị số") from exc
        if not np.isfinite(value) or not spec["min"] <= value <= spec["max"]:
            raise ValueError(f"{module_id}.{name}: phải nằm trong [{spec['min']}, {spec['max']}]")
        if spec["type"] == "int":
            if not value.is_integer():
                raise ValueError(f"{module_id}.{name}: cần số nguyên")
            value = int(value)
        out[name] = value
    return out


def check_image(module_id, image):
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{module_id}: image phải là ndarray")
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] == 3):
        raise ValueError(f"{module_id}: cần ảnh xám HxW hoặc ảnh BGR HxWx3")
    if image.size == 0:
        raise ValueError(f"{module_id}: ảnh không được rỗng")
    if image.dtype != np.uint8:
        raise ValueError(f"{module_id}: chỉ hỗ trợ ảnh uint8")


def roi_mask(module_id, roi, shape):
    if roi is None:
        return None
    h, w = shape[:2]
    if isinstance(roi, np.ndarray):
        if roi.shape != (h, w):
            raise ValueError(f"{module_id}: mask phải có kích thước HxW giống ảnh")
        mask = roi.astype(bool)
    elif isinstance(roi, dict) and roi.get("type") == "rect":
        try:
            coords = [float(roi[k]) for k in ("x", "y", "w", "h")]
            if not all(np.isfinite(v) and v.is_integer() for v in coords):
                raise ValueError("tọa độ phải là số nguyên")
            x, y, rw, rh = map(int, coords)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{module_id}: ROI rect cần x, y, w, h nguyên hợp lệ") from exc
        if rw <= 0 or rh <= 0:
            raise ValueError(f"{module_id}: chiều rộng và chiều cao ROI phải > 0")
        x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + rw), min(h, y + rh)
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"{module_id}: ROI nằm ngoài ảnh")
        mask = np.zeros((h, w), bool)
        mask[y1:y2, x1:x2] = True
    else:
        raise ValueError(f"{module_id}: ROI không hợp lệ")
    if not mask.any():
        raise ValueError(f"{module_id}: mask ROI không được rỗng")
    return mask


def apply_mask(image, result, mask):
    if mask is None:
        return result
    out = image.copy()
    out[mask] = result[mask]
    return out


def demo(process, description):
    import argparse
    import cv2
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("image_path", nargs="?",
                        default=r"D:\Model_opencv\image\image29-261x300.jpg")
    args = parser.parse_args()
    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(args.image_path)
    result, meta = process(image)
    print(meta)
    cv2.imshow("Output", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
