"""
M-TONE-03  White Balance / Tint

Mô hình gain đơn giản theo từng kênh (không phải Kelvin thật), áp dụng trực tiếp
trên giá trị pixel hiện có (không chuyển sang không gian tuyến tính).
Ảnh đầu vào: BGR 3 kênh (thứ tự của OpenCV). Module KHÔNG ghi log.
"""
import numpy as np
import cv2

MODULE_ID = "white_balance"
VERSION = "1.1.0"

# Hệ số cố định. Đổi bất kỳ số nào ở đây => phải tăng VERSION.
TEMP_STRENGTH = 0.30
TINT_MAGENTA_RB_UP = 0.15      # tint > 0: R,B tăng
TINT_MAGENTA_G_DOWN = 0.15     # tint > 0: G giảm
TINT_GREEN_G_UP = 0.15         # tint < 0: G tăng
TINT_GREEN_RB_DOWN = 0.10      # tint < 0: R,B giảm

PARAMS_SCHEMA = {
    "temperature": {"type": "float", "min": -100.0, "max": 100.0, "default": 0.0,
                    "ui": "slider", "step": 1.0,
                    "desc": "-100 lạnh/xanh dương, 0 giữ nguyên, +100 ấm/vàng đỏ."},
    "tint": {"type": "float", "min": -100.0, "max": 100.0, "default": 0.0,
             "ui": "slider", "step": 1.0,
             "desc": "-100 xanh lá, 0 giữ nguyên, +100 magenta."},
}


def _fill_and_check(params):
    params = dict(params or {})
    unknown = set(params) - set(PARAMS_SCHEMA)
    if unknown:
        raise ValueError(f"{MODULE_ID}: tham số không hợp lệ: {sorted(unknown)}")
    out = {}
    for name, spec in PARAMS_SCHEMA.items():
        v = float(params.get(name, spec["default"]))
        # KHÔNG tự clip âm thầm: log phải ghi đúng giá trị đã dùng.
        if not (spec["min"] <= v <= spec["max"]):
            raise ValueError(f"{MODULE_ID}.{name}: {v} ngoài khoảng "
                             f"[{spec['min']}, {spec['max']}]")
        out[name] = v
    return out


def compute_gains(temperature, tint):
    """Trả (gain_b, gain_g, gain_r). Tách riêng để test và để ghi vào log."""
    gb = gg = gr = 1.0
    t = temperature / 100.0
    if t > 0:
        gr *= 1.0 + TEMP_STRENGTH * t
        gb *= 1.0 - TEMP_STRENGTH * t
    elif t < 0:
        t = -t
        gb *= 1.0 + TEMP_STRENGTH * t
        gr *= 1.0 - TEMP_STRENGTH * t

    s = tint / 100.0
    if s > 0:
        gr *= 1.0 + TINT_MAGENTA_RB_UP * s
        gb *= 1.0 + TINT_MAGENTA_RB_UP * s
        gg *= 1.0 - TINT_MAGENTA_G_DOWN * s
    elif s < 0:
        s = -s
        gg *= 1.0 + TINT_GREEN_G_UP * s
        gr *= 1.0 - TINT_GREEN_RB_DOWN * s
        gb *= 1.0 - TINT_GREEN_RB_DOWN * s
    return gb, gg, gr


def _roi_mask(roi, h, w):
    """None -> None (toàn ảnh). Hỗ trợ rect dict hoặc mask ndarray HxW."""
    if roi is None:
        return None
    if isinstance(roi, np.ndarray):
        if roi.shape != (h, w):
            raise ValueError(f"{MODULE_ID}: mask phải có kích thước {w}x{h}")
        return roi.astype(bool)
    if isinstance(roi, dict) and roi.get("type") == "rect":
        m = np.zeros((h, w), bool)
        x, y = max(int(roi["x"]), 0), max(int(roi["y"]), 0)
        m[y:y + int(roi["h"]), x:x + int(roi["w"])] = True
        if not m.any():
            raise ValueError(f"{MODULE_ID}: ROI nằm ngoài ảnh")
        return m
    raise ValueError(f"{MODULE_ID}: ROI không hợp lệ")


def process(image, params=None, roi=None):
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{MODULE_ID}: image phải là ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{MODULE_ID}: cần ảnh màu BGR 3 kênh (ảnh xám không áp dụng được)")
    if image.dtype not in (np.uint8, np.uint16, np.float32):
        raise ValueError(f"{MODULE_ID}: dtype {image.dtype} chưa hỗ trợ")

    p = _fill_and_check(params)
    gb, gg, gr = compute_gains(p["temperature"], p["tint"])

    gains = np.array([gb, gg, gr], dtype=np.float32)       # thứ tự B, G, R
    scaled = image.astype(np.float32) * gains               # broadcast theo kênh

    if image.dtype == np.float32:
        result = scaled                                     # float: không cắt, giữ headroom
        clipped_pct = 0.0
    else:
        hi = float(np.iinfo(image.dtype).max)
        clipped_pct = float((scaled > hi).any(axis=2).mean() * 100.0)
        result = np.clip(np.rint(scaled), 0, hi).astype(image.dtype)

    mask = _roi_mask(roi, *image.shape[:2])
    if mask is not None:
        out = image.copy()
        out[mask] = result[mask]
        clipped_pct = float((scaled[mask] > (hi if image.dtype != np.float32 else np.inf))
                            .any(axis=1).mean() * 100.0) if image.dtype != np.float32 else 0.0
    else:
        out = result

    meta = {
        "gains_bgr": [float(gb), float(gg), float(gr)],     # hệ số thực tế đã nhân
        "clipped_pixels_pct": round(clipped_pct, 4),        # % pixel bị cắt ở mức trắng
        "roi_applied": mask is not None,
        "color_order": "BGR",
    }
    return out, meta


if __name__ == "__main__":
    # Chỉ chạy khi gọi trực tiếp: python white_balance.py ảnh.jpg
    import sys
    img = cv2.imread(sys.argv[1])
    if img is None:
        raise FileNotFoundError(sys.argv[1])
    out, meta = process(img, {"temperature": 80, "tint": 50})
    print(meta)
    cv2.imshow("Output", out)
    cv2.waitKey(0)