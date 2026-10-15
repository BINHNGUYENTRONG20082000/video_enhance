"""
M-GEO-01  Crop / Rotate

Chức năng: cắt vùng ảnh cần thiết, xoay theo góc cố định (90/180/270) hoặc góc
tùy chỉnh để căn thẳng khung hình, hỗ trợ các tỷ lệ crop phổ biến.

Thứ tự xử lý:  XOAY  ->  CẮT.
Toạ độ crop_x/crop_y/crop_w/crop_h là toạ độ pixel TRÊN ẢNH SAU KHI XOAY
(đúng với những gì người dùng nhìn thấy trên màn hình khi kéo khung).

Quy ước góc: dương = xoay NGƯỢC chiều kim đồng hồ (giống OpenCV).
Module KHÔNG ghi log. Chỉ trả (output, metadata). Log do run_step làm.
Gọi trực tiếp process() chỉ để test; ở sản phẩm luôn gọi qua run_step.
"""
import math
import numpy as np
import cv2

MODULE_ID = "crop_rotate"
VERSION = "1.0.0"

# Gợi ý cho giao diện chọn vùng bằng chuột (không ảnh hưởng thuật toán)
SUPPORTS_ROI = False
SELECTION_SPACE = {
    "base_overrides": {"aspect_ratio": "free"},            # ảnh để bấm chọn: chỉ xoay, chưa cắt
    "invalidated_by": ["rotate_fixed", "angle", "expand"],  # đổi các tham số này thì khung cũ hết đúng
}

ASPECT_PRESETS = {
    "free": None,
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:2": (3, 2),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "3:4": (3, 4),
    "2:3": (2, 3),
    "custom": "custom",   # dùng ratio_w / ratio_h
}

INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

BORDER = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT_101,
}

# Dev/UI đọc schema này để tự sinh slider, dropdown, ô nhập. Thêm tham số không cần sửa UI.
PARAMS_SCHEMA = {
    "rotate_fixed": {"type": "enum", "values": [0, 90, 180, 270], "default": 0,
                     "ui": "buttons", "unit": "deg",
                     "desc": "Xoay cố định, không mất chất lượng (hoán vị pixel)."},
    "angle": {"type": "float", "min": -180.0, "max": 180.0, "default": 0.0,
              "ui": "slider", "step": 0.1, "unit": "deg",
              "desc": "Góc tùy chỉnh để căn thẳng. Dương = ngược chiều kim đồng hồ."},
    "expand": {"type": "bool", "default": True, "ui": "checkbox",
               "desc": "True: mở rộng khung để giữ toàn bộ ảnh. False: giữ kích thước, mất góc."},
    "interpolation": {"type": "enum", "values": list(INTERP), "default": "cubic",
                      "ui": "dropdown", "desc": "Nội suy khi xoay góc lẻ."},
    "border_mode": {"type": "enum", "values": list(BORDER), "default": "constant",
                    "ui": "dropdown", "desc": "Cách lấp vùng trống sinh ra khi xoay."},
    "border_value": {"type": "float", "min": 0.0, "max": 65535.0, "default": 0.0,
                     "ui": "number", "desc": "Giá trị lấp khi border_mode = constant."},
    "aspect_ratio": {"type": "enum", "values": list(ASPECT_PRESETS), "default": "free",
                     "ui": "dropdown", "desc": "Tỷ lệ crop. 'free' = không ép tỷ lệ."},
    "ratio_w": {"type": "float", "min": 0.1, "max": 100.0, "default": 1.0,
                "ui": "number", "desc": "Chỉ dùng khi aspect_ratio = custom."},
    "ratio_h": {"type": "float", "min": 0.1, "max": 100.0, "default": 1.0,
                "ui": "number", "desc": "Chỉ dùng khi aspect_ratio = custom."},
    "crop_x": {"type": "int", "min": 0, "max": 100000, "default": 0, "ui": "rect.x", "unit": "px"},
    "crop_y": {"type": "int", "min": 0, "max": 100000, "default": 0, "ui": "rect.y", "unit": "px"},
    "crop_w": {"type": "int", "min": 0, "max": 100000, "default": 0, "ui": "rect.w", "unit": "px",
               "desc": "0 = không cắt theo khung."},
    "crop_h": {"type": "int", "min": 0, "max": 100000, "default": 0, "ui": "rect.h", "unit": "px",
               "desc": "0 = không cắt theo khung."},
}


# ----------------------------------------------------------------------------
# Tham số
# ----------------------------------------------------------------------------
def _fill_and_check(params: dict) -> dict:
    """Điền mặc định, kiểm tra kiểu/khoảng/giá trị hợp lệ. Trả dict đầy đủ."""
    params = dict(params or {})
    unknown = set(params) - set(PARAMS_SCHEMA)
    if unknown:
        raise ValueError(f"{MODULE_ID}: tham số không hợp lệ: {sorted(unknown)}")

    out = {}
    for name, spec in PARAMS_SCHEMA.items():
        v = params.get(name, spec["default"])
        t = spec["type"]
        if t == "int":
            v = int(v)
        elif t == "float":
            v = float(v)
        elif t == "bool":
            v = bool(v)
        if t == "enum":
            if v not in spec["values"]:
                raise ValueError(f"{MODULE_ID}.{name}: '{v}' không thuộc {spec['values']}")
        elif t in ("int", "float"):
            if not (spec["min"] <= v <= spec["max"]):
                raise ValueError(f"{MODULE_ID}.{name}: {v} ngoài khoảng "
                                 f"[{spec['min']}, {spec['max']}]")
        out[name] = v
    return out


# ----------------------------------------------------------------------------
# Xoay
# ----------------------------------------------------------------------------
def _rot90_matrix(k: int, w: int, h: int) -> np.ndarray:
    """Ma trận 3x3 (ảnh gốc -> ảnh sau xoay) cho xoay ngược chiều kim đồng hồ k*90 độ."""
    if k == 1:
        m = [[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]]
    elif k == 2:
        m = [[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]]
    elif k == 3:
        m = [[0, -1, h - 1], [1, 0, 0], [0, 0, 1]]
    else:
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return np.array(m, dtype=np.float64)


def _rotate(image, p):
    h, w = image.shape[:2]
    total = (p["rotate_fixed"] + p["angle"]) % 360.0

    # Góc là bội của 90 -> xoay không mất mát bằng np.rot90
    if abs(total / 90.0 - round(total / 90.0)) < 1e-9:
        k = int(round(total / 90.0)) % 4
        out = np.ascontiguousarray(np.rot90(image, k)) if k else image
        return out, _rot90_matrix(k, w, h), "lossless_rot90", total

    # Góc lẻ: warpAffine. Dùng angle chuẩn hóa về (-180, 180]
    ang = total if total <= 180 else total - 360.0
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)   # dương = ngược chiều kim đồng hồ

    if p["expand"]:
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw = int(math.ceil(h * sin + w * cos))
        nh = int(math.ceil(h * cos + w * sin))
        M[0, 2] += (nw - 1) / 2.0 - cx
        M[1, 2] += (nh - 1) / 2.0 - cy
    else:
        nw, nh = w, h

    out = cv2.warpAffine(
        image, M, (nw, nh),
        flags=INTERP[p["interpolation"]],
        borderMode=BORDER[p["border_mode"]],
        borderValue=p["border_value"],
    )
    M3 = np.vstack([M, [0, 0, 1]])
    return out, M3, "warp_affine", ang


# ----------------------------------------------------------------------------
# Crop
# ----------------------------------------------------------------------------
def _resolve_ratio(p):
    preset = ASPECT_PRESETS[p["aspect_ratio"]]
    if preset is None:
        return None
    if preset == "custom":
        return (p["ratio_w"], p["ratio_h"])
    return preset


def _resolve_crop_rect(img_w, img_h, p):
    """Trả (x, y, w, h) nguyên, nằm trong ảnh; hoặc None nếu không cắt."""
    ratio = _resolve_ratio(p)
    has_rect = p["crop_w"] > 0 and p["crop_h"] > 0

    if not has_rect and ratio is None:
        return None

    if has_rect:
        x0 = min(max(p["crop_x"], 0), img_w - 1)
        y0 = min(max(p["crop_y"], 0), img_h - 1)
        x1 = min(p["crop_x"] + p["crop_w"], img_w)
        y1 = min(p["crop_y"] + p["crop_h"], img_h)
        bx, by, bw, bh = x0, y0, x1 - x0, y1 - y0
    else:
        bx, by, bw, bh = 0, 0, img_w, img_h

    if bw < 1 or bh < 1:
        raise ValueError(f"{MODULE_ID}: vùng crop nằm ngoài ảnh")

    if ratio is not None:
        r = ratio[0] / ratio[1]
        if bw / bh > r:                # khung quá rộng -> thu chiều rộng
            nw, nh = int(round(bh * r)), bh
        else:                          # khung quá cao -> thu chiều cao
            nw, nh = bw, int(round(bw / r))
        nw, nh = max(1, min(nw, bw)), max(1, min(nh, bh))
        bx += (bw - nw) // 2           # căn giữa trong khung
        by += (bh - nh) // 2
        bw, bh = nw, nh

    return int(bx), int(by), int(bw), int(bh)


# ----------------------------------------------------------------------------
# Hàm chính
# ----------------------------------------------------------------------------
def process(image: np.ndarray, params: dict = None, roi=None):
    """
    image : ndarray HxW hoặc HxWxC, dtype uint8 / uint16 / float32
    params: xem PARAMS_SCHEMA (thiếu thì lấy mặc định)
    roi   : không hỗ trợ (module làm thay đổi kích thước ảnh)
    return: (output, metadata)
    """
    if roi is not None:
        raise ValueError(f"{MODULE_ID}: không hỗ trợ ROI/mask. Dùng crop_x/y/w/h.")
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError(f"{MODULE_ID}: image phải là ndarray 2D hoặc 3D")
    if image.dtype not in (np.uint8, np.uint16, np.float32):
        raise ValueError(f"{MODULE_ID}: dtype {image.dtype} chưa hỗ trợ (uint8/uint16/float32)")

    p = _fill_and_check(params)
    in_h, in_w = image.shape[:2]

    # 1) Xoay
    rotated, M_rot, rot_method, eff_angle = _rotate(image, p)
    rh, rw = rotated.shape[:2]

    # 2) Cắt
    rect = _resolve_crop_rect(rw, rh, p)
    if rect is None:
        out = rotated
        T = np.eye(3)
        applied = None
    else:
        x, y, w, h = rect
        out = np.ascontiguousarray(rotated[y:y + h, x:x + w])
        T = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], dtype=np.float64)
        applied = {"x": x, "y": y, "w": w, "h": h}

    M_total = T @ M_rot   # toạ độ ảnh gốc -> toạ độ ảnh kết quả (dùng khi đo đạc)

    meta = {
        "input_size": {"w": in_w, "h": in_h},
        "output_size": {"w": out.shape[1], "h": out.shape[0]},
        "rotation": {"method": rot_method, "effective_angle_deg": float(eff_angle),
                     "lossless": rot_method == "lossless_rot90"},
        "rotated_size": {"w": rw, "h": rh},
        "crop_rect_applied": applied,                 # toạ độ trên ảnh đã xoay
        "aspect_ratio": p["aspect_ratio"],
        "transform_matrix_orig_to_out": M_total.tolist(),
        "interpolation_used": p["interpolation"] if rot_method == "warp_affine" else None,
        "resampled": rot_method == "warp_affine",     # True = pixel đã bị nội suy
    }
    return out, meta


if __name__ == "__main__":
    img = np.zeros((300, 400, 3), np.uint8)
    cv2.rectangle(img, (150, 100), (250, 200), (255, 255, 255), -1)
    out, meta = process(img, {"angle": 5, "aspect_ratio": "16:9"})
    print(out.shape, meta["rotation"], meta["crop_rect_applied"])