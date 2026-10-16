"""Luồng xử lý ảnh dùng chung cho Gradio và API, không phụ thuộc giao diện."""
import importlib
import math
import pkgutil

import cv2
import numpy as np

from modules.color import PARAMS_SCHEMA as COLOR_PARAMS_SCHEMA
from modules.lighting import PARAMS_SCHEMA as LIGHTING_PARAMS_SCHEMA
from modules.sharpen import PARAMS_SCHEMA as SHARPEN_PARAMS_SCHEMA
from modules.dehaze import PARAMS_SCHEMA as DEHAZE_PARAMS_SCHEMA
from modules.vignette import PARAMS_SCHEMA as VIGNETTE_PARAMS_SCHEMA
from modules.up_down import PARAMS_SCHEMA as RESIZE_PARAMS_SCHEMA
from modules.deblur import PARAMS_SCHEMA as DEBLUR_PARAMS_SCHEMA
from modules.texture import PARAMS_SCHEMA as TEXTURE_PARAMS_SCHEMA
from modules.clarity import PARAMS_SCHEMA as CLARITY_PARAMS_SCHEMA

import modules as _pkg   # thư mục modules/ (namespace package)


FILTER_CONTROLS = {
    "median": {"fixed": {}, "controls": {
        "kernel_size": {"type": "int", "ui": "slider", "min": 3, "max": 9,
                        "step": 2, "default": 3, "label": "Xóa chấm nhiễu",
                        "desc": "Kéo sang phải để lọc mạnh hơn; có thể mất chi tiết nhỏ."}}},
    "bilateral": {"fixed": {"diameter": 9, "sigma_space": 75.0}, "controls": {
        "sigma_color": {"type": "float", "ui": "slider", "min": 0.1, "max": 255.0,
                        "step": 0.1, "default": 75.0, "label": "Làm mịn giữ viền",
                        "desc": "Kéo sang phải để làm mượt những vùng có màu gần nhau."}}},
    "nlm": {"fixed": {"template_window_size": 7, "search_window_size": 15}, "controls": {
        "h": {"type": "float", "ui": "slider", "min": 0.0, "max": 30.0,
              "step": 0.1, "default": 10.0, "label": "Giảm nhiễu hạt",
              "desc": "0 giữ nguyên ảnh; kéo sang phải giảm hạt mạnh hơn."}}},
    "guided": {"fixed": {"radius": 8}, "controls": {
        "smoothness": {"type": "float", "ui": "slider", "min": 0.0, "max": 100.0,
                       "step": 1.0, "default": 50.0, "label": "Làm mượt theo cấu trúc",
                       "desc": "Kéo sang phải để làm mượt mạnh hơn; 0 là mức thấp nhất.",
                       "param": "eps", "transform": lambda s: 10 ** (-6 + s / 10)}}},
}

SIMPLE_CONTROLS = {
    "clahe": {"fixed": {"tile_grid_x": 8, "tile_grid_y": 8}, "controls": {
        "clip_limit": {"type": "float", "ui": "slider", "min": 0.1, "max": 40.0,
                       "step": 0.1, "default": 2.0, "label": "Tăng tương phản",
                       "desc": "Kéo sang phải để làm rõ khác biệt giữa vùng sáng và tối."}}},
    "gamma_levels": {"fixed": {"black": 0, "white": 255}, "controls": {
        "brightness": {"type": "float", "ui": "slider", "min": -100.0, "max": 100.0,
                       "step": 1.0, "default": 0.0, "label": "Sáng / tối",
                       "desc": "Kéo trái làm tối, kéo phải làm sáng; 0 giữ nguyên ảnh.",
                       "param": "gamma", "transform": lambda s: 2 ** (-s / 50)}}},
    "color": {"fixed": {}, "controls": COLOR_PARAMS_SCHEMA},
    "texture": {"fixed": {}, "controls": TEXTURE_PARAMS_SCHEMA},
    "clarity": {"fixed": {}, "controls": CLARITY_PARAMS_SCHEMA},
    "lighting": {"fixed": {}, "controls": LIGHTING_PARAMS_SCHEMA},
    "sharpen": {"fixed": {}, "controls": SHARPEN_PARAMS_SCHEMA},
    "deblur": {"fixed": {"blur_radius": DEBLUR_PARAMS_SCHEMA["blur_radius"]["default"]},
               "controls": {"deblur_strength": DEBLUR_PARAMS_SCHEMA["deblur_strength"]}},
    "dehaze": {"fixed": {}, "controls": DEHAZE_PARAMS_SCHEMA},
    "vignette": {"fixed": {}, "controls": VIGNETTE_PARAMS_SCHEMA},
    "up_down": {"fixed": {}, "controls": {name: RESIZE_PARAMS_SCHEMA[name]
                for name in ("width", "height")}},
}


def resolve_ui_params(module_id, values, method=None):
    """Đổi slider dễ hiểu sang tham số kỹ thuật; không sửa API của module."""
    if module_id == "filter":
        if not isinstance(method, str) or method not in FILTER_CONTROLS:
            raise ValueError("Hãy chọn bộ lọc hợp lệ")
        profile = FILTER_CONTROLS[method]
    elif module_id == "white_balance":
        profile = {"fixed": {}, "controls": {name: COLOR_PARAMS_SCHEMA[name]
                   for name in ("temperature", "tint")}}
    else:
        profile = SIMPLE_CONTROLS[module_id]
    params = dict(profile["fixed"])
    for name, spec in profile["controls"].items():
        try:
            value = float(values.get(name, spec["default"]))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{spec['label']}: cần giá trị số") from exc
        if not math.isfinite(value) or not spec["min"] <= value <= spec["max"]:
            raise ValueError(f"{spec['label']}: ngoài khoảng cho phép")
        if spec["type"] == "int":
            if not value.is_integer() or (value - spec["min"]) % spec["step"]:
                raise ValueError(f"{spec['label']}: cần giá trị theo bước {spec['step']}")
            value = int(value)
        params[spec.get("param", name)] = spec.get("transform", lambda v: v)(value)
    if module_id == "filter":
        params["filter"] = method
    return params


# ---------------------------------------------------------------- nạp module
def discover():
    found, skipped = {}, []
    for info in pkgutil.iter_modules(_pkg.__path__):
        try:
            m = importlib.import_module(f"modules.{info.name}")
        except Exception as e:
            skipped.append(f"{info.name}: lỗi import ({e})")
            continue
        if all(hasattr(m, a) for a in ("MODULE_ID", "VERSION", "PARAMS_SCHEMA", "process")):
            found[m.MODULE_ID] = m
        else:
            skipped.append(f"{info.name}: thiếu MODULE_ID/VERSION/PARAMS_SCHEMA/process")
    return found, skipped


def editor_image(value):
    """Chuẩn hóa ảnh RGB đầu vào; không dùng ảnh kết quả làm nguồn."""
    if value is None:
        return None
    image = value.get("composite") if isinstance(value, dict) else value
    if image is None and isinstance(value, dict):
        image = value.get("background")
    if image is None:
        return None
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Cần ảnh uint8 hợp lệ")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Cần ảnh RGB hoặc RGBA")
    return np.ascontiguousarray(image[:, :, :3])


def process_pipeline(mods, image, values):
    """Chạy các bộ lọc từ ảnh nguồn RGB, resize ở cuối và trả histogram riêng."""
    source = editor_image(image)
    if source is None:
        return None, None, {"steps": []}
    result = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
    steps = []

    def apply(module_id, params):
        nonlocal result
        if module_id not in mods:
            raise ValueError(f"Không tải được module {module_id}")
        result, meta = mods[module_id].process(result, params)
        steps.append({"module": module_id, "params": params, **meta})

    def apply_adjustment(module_id):
        params = resolve_ui_params(module_id, values)
        if any(params.values()):
            apply(module_id, params)

    for method, profile in FILTER_CONTROLS.items():
        if values.get(f"{method}_enabled", False):
            raw = {name: values.get(f"{method}_{name}", spec["default"])
                   for name, spec in profile["controls"].items()}
            apply("filter", resolve_ui_params("filter", raw, method))
    deblur_params = resolve_ui_params("deblur", values)
    if deblur_params["deblur_strength"] > 0:
        apply("deblur", deblur_params)
    apply_adjustment("dehaze")
    apply("color", resolve_ui_params("color", values))
    apply("gamma_levels", resolve_ui_params("gamma_levels", values))
    apply_adjustment("lighting")
    if values.get("clahe_enabled", False):
        apply("clahe", resolve_ui_params("clahe", values))
    apply_adjustment("clarity")
    apply_adjustment("texture")
    apply_adjustment("sharpen")
    apply_adjustment("vignette")
    resize_params = resolve_ui_params("up_down", values)
    if (resize_params["width"] or result.shape[1],
        resize_params["height"] or result.shape[0]) != (result.shape[1], result.shape[0]):
        apply("up_down", resize_params)
    chart = None
    if values.get("histogram_enabled", False):
        chart, _ = mods["histogram"].process(result, {
            "mode": values.get("histogram_mode", "rgb"),
            "normalize": values.get("histogram_normalize", False),
        })
        chart = cv2.cvtColor(chart, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), chart, {"steps": steps}




# Giao diện dùng chung cho HTTP API và backend Python (ảnh BGR uint8).
MAX_PIXELS = 32_000_000


def control_schema():
    """Các tham số công khai, cùng giới hạn/mặc định với giao diện thử nghiệm."""
    groups = []
    definitions = [
        ("lighting", "Ánh sáng", ("gamma_levels", "lighting", "clahe")),
        ("color", "Màu sắc", ("color",)),
        ("detail", "Chi tiết", ("deblur", "sharpen", "texture")),
        ("effects", "Hiệu ứng", ("clarity", "dehaze", "vignette")),
        ("geometry", "Kích thước ảnh", ("up_down",)),
    ]
    def public_spec(spec):
        return {k: v for k, v in spec.items() if k not in ("param", "transform")}
    for group_id, label, modules_in_group in definitions:
        controls = {}
        for module_id in modules_in_group:
            for name, spec in SIMPLE_CONTROLS[module_id]["controls"].items():
                controls[name] = public_spec(spec)
        if group_id == "lighting":
            controls["clahe_enabled"] = {"type": "bool", "default": False, "label": "Bật tăng tương phản"}
        groups.append({"id": group_id, "label": label, "controls": controls})
    controls = {}
    for method, profile in FILTER_CONTROLS.items():
        label = next(iter(profile["controls"].values()))["label"]
        controls[f"{method}_enabled"] = {"type": "bool", "default": False, "label": f"Bật {label.lower()}"}
        for name, spec in profile["controls"].items():
            controls[f"{method}_{name}"] = public_spec(spec)
    groups.insert(2, {"id": "denoise", "label": "Khử nhiễu / Làm mịn", "controls": controls})
    groups.append({"id": "histogram", "label": "Biểu đồ", "controls": {
        "histogram_enabled": {"type": "bool", "default": False, "label": "Hiện biểu đồ"},
        "histogram_mode": {"type": "enum", "values": ["rgb", "gray"], "default": "rgb", "label": "Biểu đồ"},
        "histogram_normalize": {"type": "bool", "default": False, "label": "Chuẩn hóa biểu đồ"},
    }})
    return groups


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("settings phải là một JSON object")
    schema = {name: spec for group in control_schema() for name, spec in group["controls"].items()}
    unknown = set(settings) - schema.keys()
    if unknown:
        raise ValueError("Tham số không hỗ trợ: " + ", ".join(sorted(unknown)))
    for name, value in settings.items():
        spec = schema[name]
        kind = spec["type"]
        if kind == "bool":
            if type(value) is not bool:
                raise ValueError(f"{name} phải là true hoặc false")
        elif kind == "enum":
            if isinstance(value, bool) or value not in spec["values"]:
                raise ValueError(f"{name}: giá trị không hợp lệ")
        else:
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} phải là số hữu hạn")
            if not spec["min"] <= value <= spec["max"]:
                raise ValueError(f"{name}: ngoài khoảng {spec['min']}..{spec['max']}")
            if kind == "int" and (not float(value).is_integer() or (value - spec["min"]) % spec.get("step", 1)):
                raise ValueError(f"{name}: cần số nguyên theo bước {spec.get('step', 1)}")
    return dict(settings)


def process_image(image, settings=None, *, mods=None):
    """Trả (ảnh BGR, histogram BGR hoặc None, metadata); không đọc/ghi file."""
    values = validate_settings({} if settings is None else settings)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("image phải là ảnh BGR uint8 có ba kênh")
    height, width = image.shape[:2]
    if width * height > MAX_PIXELS:
        raise ValueError("Ảnh nguồn vượt giới hạn 32 triệu pixel")
    output_w = values.get("width", 0) or width
    output_h = values.get("height", 0) or height
    if output_w * output_h > MAX_PIXELS:
        raise ValueError("Ảnh đầu ra vượt giới hạn 32 triệu pixel")
    if mods is None:
        mods, _ = discover()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result, chart, meta = process_pipeline(mods, rgb, values)
    meta.update({"source_size": {"width": width, "height": height},
                 "output_size": {"width": result.shape[1], "height": result.shape[0]}})
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR), (None if chart is None else cv2.cvtColor(chart, cv2.COLOR_RGB2BGR)), meta
