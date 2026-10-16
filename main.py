"""Giao diện Gradio gọi Photo Filters API; API chạy ở tiến trình riêng."""
import gradio as gr
from api.api_client import ImageApiClient, ImageApiError, editor_image


def make_input(name, spec):
    kind = spec["type"]
    label = spec.get("label", name) + (f" ({spec['unit']})" if spec.get("unit") else "")
    common = {"label": label, "info": spec.get("desc"), "value": spec["default"]}
    if kind == "enum":
        return gr.Dropdown(choices=spec["values"], **common)
    if kind == "bool":
        return gr.Checkbox(**common)
    if kind in ("float", "int"):
        if spec.get("ui") == "number":
            return gr.Number(precision=0 if kind == "int" else None,
                             minimum=spec.get("min"), maximum=spec.get("max"), **common)
        return gr.Slider(spec["min"], spec["max"], step=spec.get("step", 1 if kind == "int" else 0.1), **common)
    raise ImageApiError(f"API trả loại control chưa hỗ trợ: {name} ({kind})")


def build(api_client=None):
    client = api_client if api_client is not None else ImageApiClient()
    schema = client.configuration()
    groups = schema["groups"]
    control_specs = {name: spec for group in groups for name, spec in group["controls"].items()}
    required = {"width", "height"}
    if not required <= control_specs.keys():
        raise ImageApiError("API thiếu các tham số chiều rộng và chiều cao")

    with gr.Blocks(title="Chỉnh sửa ảnh") as demo:
        gr.Markdown("## Chỉnh sửa ảnh\nTải ảnh một lần, kết hợp các chỉnh sửa và xem kết quả ngay.")
        controls = {}

        def add_control(name, spec):
            comp = make_input(name, spec)
            if name.startswith("guided_") and not schema.get("capabilities", {}).get("guided_available", False):
                comp.interactive = False
                comp.info = "API chưa có Guided Filter; cần opencv-contrib-python."
            controls[name] = comp

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                inp = gr.Image(label="Ảnh gốc", type="numpy", sources=["upload"], format="png")
                for group in groups:
                    if group["id"] == "histogram":
                        continue
                    with gr.Accordion(group["label"], open=group["id"] in {"lighting", "color", "denoise", "detail"}):
                        for name, spec in group["controls"].items():
                            add_control(name, spec)
                reset = gr.Button("Đặt lại tất cả")
            with gr.Column(scale=2):
                out = gr.Image(label="Kết quả", type="numpy", interactive=False,
                               format="png", buttons=["download", "fullscreen"])
                for group in groups:
                    if group["id"] == "histogram":
                        with gr.Accordion(group["label"], open=False):
                            for name, spec in group["controls"].items():
                                add_control(name, spec)
                            chart = gr.Image(label="Phân bố độ sáng / màu", interactive=False, format="png")
                if not any(group["id"] == "histogram" for group in groups):
                    chart = gr.Image(visible=False)
                status = gr.Markdown()
                with gr.Accordion("Thông tin xử lý", open=False):
                    info = gr.JSON(label="Các chỉnh sửa đã áp dụng")

        names = list(controls)
        comps = list(controls.values())
        defaults = [control_specs[name]["default"] for name in names]
        inputs = [inp] + comps
        outputs = [out, chart, info, status]

        def settings_from(vals):
            values = dict(zip(names, vals))
            # Number/Slider có thể trả 30.0; không làm tròn giá trị sai như 30.5.
            for name, value in values.items():
                if control_specs[name]["type"] == "int" and isinstance(value, (int, float)) and not isinstance(value, bool):
                    if float(value).is_integer():
                        values[name] = int(value)
            return values

        def update_result(image, *vals):
            try:
                source = editor_image(image)
                if source is None:
                    return None, None, {"steps": []}, ""
                values = settings_from(vals)
                result, histogram, meta = client.process(source, values)
                return result, histogram, meta, f"Xử lý qua API: {meta['processing_ms']} ms"
            except (ImageApiError, ValueError, TypeError) as exc:
                return None, None, {"lỗi": str(exc)}, f"Không thể xử lý ảnh: {exc}"

        def dimension_values(source, vals):
            dims = [0, 0] if source is None else [source.shape[1], source.shape[0]]
            vals = list(vals)
            for name, value in zip(("width", "height"), dims):
                vals[names.index(name)] = value
            return vals, dims

        def refresh_source(image, *vals):
            try:
                source = editor_image(image)
                vals, dims = dimension_values(source, vals)
                return list(update_result(image, *vals)) + dims
            except (ImageApiError, ValueError, TypeError) as exc:
                return [None, None, {"lỗi": str(exc)}, f"Không thể đọc ảnh: {exc}", gr.skip(), gr.skip()]

        def reset_all(image):
            refreshed = refresh_source(image, *defaults)
            reset_values = list(defaults)
            for name, value in zip(("width", "height"), refreshed[-2:]):
                if isinstance(value, (int, float)):
                    reset_values[names.index(name)] = value
            return reset_values + refreshed[:-2]

        events = dict(show_progress="hidden", trigger_mode="always_last",
                      concurrency_id="editor", concurrency_limit=1)
        dimensions = [controls["width"], controls["height"]]
        inp.change(refresh_source, [inp] + comps, outputs + dimensions, **events)
        for comp in controls.values():
            comp.input(update_result, inputs, outputs, **events)
        reset.click(reset_all, [inp], comps + outputs, **events)
    return demo


if __name__ == "__main__":
    try:
        demo = build()
    except ImageApiError as exc:
        raise SystemExit(str(exc)) from exc
    demo.launch()
