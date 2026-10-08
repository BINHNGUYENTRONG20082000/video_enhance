"""Thin Gradio UI over the shared image and video pipeline.

Stage controls stay on screen. Four Restormer tasks plus EnlightenGAN are selectable.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
import numpy as np

from video_enhance.io_ffmpeg import FFmpegError, probe, read_frame_at, require_ffmpeg
from video_enhance.io_image import read_image
from video_enhance.media import MediaTypeError
from video_enhance.pipeline import build_pipeline

_CACHED = None
_CACHE_KEY = None
_CACHE_LOCK = threading.Lock()


def describe_video(video, image, superres: bool, scale: int | float | str) -> str:
    video_path = _media_path(video)
    if video_path is not None:
        require_ffmpeg()
        info = probe(video_path)
        width, height = info.width, info.height
        frame_text = str(info.frame_count) if info.frame_count is not None else "?"
        audio_text = "có audio" if info.has_audio else "không có audio"
        details = f"{width}×{height}, {info.fps:.3f} fps, {frame_text} frames, {audio_text}. "
    else:
        image_path = _media_path(image)
        if image_path is None:
            return ""
        frame = read_image(image_path)
        height, width = frame.shape[:2]
        details = f"{width}×{height}. "
    output_scale = int(scale) if superres else 1
    return details + f"Đầu ra dự kiến: {width * output_scale}×{height * output_scale}."


def preview_video(
    video,
    image,
    denoise: bool,
    derain: bool,
    deblur: bool,
    defocus: bool,
    enlight: bool,
    superres: bool,
    scale: int | float | str,
    tile: int | float,
    device: str,
    fp16: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    _settings(scale, device)
    video_path = _media_path(video)
    image_path = _media_path(image)
    if video_path is not None:
        if not video_path.is_file():
            raise gr.Error(f"Không tìm thấy video: {video_path}")
        original = read_frame_at(video_path, 0)
        enhanced = _process(original, denoise, derain, deblur, defocus, enlight, tile, device, fp16)
        return original, enhanced, "Đã xem trước frame 0."
    if image_path is not None:
        if not image_path.is_file():
            raise gr.Error(f"Không tìm thấy ảnh: {image_path}")
        original = read_image(image_path)
        enhanced = _process(original, denoise, derain, deblur, defocus, enlight, tile, device, fp16)
        return original, enhanced, "Đã xem trước ảnh."
    raise gr.Error("Hãy chọn một video hoặc một ảnh.")


def render_video(
    video,
    denoise: bool,
    derain: bool,
    deblur: bool,
    defocus: bool,
    enlight: bool,
    superres: bool,
    scale: int | float | str,
    tile: int | float,
    device: str,
    fp16: bool,
    progress=gr.Progress(),
) -> tuple[str, str]:
    _settings(scale, device)
    del superres
    path = _require_video(video)
    destination = path.with_name(f"{path.stem}-enhanced.mp4")
    pipeline = _pipeline(denoise, derain, deblur, defocus, enlight, tile, device, fp16)

    def _callback(done: int, total: int | None) -> None:
        if progress is None:
            return
        if total:
            progress(min(done / total, 1), desc=f"Frame {done}/{total}")
        else:
            progress(0, desc=f"Frame {done}")

    pipeline.run_video(path, destination, progress=_callback)
    return str(destination), f"Đã ghi {destination.name}."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="video enhance") as demo:
        gr.Markdown("# video enhance")
        gr.Markdown(
            "Bốn model Restormer nằm trong weights/restormer: real denoising, deraining, "
            "motion deblurring, defocus deblurring. Enlight dùng EnlightenGAN. "
            "Super resolution chưa gắn model. Mỗi frame xử lý riêng, video có thể nhấp nháy nhẹ."
        )
        ffmpeg_note = _ffmpeg_note()
        if ffmpeg_note:
            gr.Markdown(ffmpeg_note)

        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Video đầu vào", sources=["upload"])
                image = gr.Image(label="Ảnh đầu vào", type="filepath", sources=["upload"], format="png")
            with gr.Column():
                with gr.Row():
                    denoise = gr.Checkbox(value=True, label="Real denoising")
                    derain = gr.Checkbox(value=False, label="Deraining")
                    deblur = gr.Checkbox(value=True, label="Motion deblurring")
                    defocus = gr.Checkbox(value=False, label="Defocus deblurring")
                with gr.Row():
                    enlight = gr.Checkbox(value=True, label="Enlight")
                    superres = gr.Checkbox(value=True, label="Super resolution")
                scale = gr.Radio(choices=[2, 4], value=2, label="Scale siêu phân giải")
                tile = gr.Slider(
                    minimum=0,
                    maximum=4096,
                    value=512,
                    step=64,
                    label="Tile",
                    info="Cả khung hình vẫn được xử lý. Tile là cạnh mỗi mảnh; các mảnh chồng nhau rồi ghép lại. 0 đưa cả khung vào model một lần.",
                )
                device = gr.Radio(choices=["auto", "cuda", "cpu"], value="auto", label="Thiết bị")
                fp16 = gr.Checkbox(value=True, label="FP16 trên CUDA")

        gr.Markdown(
            "Video 1080p với scale 4 ra khung 4K. Tile và FP16 là cách giữ VRAM. Mặc định scale là 2."
        )
        info = gr.Textbox(label="Thông tin", interactive=False)
        status = gr.Textbox(label="Trạng thái", interactive=False)
        with gr.Row():
            preview_button = gr.Button("Xem trước một frame", variant="primary")
            render_button = gr.Button("Xử lý cả video")
        with gr.Row():
            original = gr.Image(label="Gốc", type="numpy", format="png")
            enhanced = gr.Image(label="Đã xử lý", type="numpy", format="png")
        output = gr.Video(label="Video đã xử lý")

        task_inputs = [denoise, derain, deblur, defocus, enlight, superres, scale, tile, device, fp16]
        preview_inputs = [video, image, *task_inputs]
        render_inputs = [video, *task_inputs]
        for trigger in (video.change, image.change, scale.change, superres.change):
            trigger(_guard(describe_video), inputs=[video, image, superres, scale], outputs=info)
        superres.change(lambda enabled: gr.update(interactive=enabled), inputs=superres, outputs=scale)
        preview_button.click(_guard(preview_video), inputs=preview_inputs, outputs=[original, enhanced, status])
        render_button.click(_guard(render_video), inputs=render_inputs, outputs=[output, status])
    return demo


def main() -> None:
    build_demo().queue().launch()


def _process(
    frame: np.ndarray,
    denoise: bool,
    derain: bool,
    deblur: bool,
    defocus: bool,
    enlight: bool,
    tile: int | float,
    device: str,
    fp16: bool,
) -> np.ndarray:
    return _pipeline(denoise, derain, deblur, defocus, enlight, tile, device, fp16).process_frame(frame)


def _pipeline(
    denoise: bool,
    derain: bool,
    deblur: bool,
    defocus: bool,
    enlight: bool,
    tile: int | float,
    device: str,
    fp16: bool,
):
    global _CACHED, _CACHE_KEY
    key = (
        bool(denoise),
        bool(derain),
        bool(deblur),
        bool(defocus),
        bool(enlight),
        int(tile),
        device,
        bool(fp16),
    )
    with _CACHE_LOCK:
        if _CACHED is not None and _CACHE_KEY == key:
            return _CACHED
        if _CACHED is not None:
            _CACHED.close()
        pipeline = build_pipeline(
            denoise=bool(denoise),
            derain=bool(derain),
            deblur=bool(deblur),
            defocus=bool(defocus),
            enlight=bool(enlight),
            device=device,
            tile=int(tile),
            fp16=bool(fp16),
        )
        pipeline.load()
        _CACHED = pipeline
        _CACHE_KEY = key
        return pipeline


def _settings(scale: int | float | str, device: str) -> None:
    chosen = int(scale)
    if chosen not in (2, 4):
        raise ValueError("Scale phải là 2 hoặc 4.")
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("Thiết bị phải là auto, cuda hoặc cpu.")


def _media_path(value) -> Path | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        value = value.get("video") or value.get("path") or value.get("name")
    if value is None:
        return None
    return Path(str(value))


def _require_video(video) -> Path:
    path = _media_path(video)
    if path is None:
        raise gr.Error("Hãy chọn một video.")
    if not path.is_file():
        raise gr.Error(f"Không tìm thấy video: {path}")
    return path


def _ffmpeg_note() -> str:
    try:
        require_ffmpeg()
    except FFmpegError as exc:
        return f"**FFmpeg:** {exc}"
    return ""


def _guard(func):
    def _wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except gr.Error as exc:
            raise gr.Error(str(exc), title=str(exc), print_exception=False) from exc
        except (FFmpegError, MediaTypeError, RuntimeError, OSError, ValueError) as exc:
            raise gr.Error(str(exc), title=str(exc), print_exception=False) from exc

    return _wrapped


if __name__ == "__main__":
    main()
