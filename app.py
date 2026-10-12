"""Thin Gradio UI over the shared image and video pipeline.

Stage controls stay on screen. Four Restormer tasks, EnlightenGAN, and two HAT x4
super-resolution models are selectable.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
import numpy as np

from video_enhance.io_ffmpeg import FFmpegError, probe, read_frame_at, require_ffmpeg
from video_enhance.io_image import read_image
from video_enhance.media import MediaTypeError
from video_enhance.api import Enhancer
from video_enhance.stages.hat import SCALE, MODELS


def describe_video(video, image, superres: bool, sr_model: str) -> str:
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
    _settings(sr_model, "auto")
    output_scale = SCALE if superres else 1
    model_text = ""
    if superres:
        model_text = f" Model: {_model_label(sr_model)}."
    return details + f"Đầu ra dự kiến: {width * output_scale}×{height * output_scale}.{model_text}"


def preview_video(
    video,
    image,
    denoise: bool,
    derain: bool,
    deblur: bool,
    defocus: bool,
    enlight: bool,
    superres: bool,
    sr_model: str,
    tile: int | float,
    device: str,
    fp16: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    _settings(sr_model, device)
    video_path = _media_path(video)
    image_path = _media_path(image)
    if video_path is not None:
        if not video_path.is_file():
            raise gr.Error(f"Không tìm thấy video: {video_path}")
        original = read_frame_at(video_path, 0)
        enhanced = _process(
            original, denoise, derain, deblur, defocus, enlight, superres, sr_model, tile, device, fp16
        )
        return original, enhanced, "Đã xem trước frame 0."
    if image_path is not None:
        if not image_path.is_file():
            raise gr.Error(f"Không tìm thấy ảnh: {image_path}")
        original = read_image(image_path)
        enhanced = _process(
            original, denoise, derain, deblur, defocus, enlight, superres, sr_model, tile, device, fp16
        )
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
    sr_model: str,
    tile: int | float,
    device: str,
    fp16: bool,
    progress=gr.Progress(),
) -> tuple[str, str]:
    _settings(sr_model, device)
    path = _require_video(video)
    destination = path.with_name(f"{path.stem}-enhanced.mp4")
    enhancer = _enhancer(tile, device, fp16)

    def _callback(done: int, total: int | None) -> None:
        if progress is None:
            return
        if total:
            progress(min(done / total, 1), desc=f"Frame {done}/{total}")
        else:
            progress(0, desc=f"Frame {done}")

    enhancer.run(
        path,
        destination,
        progress=_callback,
        denoise=bool(denoise),
        derain=bool(derain),
        deblur=bool(deblur),
        defocus=bool(defocus),
        enlight=bool(enlight),
        superres=bool(superres),
        sr_model=sr_model,
    )
    return str(destination), f"Đã ghi {destination.name}."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="video enhance") as demo:
        gr.Markdown("# video enhance")
        gr.Markdown(
            "Bốn model Restormer nằm trong weights/restormer: real denoising, deraining, "
            "motion deblurring, defocus deblurring. Enlight dùng EnlightenGAN. "
            "Super resolution x4 chọn HAT-L hoặc Real HAT GAN. Mọi công cụ tắt sẵn, bật đúng công cụ cần dùng. "
            "Mỗi frame xử lý riêng, video có thể nhấp nháy nhẹ."
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
                    denoise = gr.Checkbox(value=False, label="Real denoising")
                    derain = gr.Checkbox(value=False, label="Deraining")
                    deblur = gr.Checkbox(value=False, label="Motion deblurring")
                    defocus = gr.Checkbox(value=False, label="Defocus deblurring")
                with gr.Row():
                    enlight = gr.Checkbox(value=False, label="Enlight")
                    superres = gr.Checkbox(value=False, label="Super resolution x4")
                sr_model = gr.Radio(
                    choices=[(model.label, model.key) for model in MODELS],
                    value="hat-l",
                    label="Model siêu phân giải",
                    interactive=False,
                )
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
            "Cả hai model phóng x4: video 1080p ra khung 4K. HAT-L là bản lớn nhất, 12 nhóm, "
            "huấn luyện ImageNet. Real HAT GAN là bản GAN cho ảnh thật, 6 nhóm — đây là kích thước "
            "lớn nhất mà nhóm tác giả phát hành cho Real HAT GAN. Mỗi lượt chỉ một model trong VRAM: "
            "model chạy xong thì được giải phóng, rồi mới tới model sau. Tile và FP16 giữ VRAM. "
            "Với HAT, ô FP16 chạy bằng BF16 vì FP16 làm tràn số."
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

        task_inputs = [denoise, derain, deblur, defocus, enlight, superres, sr_model, tile, device, fp16]
        preview_inputs = [video, image, *task_inputs]
        render_inputs = [video, *task_inputs]
        for trigger in (video.change, image.change, sr_model.change, superres.change):
            trigger(_guard(describe_video), inputs=[video, image, superres, sr_model], outputs=info)
        superres.change(lambda enabled: gr.update(interactive=enabled), inputs=superres, outputs=sr_model)
        preview_button.click(_guard(preview_video), inputs=preview_inputs, outputs=[original, enhanced, status])
        render_button.click(_guard(render_video), inputs=render_inputs, outputs=[output, status])
    return demo


def main() -> None:
    _install_gradio_file_serving()
    build_demo().queue().launch()


def _install_gradio_file_serving() -> None:
    """Publish an upload before Gradio tells the browser its URL.

    On Windows the player then range-requests that file immediately. If the
    cache move or a virus scan still holds it, the open fails with
    PermissionError and the preview request dies.
    """
    import gradio.route_utils as route_utils
    import gradio.routes as routes
    import gradio.static_server as static_server

    current = route_utils.upload_fn
    if not getattr(current, "_publishes_before_response", False):
        wrapped = _publishing_upload(current)
        route_utils.upload_fn = wrapped
        routes.upload_fn = wrapped
        static_server.upload_fn = wrapped
    if sys.platform == "win32":
        _retry_locked_gradio_reads()


def _publishing_upload(original):
    async def upload_fn(*args, **kwargs):
        kwargs["force_move"] = True
        return await original(*args, **kwargs)

    upload_fn._publishes_before_response = True
    return upload_fn


def _retry_locked_gradio_reads() -> None:
    import anyio
    from gradio._vendor import aiofiles

    if not getattr(anyio.open_file, "_retries_gradio_lock", False):
        anyio.open_file = _wrap_open(anyio.open_file)
    if not getattr(aiofiles.open, "_retries_gradio_lock", False):
        aiofiles.open = _wrap_aiofiles_open(aiofiles.open)


def _wrap_open(original):
    async def open_file(path, *args, **kwargs):
        return await _call_with_permission_retry(original, path, *args, **kwargs)

    open_file._retries_gradio_lock = True
    return open_file


def _wrap_aiofiles_open(original):
    def open_file(path, *args, **kwargs):
        return _RetryingAiofile(original, path, args, kwargs)

    open_file._retries_gradio_lock = True
    return open_file


class _RetryingAiofile:
    """Match aiofiles.open: callable returns an async context manager, not a coroutine."""

    def __init__(self, original, path, args, kwargs, attempts: int = 12, sleep=None):
        self._original = original
        self._path = path
        self._args = args
        self._kwargs = kwargs
        self._attempts = attempts
        self._sleep = sleep
        self._manager = None

    def __await__(self):
        return self._acquire().__await__()

    async def __aenter__(self):
        return await self._acquire()

    async def __aexit__(self, exc_type, exc, tb):
        if self._manager is None:
            return False
        return await self._manager.__aexit__(exc_type, exc, tb)

    async def _acquire(self):
        sleep = self._sleep or asyncio.sleep
        delay = 0.05
        for attempt in range(self._attempts):
            manager = self._original(self._path, *self._args, **self._kwargs)
            try:
                opened = await manager
            except PermissionError:
                if attempt + 1 == self._attempts or not _is_gradio_temp(self._path):
                    raise
                await sleep(delay)
                delay = min(delay * 1.5, 0.4)
                continue
            self._manager = manager
            return opened


async def _call_with_permission_retry(op, path, *args, attempts: int = 12, sleep=None, **kwargs):
    if sleep is None:
        sleep = asyncio.sleep
    delay = 0.05
    for attempt in range(attempts):
        try:
            return await op(path, *args, **kwargs)
        except PermissionError:
            if attempt + 1 == attempts or not _is_gradio_temp(path):
                raise
            await sleep(delay)
            delay = min(delay * 1.5, 0.4)


def _is_gradio_temp(path) -> bool:
    if isinstance(path, int):
        return False
    text = os.fspath(path).replace("/", os.sep)
    return f"{os.sep}gradio{os.sep}" in text


def _process(
    frame: np.ndarray,
    denoise: bool,
    derain: bool,
    deblur: bool,
    defocus: bool,
    enlight: bool,
    superres: bool,
    sr_model: str,
    tile: int | float,
    device: str,
    fp16: bool,
) -> np.ndarray:
    return _enhancer(tile, device, fp16).process_frame(
        frame,
        denoise=bool(denoise),
        derain=bool(derain),
        deblur=bool(deblur),
        defocus=bool(defocus),
        enlight=bool(enlight),
        superres=bool(superres),
        sr_model=sr_model,
    )


def _enhancer(tile: int | float, device: str, fp16: bool) -> Enhancer:
    return Enhancer(device=device, tile=int(tile), fp16=bool(fp16))


def _settings(sr_model: str, device: str) -> None:
    _model_label(sr_model)
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("Thiết bị phải là auto, cuda hoặc cpu.")


def _model_label(sr_model: str) -> str:
    for model in MODELS:
        if model.key == sr_model:
            return model.label
    raise ValueError("Model siêu phân giải phải là HAT-L x4 hoặc Real HAT GAN x4.")


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
