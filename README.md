# video_enhance

Xử lý ảnh và video local. Bốn weight Restormer nằm trong `weights/restormer`: real denoising, deraining, motion deblurring, defocus deblurring. Mạng và phần suy luận nằm trong project này. **Enlight** dùng [EnlightenGAN](https://github.com/arsenyinfo/EnlightenGAN-inference) (ONNX, tải `weights/enlighten.onnx` lần đầu). Super resolution chưa gắn model. Audio của video được giữ nguyên.

## Cài đặt

Cần Python 3.10+ và FFmpeg (`ffmpeg` và `ffprobe` trên `PATH`) để xử lý video.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## CLI

```powershell
video-enhance run input.png -o output.png --device cuda
video-enhance run input.png -o output.png --derain --defocus --device cuda
video-enhance run input.png -o output.png --no-denoise --no-deblur --no-enlight
video-enhance preview input.mp4 --frame 0 -o preview.png
```

Ảnh hỗ trợ png, jpg, jpeg, webp, bmp, tif, tiff. Video hỗ trợ mp4, mov, mkv, avi, webm, m4v. Đuôi file đầu ra phải cùng loại với đầu vào.

## Gradio

```powershell
python app.py
```

Giao diện có bốn model Restormer, Enlight, super resolution, cùng scale, tile, thiết bị và FP16. Có thể đưa ảnh hoặc video. Real denoising và motion deblurring bật sẵn. Deraining và defocus deblurring tắt sẵn vì mỗi model khá nặng. Tile mặc định 512. PyTorch cần bản CUDA, ví dụ `pip install torch --index-url https://download.pytorch.org/whl/cu126`.

## Kiểm thử

```powershell
pytest
```
