# video_enhance

Pipeline local để khôi phục và phóng to ảnh hoặc video. CLI và giao diện Gradio dùng chung một pipeline. Mỗi công cụ là một model riêng. Tại một thời điểm chỉ một model nằm trong VRAM: model chạy xong thì được giải phóng, rồi model kế tiếp mới được nạp.

Audio của video được giữ từ file gốc và gắn lại ở lần ghi cuối. Phần mềm khác tích hợp bằng lớp `Enhancer`. Giao diện Gradio và lệnh `video-enhance` chỉ là hai cách gọi lớp đó.

## Cài đặt

Cần:

- Python 3.10 trở lên
- FFmpeg trên `PATH`, gồm cả `ffmpeg` và `ffprobe`. Bản FFmpeg phải có encoder `libx264` và `libx264rgb`. `libx264rgb` dùng cho file trung gian khi một lượt chạy nhiều model.
- GPU NVIDIA nếu muốn chạy CUDA. CPU vẫn chạy được, nhưng HAT-L và Real HAT GAN rất chậm.

Cài bản PyTorch CUDA trước, rồi mới cài project. Pip mặc định không thay wheel đã thỏa dependency, nên thứ tự này giữ lại bản CUDA.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[ui,dev]"
```

`pip install -e .` chỉ cài thư viện. `.[ui]` thêm Gradio. `.[dev]` thêm pytest và Gradio. OpenCV, ONNX Runtime GPU và các gói CUDA DLL (`nvidia-cublas`, `nvidia-cudnn`, `nvidia-cuda-runtime`, `nvidia-cuda-nvrtc`) nằm trong thư viện. Trên Windows, EnlightenGAN tự đăng ký các thư mục DLL đó trước khi tạo session ONNX.

Kiểm tra:

```powershell
ffmpeg -hide_banner -encoders
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest
```

## Weight

Thư mục `weights/` không nằm trong git. Máy bàn giao cần copy nguyên thư mục này, hoặc để HAT và Enlighten tự tải khi chạy lần đầu. Restormer không tự tải.

| File | Công cụ | Cách có file |
| --- | --- | --- |
| `weights/restormer/real_denoising.pth` | Real denoising | Copy checkpoint chính thức |
| `weights/restormer/deraining.pth` | Deraining | Copy checkpoint chính thức |
| `weights/restormer/motion_deblurring.pth` | Motion deblurring | Copy checkpoint chính thức |
| `weights/restormer/single_image_defocus_deblurring.pth` | Defocus deblurring | Copy checkpoint chính thức |
| `weights/hat/HAT-L_SRx4_ImageNet-pretrain.pth` | HAT-L x4 | Đã có, hoặc tự tải và kiểm SHA-256 |
| `weights/hat/Real_HAT_GAN_SRx4.pth` | Real HAT GAN x4 | Đã có, hoặc tự tải và kiểm SHA-256 |
| `weights/enlighten.onnx` | Enlight | Đã có, hoặc tự tải lần đầu |

Nếu file HAT thiếu hoặc nhỏ hơn 150 MB, `ensure_weights` tải lại và đối chiếu SHA-256. File đủ lớn nhưng hỏng thì xóa đi rồi chạy lại. Nguồn và mã hash nằm trong [SOURCES.md](SOURCES.md).

## Tích hợp vào phần mềm khác

```python
from video_enhance import Enhancer
import numpy as np

frame = np.zeros((64, 64, 3), dtype=np.uint8)  # RGB, uint8, (cao, rộng, 3)

with Enhancer(device="cuda", tile=512, fp16=True) as enhancer:
    restored = enhancer.process_frame(frame, denoise=True, deblur=True)
    enlarged = enhancer.process_frame(frame, superres=True, sr_model="real-hat-gan")
    enhancer.run("input.mp4", "output.mp4", enlight=True, superres=True, sr_model="hat-l")
```

`process_frame` trả về một mảng mới. `run` ghi file. `preview(path, frame=0)` xử lý một frame của video, hoặc cả một ảnh. Mỗi lần gọi tự nạp model, chạy, rồi giải phóng VRAM. Không cần giữ model giữa hai lần gọi.

Weight mặc định là thư mục `weights` của checkout này. Phần mềm nhúng có thể trỏ chỗ khác:

```python
Enhancer(weights_dir=r"D:\models\video_enhance")
```

Hoặc đặt biến môi trường `VIDEO_ENHANCE_WEIGHTS` tới thư mục cha của `restormer`, `hat` và `enlighten.onnx`.

`sr_model` nhận `hat-l` hoặc `real-hat-gan`. Cả hai phóng x4. `device` nhận `auto`, `cuda`, `cpu`. `tile=0` đưa cả khung vào model.

## Chạy

Mọi công cụ mặc định tắt. Bật đúng công cụ cần dùng.

```powershell
python app.py
```

Giao diện nhận ảnh (png, jpg, jpeg, webp, bmp, tif, tiff) hoặc video (mp4, mov, mkv, avi, webm, m4v). Video đã xử lý ghi cạnh file đầu vào, tên `*-enhanced.mp4`.

```powershell
video-enhance run input.png -o output.png --denoise --device cuda
video-enhance run input.png -o output.png --superres --sr-model hat-l --device cuda
video-enhance run input.png -o output.png --superres --sr-model real-hat-gan --device cuda
video-enhance run input.mp4 -o output.mp4 --deblur --enlight --device cuda
video-enhance preview input.mp4 --frame 0 --deblur -o preview.png
```

Đuôi file đầu ra phải cùng loại với đầu vào. Không ghi đè lên chính file đầu vào.

Cờ CLI:

| Cờ | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--denoise` / `--no-denoise` | tắt | Restormer real denoising |
| `--derain` / `--no-derain` | tắt | Restormer deraining |
| `--deblur` / `--no-deblur` | tắt | Restormer motion deblurring |
| `--defocus` / `--no-defocus` | tắt | Restormer defocus deblurring |
| `--enlight` / `--no-enlight` | tắt | EnlightenGAN |
| `--superres` / `--no-superres` | tắt | HAT x4 |
| `--sr-model` | `hat-l` | `hat-l` hoặc `real-hat-gan` |
| `--device` | `auto` | `auto`, `cuda`, `cpu` |
| `--tile` | `512` | Cạnh mỗi mảnh. `0` đưa cả khung vào model |
| `--fp16` / `--no-fp16` | bật | Restormer dùng FP16 trên CUDA. HAT dùng BF16 |
| `--config` | không | Chỉ kiểm tra file YAML rỗng. Không chọn model |

Thứ tự khi bật nhiều công cụ: denoise, derain, deblur, defocus, enlight, rồi super resolution. Super resolution luôn phóng x4. Không có model x2.

## Cấu trúc

```text
app.py                         Giao diện Gradio
src/video_enhance/api.py       Lớp Enhancer, API để nhúng
src/video_enhance/paths.py     Tìm thư mục weights
src/video_enhance/cli.py       Lệnh video-enhance
src/video_enhance/pipeline.py  Ghép stage, xử lý ảnh và video
src/video_enhance/media.py     Phân loại đuôi ảnh / video
src/video_enhance/io_image.py  Đọc ghi ảnh RGB bằng OpenCV
src/video_enhance/io_ffmpeg.py Giải mã, mã hóa, giữ audio
src/video_enhance/config.py    Đọc YAML. Chưa gắn vào pipeline
src/video_enhance/stages/base.py        Giao diện load / process / close
src/video_enhance/stages/restormer.py   Bốn tác vụ Restormer
src/video_enhance/stages/restormer_arch.py
src/video_enhance/stages/enlighten.py   EnlightenGAN ONNX
src/video_enhance/stages/hat.py         HAT-L x4 và Real HAT GAN x4
src/video_enhance/stages/hat_arch.py    Mạng HAT, không phụ thuộc BasicSR
configs/default.yaml           YAML rỗng, được chấp nhận
tests/                         pytest
weights/                       Checkpoint, không commit
```

`build_pipeline` tạo danh sách stage. `Pipeline.process_frame` dùng cho ảnh và cho xem trước một frame: nạp một stage, xử lý, `close`, giải phóng CUDA, rồi mới tới stage sau.

Video đi theo từng lượt. Mỗi lượt nạp một model, đọc hết frame, ghi ra file, rồi đóng model. Lượt chưa phải cuối ghi MP4 lossless RGB (`libx264rgb`, `-qp 0`) để model sau nhận đúng pixel. Lượt cuối ghi H.264 CRF 18, pixel `yuv420p`, và copy audio từ video gốc. File trung gian có dạng `.ten.passN.mp4` và bị xóa khi xong. Nếu Windows còn khóa file, lần xóa có thể bỏ qua và file ẩn đó còn lại cạnh đầu ra.

`Pipeline.load()` cố ý không nạp model. Model chỉ nạp bên trong lúc xử lý.

Tile 512 là cạnh mỗi mảnh, không phải số mảnh. Restormer chồng 32 pixel rồi lấy trung bình. HAT lấy thêm viền 32 pixel rồi cắt, vì cửa sổ attention là 16. `0` xử lý cả khung một lần.

Ô FP16 trên giao diện là một công tắc. Restormer chạy `float16`. HAT không chạy `float16` vì activation vượt 65504 và ra NaN. Trên GPU có BF16, HAT chạy `bfloat16`. Không có BF16 thì HAT ở lại `float32`.

Giao diện Gradio vá hai lỗi mở file trên Windows: upload được công bố trước khi trả URL, và lần đọc file tạm bị `PermissionError` thì thử lại tối đa 12 lần. Vá này nằm trong `app.py`, không nằm trong pipeline.

## Kiểm thử

```powershell
pytest
```

Test weight bỏ qua nếu thiếu checkpoint. Test video bỏ qua nếu không có FFmpeg.

## Giới hạn

- Mỗi frame xử lý độc lập. Video có thể nhấp nháy nhẹ giữa các frame.
- `configs/default.yaml` và `--config` chưa điều khiển model. Thêm key vào YAML sẽ bị từ chối.
- Real HAT GAN chính thức chỉ có kiến trúc HAT 6 nhóm. Không có weight Real HAT GAN bản L. Đừng nạp weight GAN vào mạng 12 nhóm của HAT-L.
- Phóng đại chỉ có x4.
- Nguồn mã và giấy phép nằm trong [SOURCES.md](SOURCES.md).
