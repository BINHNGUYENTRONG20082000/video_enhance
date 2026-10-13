# video_enhance

Thư viện xử lý ảnh và video chạy local. Ba nhóm model được gắn sẵn: Restormer cho khôi phục, EnlightenGAN cho làm sáng, và HAT cho siêu phân giải x4. Giao diện Gradio và lệnh `video-enhance` đều gọi cùng một API, lớp `Enhancer`.

Mỗi model được nạp, chạy, rồi giải phóng trước khi model tiếp theo bắt đầu, nên VRAM chỉ chứa một model tại một thời điểm. Video giữ nguyên audio của file gốc.

Nguồn mã, giấy phép và mã băm weight nằm trong [SOURCES.md](SOURCES.md).

## Model

| Công cụ | Model | Tác dụng | Đầu ra |
| --- | --- | --- | --- |
| `denoise` | Restormer real denoising | Khử nhiễu ảnh thực | Giữ kích thước |
| `derain` | Restormer deraining | Khử mưa | Giữ kích thước |
| `deblur` | Restormer motion deblurring | Khử nhòe chuyển động | Giữ kích thước |
| `defocus` | Restormer defocus deblurring | Khử nhòe ngoài tiêu cự | Giữ kích thước |
| `enlight` | EnlightenGAN | Làm sáng vùng tối | Giữ kích thước |
| `superres` | HAT-L x4 hoặc Real HAT GAN x4 | Siêu phân giải | Phóng x4 |

Khi bật nhiều công cụ trong một lần gọi, thứ tự cố định là denoise, derain, deblur, defocus, enlight, rồi super resolution.

HAT-L là bản lớn nhất của HAT cổ điển: 12 nhóm, weight ImageNet. Real HAT GAN là bản GAN cho ảnh thật: 6 nhóm. Không có bản Real HAT GAN kích thước HAT-L. Cả hai chỉ phóng x4.

## Yêu cầu

- Python 3.10 trở lên
- FFmpeg trên `PATH`, gồm `ffmpeg` và `ffprobe`, có encoder `libx264` và `libx264rgb`
- GPU NVIDIA nếu chạy CUDA. CPU dùng được. HAT-L và Real HAT GAN trên CPU rất chậm

`libx264rgb` chỉ cần khi một lần chạy dùng từ hai model trở lên. File trung gian được ghi lossless để model sau nhận đúng pixel. File cuối là H.264 CRF 18, `yuv420p`.

## Cài đặt

Cài PyTorch bản CUDA trước, sau đó cài project. Pip không thay thế một bản PyTorch đã thỏa dependency.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[ui,dev]"
```

| Lệnh | Nội dung |
| --- | --- |
| `pip install -e .` | Thư viện, đủ để nhúng vào phần mềm khác |
| `pip install -e ".[ui]"` | Thêm giao diện Gradio |
| `pip install -e ".[dev]"` | Thêm pytest và Gradio |

Thư viện kéo theo OpenCV, ONNX Runtime GPU và các wheel CUDA dùng cho EnlightenGAN trên Windows (`nvidia-cublas`, `nvidia-cudnn`, `nvidia-cuda-runtime`, `nvidia-cuda-nvrtc`).

Kiểm tra môi trường:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest
```

## Weight

Thư mục `weights/` không được commit. Khi bàn giao, copy cả thư mục này theo mã nguồn.

| Đường dẫn | Tự tải nếu thiếu |
| --- | --- |
| `weights/restormer/real_denoising.pth` | Không |
| `weights/restormer/deraining.pth` | Không |
| `weights/restormer/motion_deblurring.pth` | Không |
| `weights/restormer/single_image_defocus_deblurring.pth` | Không |
| `weights/hat/HAT-L_SRx4_ImageNet-pretrain.pth` | Có, kèm kiểm SHA-256 |
| `weights/hat/Real_HAT_GAN_SRx4.pth` | Có, kèm kiểm SHA-256 |
| `weights/enlighten.onnx` | Có |

File HAT bị coi là thiếu khi dung lượng dưới 150 MB. File đủ dung lượng nhưng hỏng thì xóa rồi chạy lại. URL và mã băm nằm trong [SOURCES.md](SOURCES.md).

Thư mục mặc định là `weights/` ở gốc checkout. Phần mềm nhúng chọn chỗ khác bằng đối số hoặc biến môi trường:

```powershell
$env:VIDEO_ENHANCE_WEIGHTS = "D:\models\video_enhance"
```

Biến này trỏ tới thư mục cha của `restormer/`, `hat/` và `enlighten.onnx`.

## Dùng như thư viện

Ảnh đầu vào là mảng NumPy RGB, `uint8`, dạng `(cao, rộng, 3)`.

```python
from video_enhance import Enhancer

with Enhancer(device="cuda", tile=512, fp16=True, weights_dir=r"D:\models\video_enhance") as enhancer:
    restored = enhancer.process_frame(frame, denoise=True, deblur=True)
    enlarged = enhancer.process_frame(frame, superres=True, sr_model="real-hat-gan")
    enhancer.preview("input.mp4", frame=0, deblur=True)
    enhancer.run("input.mp4", "output.mp4", enlight=True, superres=True, sr_model="hat-l")
```

| Phương thức | Kết quả |
| --- | --- |
| `process_frame` | Trả về một mảng mới |
| `preview` | Xử lý một frame video, hoặc cả một ảnh |
| `run` | Ghi file. Đuôi đầu ra phải cùng loại với đầu vào |

`device` nhận `auto`, `cuda` hoặc `cpu`. `sr_model` nhận `hat-l` hoặc `real-hat-gan`, và chỉ có tác dụng khi `superres=True`. `tile` là cạnh mỗi mảnh, mặc định 512. `tile=0` đưa cả khung vào model một lần. `fp16=True` chạy Restormer ở FP16. HAT dùng BF16 trên GPU hỗ trợ BF16, vì FP16 làm tràn số. GPU không có BF16 thì HAT ở lại FP32.

Mọi công cụ mặc định tắt. Mỗi lần gọi tự đóng model khi xong, kể cả khi có lỗi.

## Dòng lệnh

```powershell
video-enhance run input.png -o output.png --denoise --device cuda
video-enhance run input.mp4 -o output.mp4 --deblur --enlight --superres --sr-model hat-l
video-enhance preview input.mp4 --frame 0 --deblur -o preview.png
```

Không ghi đè lên chính file đầu vào.

| Cờ | Mặc định |
| --- | --- |
| `--denoise`, `--derain`, `--deblur`, `--defocus`, `--enlight`, `--superres` | Tắt. Dạng phủ định là `--no-denoise` |
| `--sr-model {hat-l,real-hat-gan}` | `hat-l` |
| `--device {auto,cuda,cpu}` | `auto` |
| `--tile` | `512` |
| `--fp16` / `--no-fp16` | Bật |
| `--config` | Không dùng để chọn model. File YAML chỉ được chấp nhận khi rỗng |

Ảnh nhận png, jpg, jpeg, webp, bmp, tif, tiff. Video nhận mp4, mov, mkv, avi, webm, m4v.

## Giao diện

```powershell
python app.py
```

Mọi công cụ tắt sẵn. Bật Super resolution thì mới chọn được HAT-L x4 hoặc Real HAT GAN x4. Video kết quả được ghi cạnh file đầu vào, với hậu tố `-enhanced.mp4`.

## Bố cục mã

```text
app.py                              Giao diện Gradio
src/video_enhance/api.py            Enhancer
src/video_enhance/pipeline.py       Điều phối từng model và encode video
src/video_enhance/stages/           Restormer, EnlightenGAN, HAT
src/video_enhance/io_image.py       Đọc và ghi ảnh RGB
src/video_enhance/io_ffmpeg.py      Giải mã, mã hóa, copy audio
src/video_enhance/paths.py          Định vị thư mục weights
src/video_enhance/cli.py            Lệnh video-enhance
tests/                              pytest
```

Mạng Restormer và HAT nằm trong repository này. Không cần cài repo gốc hay BasicSR.

## Kiểm thử

```powershell
pytest
```

Bài kiểm tra weight được bỏ qua khi thiếu checkpoint. Bài kiểm tra video được bỏ qua khi không có FFmpeg.

## Giới hạn

- Từng frame được xử lý độc lập, nên video có thể nhấp nháy nhẹ.
- Siêu phân giải chỉ có hệ số x4.
- `configs/default.yaml` chưa điều khiển pipeline. Thêm key vào file này sẽ bị từ chối.
- Weight Real HAT GAN không tương thích với kiến trúc 12 nhóm của HAT-L.
