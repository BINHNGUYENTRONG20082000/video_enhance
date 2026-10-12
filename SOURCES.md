# Nguồn tham khảo

Project này tự chứa phần suy luận. Không cần cài repo gốc, BasicSR, hay package suy luận EnlightenGAN. Weight không được train lại ở đây. Chúng là checkpoint đã phát hành của từng nhóm tác giả.

## Restormer

- Bài báo: [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881), CVPR 2022.
- Tác giả: Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang.
- Mã gốc: [swz30/Restormer](https://github.com/swz30/Restormer), MIT License, copyright 2022 Syed Waqas Zamir and contributors.
- Trong repo này: `src/video_enhance/stages/restormer_arch.py` là mạng đã chuyển vào project. `src/video_enhance/stages/restormer.py` là phần nạp weight, chia tile và ghép tile, viết cho pipeline này.
- Bốn weight trong `weights/restormer/` là checkpoint chính thức, không tự tải:
  - `real_denoising.pth`, LayerNorm `BiasFree`
  - `deraining.pth`, LayerNorm `WithBias`
  - `motion_deblurring.pth`, LayerNorm `WithBias`
  - `single_image_defocus_deblurring.pth`, LayerNorm `WithBias`
- Kiến trúc suy luận chung: `dim=48`, `num_blocks=[4, 6, 6, 8]`, `num_refinement_blocks=4`, `heads=[1, 2, 4, 8]`, `ffn_expansion_factor=2.66`.

## HAT

- Bài báo: [Activating More Pixels in Image Super-Resolution Transformer](https://arxiv.org/abs/2205.04437), CVPR 2023. Bản mở rộng: [HAT: Hybrid Attention Transformer for Image Restoration](https://arxiv.org/abs/2309.05239).
- Tác giả: Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, Chao Dong. Bản mở rộng thêm Wenlong Zhang và Xiangtao Kong.
- Mã gốc: [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT), Apache License 2.0, copyright 2023 Xiangyu Chen.
- Trong repo này: `src/video_enhance/stages/hat_arch.py` lấy từ `hat/archs/hat_arch.py`. Phần registry và helper của BasicSR đã bỏ, `torch.meshgrid` dùng `indexing="ij"`, và mask attention được đưa về đúng dtype của input để chạy half precision. `src/video_enhance/stages/hat.py` là phần nạp weight và chia tile, viết cho pipeline này.
- Không dùng code train, discriminator, hay `options/` của repo gốc. Cấu hình mạng chép từ hai file test chính thức:
  - `options/test/HAT-L_SRx4_ImageNet-pretrain.yml`: 12 nhóm, `depths` và `num_heads` đều là mười hai số 6, `embed_dim=180`, `window_size=16`, `upscale=4`.
  - `options/test/HAT_GAN_Real_SRx4.yml`: 6 nhóm, cùng `embed_dim`, cửa sổ và scale. Đây là kích thước lớn nhất mà nhóm tác giả phát hành cho Real HAT GAN. Không có bản GAN kiểu HAT-L.
- Cả hai checkpoint dùng key `params_ema`.
- Weight gốc nằm trên Google Drive của nhóm tác giả: [thư mục pretrained](https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0?usp=sharing). Bản trong `weights/hat/` tải từ mirror Hugging Face `jaideepsingh/upscale_models`, cùng tên file và đã đối chiếu SHA-256:

| File | SHA-256 |
| --- | --- |
| `HAT-L_SRx4_ImageNet-pretrain.pth` | `5992bd38522f2b8faf11ea4bd8ee08de92465bb66892166576999afc36d60043` |
| `Real_HAT_GAN_SRx4.pth` | `f5b1e3bbbb05147ca2beefcc715279cb647d7976cbda67d62ea7e6e20d5ffcc7` |

- URL tải lại nằm trong `HatModel.url` ở `src/video_enhance/stages/hat.py`.
- Tile theo cách test của HAT: `tile_size` chia hết cho 16, `tile_pad=32`. Phần đệm chỉ để lấy ngữ cảnh, không trộn trung bình các mảnh.

## EnlightenGAN

- Bài báo và mã train: [VITA-Group/EnlightenGAN](https://github.com/VITA-Group/EnlightenGAN). Giấy phép BSD ở file `License` của repo đó. Copyright 2019 Yifan Jiang and Zhangyang Wang.
- Phần suy luận ONNX tham khảo [arsenyinfo/EnlightenGAN-inference](https://github.com/arsenyinfo/EnlightenGAN-inference), MIT License, copyright 2021 Arseny Kravchenko.
- Trong repo này: `src/video_enhance/stages/enlighten.py` viết lại bước suy luận, không cài package gốc. Ảnh RGB đổi sang BGR, pad bội số của 16, scale về `[0, 1]`, output `[-1, 1]` đưa về `uint8`, rồi đổi lại RGB.
- Weight `weights/enlighten.onnx` tải từ `enlighten_inference/enlighten.onnx` trên nhánh `main` của repo suy luận đó khi file chưa có hoặc nhỏ hơn 30 MB.

## Thư viện không vendored

Các package này cài bằng `pyproject.toml`, không copy mã vào repo: PyTorch, einops, OpenCV, NumPy, PyYAML, Gradio, ONNX Runtime GPU, và các wheel CUDA của NVIDIA. FFmpeg là chương trình ngoài, phải có `libx264` và `libx264rgb`.
