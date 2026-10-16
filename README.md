# Bộ lọc xử lý ảnh

Tài liệu bàn giao cho dev tích hợp bộ lọc vào backend Python. Backend nhận ảnh và tham số từ frontend, gọi hàm xử lý trực tiếp rồi trả ảnh kết quả. Không cần gọi thêm HTTP API giữa backend và bộ lọc.

## Thành phần cần bàn giao

- `src/processing.py`: kiểm tra tham số và điều phối các bộ lọc.
- `src/image_module_utils.py`, `src/detail_filter_utils.py`: các hàm hỗ trợ.
- Thư mục `modules/`: toàn bộ file bộ lọc, bao gồm `__init__.py`.
- `requirements.txt` và tài liệu này.

Giữ nguyên cấu trúc thư mục để các import `src.*` và `modules.*` hoạt động. Không cần bàn giao Gradio, API hoặc API client nếu dev chỉ gọi engine trực tiếp.

## Môi trường

Python 3.12 trở lên; đã kiểm chứng trên Windows với Python 3.14.7. Engine sử dụng NumPy, opencv-contrib-python và Matplotlib; không cần model AI, PyTorch hay CUDA.

Có thể cài thư viện bằng `python -m pip install -r requirements.txt`. File này hiện bao gồm cả thư viện HTTP API; phần gọi engine trực tiếp không phụ thuộc FastAPI, Uvicorn hoặc python-multipart.

Chỉ cài một gói OpenCV trong môi trường. Dùng opencv-contrib-python để có Guided Filter; không cài chung với opencv-python hoặc các biến thể OpenCV khác.

## Hàm xử lý ảnh

Hàm tích hợp chính là `src.processing.process_image(image, settings=None, *, mods=None)`.

**Đầu vào:**

- `image`: ảnh NumPy kiểu uint8, ba kênh BGR, kích thước height × width × 3. Đây là mảng ảnh đã giải mã, không phải đường dẫn hay bytes của file upload.
- `settings`: dictionary chứa giá trị chỉnh sửa người dùng chọn. Nếu không truyền hoặc truyền dictionary rỗng, engine dùng mặc định.
- `mods`: tùy chọn dictionary module đã nạp bằng `src.processing.discover()`. Backend có thể nạp một lần khi khởi động và dùng lại; không bắt buộc truyền.

**Đầu ra:** một tuple gồm ba giá trị theo thứ tự:

1. `result`: ảnh kết quả BGR uint8.
2. `histogram`: ảnh biểu đồ BGR uint8, hoặc None nếu không bật biểu đồ.
3. `metadata`: thông tin kích thước nguồn, kích thước đầu ra và các bước đã áp dụng.

Hàm không đọc file, lưu ảnh hay tạo HTTP response. Backend tự giải mã ảnh upload, gọi hàm và mã hóa kết quả theo định dạng cần trả cho frontend.

Ảnh từ Pillow hoặc frontend thường là RGB; cần chuyển sang BGR trước khi gọi. Backend cũng chịu trách nhiệm chuẩn hóa hướng EXIF, ảnh xám, alpha hoặc dữ liệu 16-bit về định dạng đầu vào yêu cầu.

## Quy định tham số cho frontend

Gọi `src.processing.control_schema()` để lấy danh sách nhóm tính năng và các control. Mỗi control có tên tham số, kiểu dữ liệu, nhãn, mặc định và giới hạn hoặc danh sách lựa chọn tương ứng.

Backend trả cấu hình này qua endpoint hiện có của hệ thống. Frontend dùng nó để tạo thanh kéo, ô bật/tắt và lưu giá trị người dùng chọn vào `settings`.

Tên tham số theo nhóm:

- Ánh sáng: `brightness`, `highlights`, `shadows`, `contrast`, `whites`, `blacks`; CLAHE dùng `clahe_enabled` và `clip_limit`.
- Màu sắc: `temperature`, `tint`, `vibrance`, `saturation`.
- Khử nhiễu: `median_enabled` và `median_kernel_size`; `bilateral_enabled` và `bilateral_sigma_color`; `nlm_enabled` và `nlm_h`; `guided_enabled` và `guided_smoothness`.
- Chi tiết và hiệu ứng: `deblur_strength`, `sharpness`, `texture`, `clarity`, `dehaze`, `vignette`.
- Kích thước ảnh: `width`, `height`.
- Biểu đồ: `histogram_enabled`, `histogram_mode`, `histogram_normalize`.

Lấy giới hạn và mặc định từ schema thay vì tự ghi cố định ở frontend. Không truyền các tên kỹ thuật nội bộ như gamma hoặc eps. HSL và Threshold không nằm trong schema/pipeline công khai.

## Luồng sử dụng

1. Backend lấy schema để frontend dựng giao diện.
2. Frontend gửi ảnh nguồn cùng toàn bộ trạng thái chỉnh sửa hiện tại.
3. Backend giải mã ảnh về BGR uint8 và chuyển settings thành dictionary.
4. Backend gọi process_image, nhận ảnh và metadata.
5. Backend mã hóa ảnh kết quả và trả cho frontend hiển thị.

Mỗi lượt chỉnh phải xử lý lại từ ảnh nguồn. Không dùng ảnh kết quả lần trước làm nguồn vì hiệu ứng sẽ cộng dồn và resize nhiều lần làm mất nét.

## Kích thước và lỗi

Width và height là kích thước đầu ra theo pixel. Giá trị 0 giữ chiều tương ứng của ảnh nguồn; hai chiều độc lập, không tự giữ tỷ lệ.

Ảnh nguồn và đầu ra bị giới hạn ở 32 triệu pixel; mỗi chiều resize tối đa 16384. Các tham số cắt/xoay không còn được hỗ trợ và sẽ bị từ chối khi truyền vào settings.

Process_image tự kiểm tra tham số; backend bắt ValueError và chuyển thành thông báo phù hợp. Có thể gọi riêng `src.processing.validate_settings()` nếu muốn kiểm tra trước. Backend tự quy định giới hạn upload và số tác vụ đồng thời; giới hạn của HTTP API mẫu không tự áp dụng khi gọi hàm trực tiếp.

## HTTP API và Gradio — tùy chọn

Phần này phục vụ trường hợp tách dịch vụ xử lý ảnh hoặc thử kết nối HTTP:

- Chạy API bằng `python -m uvicorn api:app --host 127.0.0.1 --port 8000`; tài liệu tương tác tại `http://127.0.0.1:8000/docs`.
- Các endpoint: GET /api/v1/health, GET /api/v1/filters và POST /api/v1/process. POST nhận file image cùng settings dạng chuỗi JSON trong multipart/form-data.
- Client Python nằm ở `api/api_client.py`; ImageApiClient nhận/trả ảnh RGB, khác với engine trực tiếp dùng BGR.
- Gradio hiện gọi API qua HTTP. Cài `requirements-demo.txt`, chạy API ở terminal thứ nhất và `python main.py` ở terminal thứ hai.
- Mặc định Gradio gọi http://127.0.0.1:8000; biến môi trường IMAGE_API_URL đổi địa chỉ gốc của API.

Backend Python tích hợp trực tiếp chỉ cần các hàm và thành phần đã mô tả ở trên.
