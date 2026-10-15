"""Đổi kích thước pixel từ ảnh nguồn cố định, không tích lũy mất chi tiết."""
import argparse
from pathlib import Path
import sys

import cv2 as cv

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_module_utils import check_image, check_params

MODULE_ID = "up_down"
VERSION = "1.1.0"
SUPPORTS_ROI = False
MAX_OUTPUT_PIXELS = 32_000_000
PARAMS_SCHEMA = {
    "width": {"type": "int", "min": 0, "max": 16384, "default": 0, "ui": "number", "step": 1,
              "label": "Chiều rộng (px)", "desc": "Chiều rộng ảnh đầu ra; 0 giữ chiều rộng nguồn."},
    "height": {"type": "int", "min": 0, "max": 16384, "default": 0, "ui": "number", "step": 1,
               "label": "Chiều cao (px)", "desc": "Chiều cao ảnh đầu ra; 0 giữ chiều cao nguồn."},
    "zoom_percent": {"type": "float", "min": 25, "max": 400, "default": 100,
                     "ui": "slider", "step": 1, "label": "Kích thước ảnh (%)",
                     "desc": "50% giảm một nửa chiều rộng/cao; 200% tăng gấp đôi; giữ tỉ lệ ảnh."},
}


def process(image, params=None, roi=None):
    """image phải là ảnh nguồn, không truyền lại ảnh đã resize từ lần trước."""
    check_image(MODULE_ID, image)
    p = check_params(MODULE_ID, PARAMS_SCHEMA, params)
    if roi is not None:
        raise ValueError("up_down: không hỗ trợ ROI; hãy cắt ảnh trước khi đổi kích thước")
    h, w = image.shape[:2]
    scale = p["zoom_percent"] / 100.0
    if p["width"] or p["height"]:
        target_w, target_h = p["width"] or w, p["height"] or h
    else:  # Giữ tương thích API phần trăm cũ; giao diện dùng width/height.
        target_w, target_h = max(1, round(w * scale)), max(1, round(h * scale))
    if (target_w, target_h) == (w, h):
        result, interpolation = image.copy(), "none"
    else:
        if target_w * target_h > MAX_OUTPUT_PIXELS:
            raise ValueError("up_down: ảnh kết quả vượt giới hạn 32 triệu pixel; hãy giảm mức phóng")
        down = target_w <= w and target_h <= h
        result = cv.resize(image, (target_w, target_h),
                           interpolation=cv.INTER_AREA if down else cv.INTER_LANCZOS4)
        interpolation = "area" if down else "lanczos4"
    return result, {"effective_params": p, "source_size": {"w": w, "h": h},
                    "output_size": {"w": target_w, "h": target_h},
                    "interpolation": interpolation}


class ZoomSession:
    """Giữ bản sao ảnh gốc cho một phiên zoom bằng chuột hoặc bàn phím."""
    def __init__(self, image):
        check_image(MODULE_ID, image)
        self._source = image.copy()
        self._source.flags.writeable = False
        self.zoom_percent = 100.0

    def render(self, zoom_percent):
        result, metadata = process(self._source, {"zoom_percent": zoom_percent})
        self.zoom_percent = metadata["effective_params"]["zoom_percent"]
        return result, metadata



class ResizeSession:
    """Mỗi kích thước đầu ra được tạo từ bản sao nguồn cố định."""
    def __init__(self, image):
        check_image(MODULE_ID, image)
        self._source = image.copy()
        self._source.flags.writeable = False
        self.height, self.width = image.shape[:2]

    def render(self, width, height):
        result, metadata = process(self._source, {'width': width, 'height': height})
        self.width = metadata['output_size']['w']
        self.height = metadata['output_size']['h']
        return result, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Đổi chiều rộng và chiều cao của ảnh")
    parser.add_argument("image_path", nargs="?", default=r"D:\Model_opencv\image\plate_1.jpg")
    parser.add_argument("--width", type=int, default=0, help="Chiều rộng đầu ra")
    parser.add_argument("--height", type=int, default=0, help="Chiều cao đầu ra")
    parser.add_argument("--output", help="Lưu trực tiếp ra file, không mở cửa sổ")
    args = parser.parse_args(argv)
    source = cv.imread(args.image_path)
    if source is None:
        print(f"Không đọc được ảnh: {args.image_path}")
        return 1
    session = ResizeSession(source)
    try:
        initial, _ = session.render(args.width, args.height)
        if args.output:
            if not cv.imwrite(args.output, initial):
                raise ValueError(f"Không lưu được ảnh: {args.output}")
            print(f"Đã lưu {session.width} × {session.height}: {args.output}")
            return 0
    except (ValueError, cv.error) as exc:
        print(exc)
        return 1
    window = "Doi kich thuoc anh"
    width_bar, height_bar = "Chieu rong", "Chieu cao"
    cv.namedWindow(window, cv.WINDOW_NORMAL)
    ready = False

    def render_controls():
        width = cv.getTrackbarPos(width_bar, window)
        height = cv.getTrackbarPos(height_bar, window)
        return session.render(width, height)[0]

    def update(_value):
        if ready:
            try:
                cv.imshow(window, render_controls())
            except (ValueError, cv.error) as exc:
                print(f"Không thể đổi kích thước: {exc}")

    try:
        cv.imshow(window, initial)
        cv.createTrackbar(width_bar, window, session.width, 16384, update)
        cv.createTrackbar(height_bar, window, session.height, 16384, update)
        cv.setTrackbarMin(width_bar, window, 1)
        cv.setTrackbarMin(height_bar, window, 1)
        ready = True
        print("Kéo chiều rộng/cao; r: kích thước nguồn; s: lưu PNG; ESC: đóng.")
        while True:
            key = cv.waitKey(30)
            if key == 27 or cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
                break
            if key < 0:
                continue
            if key & 0xFF == ord('r'):
                cv.setTrackbarPos(width_bar, window, source.shape[1])
                cv.setTrackbarPos(height_bar, window, source.shape[0])
            elif key & 0xFF == ord('s'):
                try:
                    output = render_controls()
                    path = Path(args.image_path).with_name(Path(args.image_path).stem + '_resized.png')
                    if not cv.imwrite(str(path), output):
                        raise ValueError(f"Không lưu được ảnh: {path}")
                    print(f"Đã lưu {output.shape[1]} × {output.shape[0]}: {path}")
                except (ValueError, cv.error) as exc:
                    print(exc)
    finally:
        cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
