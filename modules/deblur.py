"""Khử nhòe mất nét bằng PSF hình đĩa và bộ lọc Wiener."""
import argparse
from pathlib import Path
import sys

import cv2 as cv
import numpy as np

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_module_utils import check_image, check_params

MODULE_ID = 'deblur'
VERSION = '1.0.0'
SUPPORTS_ROI = False
RESTORATION_SNR = 1000.0
PARAMS_SCHEMA = {
    'blur_radius': {'type': 'int', 'min': 1, 'max': 100, 'default': 5, 'ui': 'slider', 'step': 1,
                    'label': 'Độ nhòe cần xử lý',
                    'desc': 'Chọn bán kính nhòe trên ảnh nguồn (pixel); thử từ mức nhỏ và điều chỉnh để rõ nhất.'},
    'deblur_strength': {'type': 'float', 'min': 0, 'max': 100, 'default': 0, 'ui': 'slider', 'step': 1,
                        'label': 'Mức khử mờ',
                        'desc': '0 giữ nguyên ảnh; kéo phải tăng tác động, mức cao có thể làm nổi nhiễu hoặc viền.'},
}

DEFAULT_IMAGE = Path(__file__).resolve().parent.parent / 'image' / 'original (1).jpg'


def _matrix(value, name):
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.size == 0:
        raise ValueError(f'{name}: cần ma trận 2D không rỗng')
    if value.dtype.kind not in 'uif' or not np.isfinite(value).all():
        raise ValueError(f'{name}: cần dữ liệu số hữu hạn')
    return np.ascontiguousarray(value, dtype=np.float32)


def _radius(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('R phải là số nguyên >= 0')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('R phải là số nguyên >= 0') from exc
    if not np.isfinite(number) or not number.is_integer() or not 0 <= number <= 2**31 - 1:
        raise ValueError('R phải là số nguyên >= 0 và nằm trong khoảng int32')
    return int(number)


def _positive(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} phải là số hữu hạn > 0') from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f'{name} phải là số hữu hạn > 0')
    return number


def calcPSF(filterSize, R):
    """filterSize = (width, height), giống cv::Size trong bản C++."""
    try:
        width, height = filterSize
        if any(isinstance(v, (bool, np.bool_)) or not np.isfinite(v) or int(v) != v or v < 1
               for v in (width, height)):
            raise ValueError()
        width, height = int(width), int(height)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('filterSize cần (chiều rộng, chiều cao) nguyên dương') from exc
    radius = _radius(R)
    h = np.zeros((height, width), dtype=np.float32)
    cv.circle(h, (width // 2, height // 2), radius, 1, -1, cv.LINE_8)
    return h / float(cv.sumElems(h)[0])


def fftshift(inputImg):
    """Đưa tâm PSF về gốc; đúng cả kích thước chẵn và lẻ."""
    source = _matrix(inputImg, 'PSF')
    return np.fft.ifftshift(source).copy()


def calcWnrFilter(input_h_PSF, nsr):
    """G = conj(H) / (|H|² + NSR), trả hai kênh thực/ảo."""
    noise = _positive(nsr, 'NSR')
    if noise > np.finfo(np.float32).max:
        raise ValueError('NSR quá lớn cho float32')
    spectrum = cv.dft(fftshift(input_h_PSF), flags=cv.DFT_COMPLEX_OUTPUT)
    real, imag = cv.split(spectrum)
    denominator = real * real + imag * imag + noise
    return cv.merge((cv.divide(real, denominator), cv.divide(-imag, denominator)))


def filter2DFreq(inputImg, H):
    """Áp bộ lọc lên phổ của ảnh; không dùng hình bộ lọc làm đầu ra."""
    image = _matrix(inputImg, 'Ảnh')
    if not isinstance(H, np.ndarray) or H.shape not in (image.shape, (*image.shape, 2)):
        raise ValueError('Bộ lọc cần cùng kích thước với ảnh, có một hoặc hai kênh')
    if H.dtype.kind not in 'uif' or not np.isfinite(H).all():
        raise ValueError('Bộ lọc cần dữ liệu số hữu hạn')
    transfer = np.ascontiguousarray(H, dtype=np.float32)
    if transfer.ndim == 2:
        transfer = cv.merge((transfer, np.zeros(image.shape, np.float32)))
    spectrum = cv.dft(image, flags=cv.DFT_COMPLEX_OUTPUT)
    filtered = cv.mulSpectrums(spectrum, transfer, 0)
    return cv.idft(filtered, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)


def deblur_image(image, R=53, SNR=5200):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError('Cần ảnh uint8 không rỗng')
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError('Cần ảnh xám HxW hoặc ảnh BGR HxWx3')
    height, width = gray.shape
    if min(height, width) < 2:
        raise ValueError('Ảnh cần ít nhất 2 × 2')
    radius, snr = _radius(R), _positive(SNR, 'SNR')
    # FFT trực tiếp trên ảnh nguồn, giữ đúng chiều rộng/cao kể cả số lẻ.
    psf = calcPSF((width, height), radius)
    transfer = calcWnrFilter(psf, 1.0 / snr)
    restored = filter2DFreq(gray, transfer)
    if not np.isfinite(restored).all():
        raise ValueError('Kết quả không hữu hạn; hãy giảm SNR')
    # Giữ thang sáng gốc; min-max normalize có thể biến ảnh đồng màu thành đen.
    return np.clip(np.rint(restored), 0, 255).astype(np.uint8)


def process(image, params=None, roi=None):
    """Khử mờ trên từng kênh màu ở đúng kích thước nguồn, dùng chung PSF."""
    check_image(MODULE_ID, image)
    p = check_params(MODULE_ID, PARAMS_SCHEMA, params)
    if roi is not None:
        raise ValueError('deblur: không hỗ trợ ROI; hãy cắt ảnh trong trình chỉnh sửa')
    metadata = {'effective_params': p, 'snr': RESTORATION_SNR,
                'method': 'wiener_disk', 'source_size': {'w': image.shape[1], 'h': image.shape[0]},
                'processed_channels': 'BGR' if image.ndim == 3 else 'GRAY'}
    if p['deblur_strength'] == 0:
        return image.copy(), metadata
    height, width = image.shape[:2]
    if min(height, width) < 2:
        raise ValueError('deblur: ảnh cần ít nhất 2 × 2')
    transfer = calcWnrFilter(calcPSF((width, height), p['blur_radius']), 1 / RESTORATION_SNR)
    amount = p['deblur_strength'] / 100.0
    output = np.empty_like(image)
    for channel in range(3 if image.ndim == 3 else 1):
        source = image[:, :, channel] if image.ndim == 3 else image
        restored = filter2DFreq(source, transfer)
        blended = source.astype(np.float32) + amount * (restored - source)
        if not np.isfinite(blended).all():
            raise ValueError('deblur: kết quả không hữu hạn')
        result = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        if image.ndim == 3:
            output[:, :, channel] = result
        else:
            output[:] = result
    return output, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description='Khử nhòe mất nét bằng Wiener filter')
    parser.add_argument('--image', default=str(DEFAULT_IMAGE), help='Đường dẫn ảnh đầu vào')
    parser.add_argument('--R', type=int, default=53, help='Bán kính nhòe hình đĩa, >= 0')
    parser.add_argument('--SNR', type=float, default=5200, help='Tỉ lệ tín hiệu/nhiễu, > 0')
    parser.add_argument('--output', default='result.png', help='Đường dẫn ảnh kết quả')
    parser.add_argument('--show', action='store_true', help='Mở cửa sổ xem ảnh')
    args = parser.parse_args(argv)
    try:
        _radius(args.R)
        _positive(args.SNR, 'SNR')
        image = cv.imread(args.image, cv.IMREAD_GRAYSCALE | cv.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError(f'Không đọc được ảnh: {args.image}')
        output = deblur_image(image, args.R, args.SNR)
        if not cv.imwrite(args.output, output):
            raise ValueError(f'Không lưu được ảnh: {args.output}')
        print(f'Đã lưu ảnh {output.shape[1]} × {output.shape[0]}: {args.output}')
        if args.show:
            try:
                cv.imshow('Original', image)
                cv.imshow('Deblur', output)
                cv.waitKey(0)
            finally:
                cv.destroyAllWindows()
    except (ValueError, cv.error) as exc:
        print(f'Lỗi: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
