"""Clarity: tăng/giảm tương phản chi tiết vừa và lớn trên luminance."""
import sys
from pathlib import Path
if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detail_filter_utils import process_band

VERSION = '1.0.0'
SUPPORTS_ROI = True

MODULE_ID = 'clarity'
PARAMS_SCHEMA = {
    'clarity': {'type': 'float', 'min': -100, 'max': 100, 'default': 0, 'ui': 'slider', 'step': 1,
                'label': 'Độ trong',
                'desc': 'Kéo phải làm rõ khối và tương phản cục bộ, kéo trái làm mềm; 0 giữ nguyên.'},
}


def process(image, params=None, roi=None):
    return process_band(image, params, roi, MODULE_ID, PARAMS_SCHEMA, 'clarity',
                        small=2.0, large=12.0, limit=0.12, gain=1.5, midtones=True)


if __name__ == '__main__':
    from src.image_module_utils import demo
    demo(process, 'Độ trong')
