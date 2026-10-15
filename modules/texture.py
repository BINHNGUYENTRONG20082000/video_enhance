"""Texture: tăng/giảm chi tiết bề mặt ở dải kích thước nhỏ."""
import sys
from pathlib import Path
if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detail_filter_utils import process_band

VERSION = '1.1.0'
SUPPORTS_ROI = True

MODULE_ID = 'texture'
PARAMS_SCHEMA = {
    'texture': {'type': 'float', 'min': -100, 'max': 100, 'default': 0, 'ui': 'slider', 'step': 1,
                'label': 'Chi tiết bề mặt',
                'desc': 'Kéo trái làm mềm bề mặt, kéo phải làm rõ tóc, vải và chi tiết nhỏ; 0 giữ nguyên.'},
}


def process(image, params=None, roi=None):
    return process_band(image, params, roi, MODULE_ID, PARAMS_SCHEMA, 'texture',
                        small=0.7, large=2.0, limit=0.08, gain=2.0)


if __name__ == '__main__':
    from src.image_module_utils import demo
    demo(process, 'Chi tiết bề mặt')
