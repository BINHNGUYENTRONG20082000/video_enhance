
from collections.abc import Mapping
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_module_utils import check_image, roi_mask

MODULE_ID = "histogram"
VERSION = "1.0.0"
SUPPORTS_ROI = False
PARAMS_SCHEMA = {
    "mode": {"type": "enum", "values": ["gray", "rgb"], "default": "gray",
             "ui": "dropdown"},
    "normalize": {"type": "bool", "default": False, "ui": "checkbox"},
}


def _fill_and_check(params):
    params = {} if params is None else params
    if not isinstance(params, Mapping):
        raise ValueError(f"{MODULE_ID}: params phải là dict")
    unknown = set(params) - set(PARAMS_SCHEMA)
    if unknown:
        raise ValueError(f"{MODULE_ID}: tham số không hợp lệ: {list(unknown)}")
    p = {name: params.get(name, spec["default"]) for name, spec in PARAMS_SCHEMA.items()}
    if p["mode"] not in ("gray", "rgb"):
        raise ValueError("mode phải là 'gray' hoặc 'rgb'")
    if not isinstance(p["normalize"], (bool, np.bool_)):
        raise ValueError("normalize phải là bool")
    p["normalize"] = bool(p["normalize"])
    return p


def calculate_histogram(image, mode="gray", normalize=False, roi=None):
    """Trả 256 bins và số pixel hoặc xác suất của mỗi kênh; chỉ nhận uint8."""
    check_image(MODULE_ID, image)
    p = _fill_and_check({"mode": mode, "normalize": normalize})
    mask = roi_mask(MODULE_ID, roi, image.shape)
    if p["mode"] == "rgb" and image.ndim != 3:
        raise ValueError("RGB histogram cần ảnh màu BGR 3 kênh")
    if p["mode"] == "gray":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        channels = {"histogram": gray}
    else:
        channels = {"red": image[:, :, 2], "green": image[:, :, 1], "blue": image[:, :, 0]}
    result = {"bins": list(range(256))}
    for name, channel in channels.items():
        pixels = channel.ravel() if mask is None else channel[mask]
        counts = np.bincount(pixels, minlength=256)
        hist = counts / pixels.size if p["normalize"] else counts
        result[name] = hist.tolist()
    return result


def plot_histogram(data, normalize=False):
    """Vẽ bằng Agg, không mở GUI; trả ảnh BGR uint8 1000x450."""
    figure = Figure(figsize=(10, 4.5), dpi=100, layout="constrained")
    canvas = FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    series = [("histogram", "#333333", "Gray")] if "histogram" in data else [
        ("red", "#dc2626", "Red"), ("green", "#16a34a", "Green"), ("blue", "#2563eb", "Blue")]
    peak = 0.0
    for name, color, label in series:
        values = np.asarray(data[name], dtype=np.float64)
        peak = max(peak, float(values.max()))
        # Đường đứng giúp nhìn thấy cả histogram ảnh đồng màu (chỉ có một bin).
        ax.vlines(data["bins"], 0, values, colors=color, alpha=0.65, linewidth=1)
        ax.step(data["bins"], values, where="mid", color=color, linewidth=1.4, label=label)
    ax.set_xlim(-1, 256)
    ax.set_ylim(0, max(peak * 1.1, 1e-12))
    ax.set_xticks([0, 32, 64, 96, 128, 160, 192, 224, 255])
    ax.set_xlabel("Pixel intensity (0-255)")
    ax.set_ylabel("Probability" if normalize else "Pixel count")
    ax.set_title("Grayscale histogram" if "histogram" in data else "RGB histogram")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def process(image, params=None, roi=None):
    p = _fill_and_check(params)
    data = calculate_histogram(image, p["mode"], p["normalize"], roi)
    mask = roi_mask(MODULE_ID, roi, image.shape)
    chart = plot_histogram(data, p["normalize"])
    return chart, {
        "mode": p["mode"], "normalized": p["normalize"],
        "pixel_count": int(image.shape[0] * image.shape[1] if mask is None else mask.sum()),
        "roi_applied": mask is not None, "color_order": "BGR",
        "histogram": data,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Histogram")
    parser.add_argument("image_path", nargs="?",
                        default=r"D:\Model_opencv\image\image29-261x300.jpg")
    parser.add_argument("--mode", choices=["gray", "rgb"], default="rgb", help="Chọn chế độ: gray hoặc rgb")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(args.image_path)
    chart, _ = process(image, {"mode": args.mode, "normalize": args.normalize})
    cv2.imshow("Histogram", chart)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
