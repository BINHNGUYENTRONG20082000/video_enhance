"""Entrypoint API; import client không khởi tạo backend xử lý ảnh."""
from importlib import import_module

__all__ = ["app", "health", "router"]


def __getattr__(name):
    if name in __all__:
        return getattr(import_module(".api", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
