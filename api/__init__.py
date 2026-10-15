"""Entrypoint công khai: uvicorn api:app và from api import router."""
from .api import app, health, router

__all__ = ["app", "health", "router"]
