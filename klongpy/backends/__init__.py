"""Backend implementations."""

from . import numpy_backend

try:
    from . import torch_backend  # noqa: F401
except Exception:  # torch may not be available
    torch_backend = None

__all__ = ["numpy_backend", "torch_backend"]
