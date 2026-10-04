"""Fail closed before ML imports: no local/CPU/MPS execution path."""
from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path
from typing import Any


class CloudOnlyError(RuntimeError):
    """Training and open-weight inference require an explicitly started Colab GPU."""


def require_colab(*, explicit: bool) -> None:
    """Check runtime identity without importing torch or any model package."""
    if not explicit:
        raise CloudOnlyError("Explicit --start-training/--start-inference consent is required in Colab.")
    try:
        installed = importlib.util.find_spec("google.colab") is not None
    except (ModuleNotFoundError, ValueError):
        installed = False
    colab_marker = any(os.environ.get(key) for key in ("COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION"))
    if platform.system() != "Linux" or not Path("/content").is_dir() or not installed or not colab_marker:
        raise CloudOnlyError("Cloud-only: run this command in a Google Colab GPU runtime. No local fallback exists.")


def require_cuda(*, explicit: bool) -> Any:
    """Only a recognized, explicitly started Colab runtime may import torch."""
    require_colab(explicit=explicit)
    import torch

    if not torch.cuda.is_available():
        raise CloudOnlyError("Select Runtime > Change runtime type > GPU in Colab; CPU/MPS fallback is forbidden.")
    return torch
