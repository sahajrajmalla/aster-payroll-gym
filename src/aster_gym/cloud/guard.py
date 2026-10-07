"""Fail closed before ML imports: no local/CPU/MPS execution path."""
from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path
from typing import Any


class CloudOnlyError(RuntimeError):
    """Training and open-weight inference require an explicitly started Colab GPU."""


_COLAB_SYSTEM_ROOTS = (Path("/usr/local/lib"), Path("/usr/lib"))


def _host_colab_package_exists() -> bool:
    """Recognize Colab's host package without importing it into the pinned virtualenv."""
    patterns = ("python3.*/dist-packages/google/colab/__init__.py",
                "python3.*/site-packages/google/colab/__init__.py",
                "python3/dist-packages/google/colab/__init__.py")
    for root in _COLAB_SYSTEM_ROOTS:
        for pattern in patterns:
            try:
                if any(path.is_file() for path in root.glob(pattern)):
                    return True
            except OSError:
                continue
    return False


def require_colab(*, explicit: bool) -> None:
    """Check runtime identity without importing torch or any model package."""
    if not explicit:
        raise CloudOnlyError("Explicit start action is required in hosted Colab.")
    colab_marker = any(os.environ.get(key) for key in ("COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION"))
    if platform.system() != "Linux" or not Path("/content").is_dir() or not colab_marker:
        raise CloudOnlyError("Cloud-only: run this command in a Google Colab GPU runtime. No local fallback exists.")
    try:
        installed = importlib.util.find_spec("google.colab") is not None
    except (ModuleNotFoundError, ValueError):
        installed = False
    # uv's isolated Python may not import the notebook host's google.colab package.
    # Inspect its bounded system location; never add host libraries to sys.path.
    if not installed and not _host_colab_package_exists():
        raise CloudOnlyError("Cloud-only: run this command in a Google Colab GPU runtime. No local fallback exists.")


def require_cuda(*, explicit: bool) -> Any:
    """Only a recognized, explicitly started Colab runtime may import torch."""
    require_colab(explicit=explicit)
    import torch

    if not torch.cuda.is_available():
        raise CloudOnlyError("Select Runtime > Change runtime type > GPU in Colab; CPU/MPS fallback is forbidden.")
    return torch
