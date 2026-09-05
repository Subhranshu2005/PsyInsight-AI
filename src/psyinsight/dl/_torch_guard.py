"""Internal helper: import torch lazily and raise a friendly error if it's
not installed, instead of crashing the whole ``psyinsight`` package on
import (torch is a heavy, optional dependency)."""

from __future__ import annotations

_TORCH_INSTALL_HINT = (
    "PyTorch is required for psyinsight.dl but is not installed. "
    "Install it with:  pip install torch  "
    "(see https://pytorch.org/get-started/locally/ for platform-specific instructions)."
)


def require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_TORCH_INSTALL_HINT) from exc
    return torch, nn


TORCH_AVAILABLE = True
try:
    import torch as _torch  # noqa: F401
except ImportError:
    TORCH_AVAILABLE = False
