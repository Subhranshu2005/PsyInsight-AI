"""Internal helper for optionally-installed Hugging Face `transformers`."""

from __future__ import annotations

_HINT = (
    "The 'transformers' (and 'torch') packages are required for "
    "psyinsight.transformers but are not installed. Install them with:  "
    "pip install transformers torch"
)

HF_AVAILABLE = True
try:
    import transformers as _transformers  # noqa: F401
except ImportError:
    HF_AVAILABLE = False


def require_hf():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_HINT) from exc
    return transformers
