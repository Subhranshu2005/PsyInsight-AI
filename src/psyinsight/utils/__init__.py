"""
PsyInsight AI — Utils
=====================

Shared, cross-cutting utilities used by every module in the package:

    * Logging      — ``get_logger`` gives every module a consistently
                      formatted logger without re-configuring handlers.
    * Reproducibility — ``set_seed`` seeds python / numpy / (optionally)
                      torch in one call.
    * Timing       — ``Timer`` context manager + ``timeit`` decorator.
    * Config       — ``PsyConfig`` tiny dataclass-like settings object
                      that can be created from a dict / json file and
                      used as ``config.get("key", default)``.
    * DataFrame helpers — ``infer_column_types``, ``ensure_dataframe``,
                      ``safe_sample``, ``memory_usage_report``.
    * Filesystem helpers — ``ensure_dir``, ``unique_path``.

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "get_logger",
    "set_seed",
    "Timer",
    "timeit",
    "PsyConfig",
    "infer_column_types",
    "ensure_dataframe",
    "safe_sample",
    "memory_usage_report",
    "ensure_dir",
    "unique_path",
    "PsyInsightError",
    "DataValidationError",
    "ModelNotFittedError",
    "dataframe_fingerprint",
    "environment_fingerprint",
    "reproducibility_snapshot",
]


# =============================================================================
# Exceptions
# =============================================================================


class PsyInsightError(Exception):
    """Base exception for all PsyInsight AI errors."""


class DataValidationError(PsyInsightError):
    """Raised when input data fails a validation / sanity check."""


class ModelNotFittedError(PsyInsightError):
    """Raised when a method requiring a fitted model is called too early."""


# =============================================================================
# Logging
# =============================================================================

_CONFIGURED_LOGGERS: Dict[str, logging.Logger] = {}
_DEFAULT_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s :: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "psyinsight", level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a single, consistently formatted handler.

    Calling this repeatedly with the same ``name`` will *not* attach duplicate
    handlers (a common gotcha with the stdlib ``logging`` module).
    """
    if name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
        logger.addHandler(handler)
    logger.propagate = False

    _CONFIGURED_LOGGERS[name] = logger
    return logger


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int = 42) -> int:
    """Seed python's ``random``, ``numpy`` and (if installed) ``torch``.

    Returns the seed used, so callers can log / store it for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:  # pragma: no cover - optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed


# Packages worth recording a version for, if installed. Anything not
# installed (e.g. torch/transformers on a CPU-only, NLP-free install) is
# silently omitted rather than raising -- reproducibility metadata should
# never be able to crash the calling analysis.
_TRACKED_PACKAGES = (
    "psyinsight-ai",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "torch",
    "transformers",
    "joblib",
    "streamlit",
)


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 hash of a DataFrame's data, column names, and
    dtypes.

    Two DataFrames with the same fingerprint are guaranteed to contain the
    same data (same values, same column order, same dtypes); this is the
    "dataset version" half of a reproducibility record -- run the same
    analysis on a differently-hashed dataset and you should expect
    different results, so recording this hash alongside a result lets
    someone verify they're looking at a reproduction of the *same* run,
    not just the same code on different data.

    Row order matters (a shuffled dataset hashes differently), which is
    intentional: order can affect e.g. train/test splits done without an
    explicit sort.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    hasher = hashlib.sha256()
    hasher.update(",".join(str(c) for c in df.columns).encode("utf-8"))
    hasher.update(",".join(str(t) for t in df.dtypes).encode("utf-8"))
    # pandas.util.hash_pandas_object handles mixed dtypes / NaNs safely and
    # is itself deterministic given the same data.
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy()
    hasher.update(row_hashes.tobytes())
    return hasher.hexdigest()


def environment_fingerprint() -> Dict[str, Any]:
    """Snapshot of the runtime environment: Python version, OS/platform,
    and the installed version of every package in ``_TRACKED_PACKAGES``
    that is actually present. Used so a research report can say exactly
    what produced it, which is a prerequisite for someone else being able
    to reproduce the result.
    """
    versions: Dict[str, str] = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue  # not installed -- omit rather than fail

    if "psyinsight-ai" not in versions:
        # Fall back to the in-package __version__ if PsyInsight is being
        # run from source without being pip-installed (e.g. `python -m`
        # from a checkout, or bundled directly into another project).
        try:
            from psyinsight import __version__ as _psyinsight_version

            versions["psyinsight-ai"] = _psyinsight_version
        except Exception:  # noqa: BLE001
            pass

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "package_versions": versions,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def reproducibility_snapshot(
    dataframe: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One-call bundle of everything needed to reproduce an analysis:
    dataset hash, random seed, environment fingerprint, and any
    caller-supplied extra metadata (e.g. model hyperparameters).

    Intended to be dropped straight into a research report or saved next
    to a model checkpoint, e.g.::

        snapshot = reproducibility_snapshot(df, seed=42, extra={"model": "random_forest"})
        report.add_section("Reproducibility", snapshot)
    """
    snapshot: Dict[str, Any] = {"environment": environment_fingerprint()}

    if dataframe is not None:
        snapshot["dataset_hash"] = dataframe_fingerprint(dataframe)
        snapshot["dataset_shape"] = {"rows": dataframe.shape[0], "columns": dataframe.shape[1]}

    snapshot["random_seed"] = seed

    if extra:
        snapshot["extra"] = dict(extra)

    return snapshot


# =============================================================================
# Timing
# =============================================================================


@contextmanager
def Timer(label: str = "block", logger: Optional[logging.Logger] = None):
    """Context manager that logs / prints the wall-clock time of a code block.

    Example
    -------
    >>> with Timer("training"):
    ...     model.fit(X, y)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    message = f"[Timer] {label} took {elapsed:.4f}s"
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def timeit(func):
    """Decorator version of :func:`Timer` for instrumenting functions/methods."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timeit] {func.__qualname__} took {elapsed:.4f}s")
        return result

    return wrapper


# =============================================================================
# Config
# =============================================================================


@dataclass
class PsyConfig:
    """Lightweight, dict-backed configuration object.

    Supports attribute access, ``.get(key, default)`` lookup, and loading
    from / saving to JSON so experiments stay reproducible.
    """

    values: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> "PsyConfig":
        self.values[key] = value
        return self

    def __getattr__(self, item: str) -> Any:
        try:
            return self.values[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __getitem__(self, item: str) -> Any:
        return self.values[item]

    def __contains__(self, item: str) -> bool:
        return item in self.values

    def update(self, other: Dict[str, Any]) -> "PsyConfig":
        self.values.update(other)
        return self

    @classmethod
    def from_json(cls, path: str) -> "PsyConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(values=json.load(fh))

    def to_json(self, path: str) -> None:
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.values, fh, indent=2, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.values)


# =============================================================================
# DataFrame helpers
# =============================================================================


def ensure_dataframe(data: Any) -> pd.DataFrame:
    """Coerce array-likes / dicts / Series into a DataFrame, raising a clear
    :class:`DataValidationError` if that isn't possible."""
    if isinstance(data, pd.DataFrame):
        return data
    try:
        return pd.DataFrame(data)
    except Exception as exc:  # noqa: BLE001
        raise DataValidationError(
            f"Could not coerce object of type {type(data)!r} into a DataFrame"
        ) from exc


def infer_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify every column of ``df`` into numeric / categorical / datetime /
    boolean / text buckets — useful for auto-EDA and auto-preprocessing.
    """
    df = ensure_dataframe(df)
    buckets: Dict[str, List[str]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "text": [],
    }

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            buckets["boolean"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            buckets["datetime"].append(col)
        elif pd.api.types.is_numeric_dtype(series):
            buckets["numeric"].append(col)
        elif isinstance(series.dtype, pd.CategoricalDtype) or series.nunique(dropna=True) <= max(
            10, int(0.05 * len(series))
        ):
            buckets["categorical"].append(col)
        else:
            non_null = series.dropna().astype(str)
            avg_len = non_null.str.len().mean() if len(non_null) else 0
            if avg_len and avg_len > 30:
                buckets["text"].append(col)
            else:
                buckets["categorical"].append(col)

    return buckets


def safe_sample(df: pd.DataFrame, n: int = 5, random_state: int = 42) -> pd.DataFrame:
    """``df.sample`` that never raises when ``n`` exceeds ``len(df)``."""
    df = ensure_dataframe(df)
    return df.sample(n=min(n, len(df)), random_state=random_state) if len(df) else df


def memory_usage_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a small dict describing the memory footprint of ``df``."""
    df = ensure_dataframe(df)
    usage = df.memory_usage(deep=True)
    return {
        "total_mb": round(usage.sum() / (1024 ** 2), 4),
        "per_column_kb": (usage / 1024).round(3).to_dict(),
        "n_rows": len(df),
        "n_columns": df.shape[1],
    }


# =============================================================================
# Filesystem helpers
# =============================================================================


def ensure_dir(path: str) -> str:
    """``os.makedirs(path, exist_ok=True)`` that also returns the path."""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def unique_path(path: str) -> str:
    """Return ``path`` unchanged if it doesn't exist, otherwise append
    ``_1``, ``_2``, ... before the extension until a free path is found."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    counter = 1
    candidate = f"{root}_{counter}{ext}"
    while os.path.exists(candidate):
        counter += 1
        candidate = f"{root}_{counter}{ext}"
    return candidate
