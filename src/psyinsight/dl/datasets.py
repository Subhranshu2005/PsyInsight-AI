"""
PsyInsight AI — Deep Learning Datasets
========================================

Thin ``torch.utils.data.Dataset`` wrappers around pandas DataFrames /
numpy arrays, for tabular and sequence (text-token) data.

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ._torch_guard import require_torch

__all__ = ["TabularDataset", "SequenceDataset"]


def _lazy_bases():
    torch, nn = require_torch()
    from torch.utils.data import Dataset

    return torch, Dataset


class TabularDataset:
    """A ``Dataset`` of ``(features, target)`` tensors built from a
    DataFrame/array pair — the standard input for the MLP models in
    :mod:`psyinsight.dl.models`.

    Instantiating this class requires PyTorch; the class itself is created
    dynamically at call time so importing :mod:`psyinsight.dl` never fails
    when torch is absent.
    """

    def __new__(cls, X, y=None):
        torch, Dataset = _lazy_bases()

        X_arr = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
        X_tensor = torch.as_tensor(X_arr, dtype=torch.float32)

        y_tensor = None
        if y is not None:
            y_arr = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
            y_tensor = torch.as_tensor(y_arr, dtype=torch.float32)

        class _TabularDatasetImpl(Dataset):
            def __len__(self_inner):
                return len(X_tensor)

            def __getitem__(self_inner, idx):
                if y_tensor is None:
                    return X_tensor[idx]
                return X_tensor[idx], y_tensor[idx]

        instance = _TabularDatasetImpl()
        instance.n_features = X_tensor.shape[1] if X_tensor.ndim > 1 else 1
        return instance


class SequenceDataset:
    """A ``Dataset`` of padded integer-token sequences + labels, for the
    simple sequence models (LSTM / tiny transformer) in
    :mod:`psyinsight.dl.models`.

    Parameters
    ----------
    sequences: list of token-id lists (already numericalized).
    labels: optional list/array of targets.
    max_len: sequences are truncated/padded (with 0) to this length.
    """

    def __new__(cls, sequences: List[List[int]], labels: Optional[List] = None, max_len: int = 64):
        torch, Dataset = _lazy_bases()

        padded = np.zeros((len(sequences), max_len), dtype=np.int64)
        for i, seq in enumerate(sequences):
            trimmed = seq[:max_len]
            padded[i, : len(trimmed)] = trimmed
        X_tensor = torch.as_tensor(padded, dtype=torch.long)

        y_tensor = None
        if labels is not None:
            y_tensor = torch.as_tensor(np.asarray(labels), dtype=torch.float32)

        class _SequenceDatasetImpl(Dataset):
            def __len__(self_inner):
                return len(X_tensor)

            def __getitem__(self_inner, idx):
                if y_tensor is None:
                    return X_tensor[idx]
                return X_tensor[idx], y_tensor[idx]

        instance = _SequenceDatasetImpl()
        instance.max_len = max_len
        return instance


def build_vocab(texts: List[str], max_vocab_size: int = 5000, min_freq: int = 1) -> dict:
    """Very small whitespace-tokenizer vocabulary builder shared by the
    sequence models. Reserves index 0 for padding and 1 for unknown tokens."""
    from collections import Counter

    counter = Counter()
    for text in texts:
        counter.update(str(text).lower().split())

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common(max_vocab_size):
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def texts_to_sequences(texts: List[str], vocab: dict) -> List[List[int]]:
    """Numericalize whitespace-tokenized text using a vocabulary built by
    :func:`build_vocab`."""
    unk = vocab["<unk>"]
    return [[vocab.get(tok, unk) for tok in str(text).lower().split()] for text in texts]
