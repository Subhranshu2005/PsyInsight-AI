"""
PsyInsight AI — Deep Learning module
=======================================

PyTorch-based models and training utilities for tabular questionnaire data
and short free-text survey responses.

This module is import-safe even when PyTorch is not installed: importing
``psyinsight.dl`` never raises, but instantiating a model / Trainer without
torch installed raises a clear ``ImportError`` with installation
instructions.

    from psyinsight.dl import MLPClassifier, MLPRegressor, Trainer
    from psyinsight.dl import TabularDataset, SequenceDataset
    from psyinsight.dl import LSTMTextClassifier, TinyTransformerClassifier, Autoencoder
"""

from ._torch_guard import TORCH_AVAILABLE
from .datasets import SequenceDataset, TabularDataset, build_vocab, texts_to_sequences
from .models import (
    Autoencoder,
    LSTMTextClassifier,
    MLPClassifier,
    MLPRegressor,
    TinyTransformerClassifier,
)
from .trainer import Trainer

__all__ = [
    "TORCH_AVAILABLE",
    "TabularDataset",
    "SequenceDataset",
    "build_vocab",
    "texts_to_sequences",
    "MLPClassifier",
    "MLPRegressor",
    "Autoencoder",
    "LSTMTextClassifier",
    "TinyTransformerClassifier",
    "Trainer",
]
