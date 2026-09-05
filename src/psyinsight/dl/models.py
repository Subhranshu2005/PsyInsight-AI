"""
PsyInsight AI — Deep Learning Models
=======================================

A small collection of PyTorch model architectures scoped to the kind of
data psychology researchers actually work with: tabular questionnaire
data (MLP / autoencoder) and short text responses (LSTM / a tiny
transformer encoder). Nothing here needs a GPU cluster to train.

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

from typing import List, Optional

from ._torch_guard import require_torch

__all__ = ["MLPClassifier", "MLPRegressor", "Autoencoder", "LSTMTextClassifier", "TinyTransformerClassifier"]


def _nn():
    torch, nn = require_torch()
    return torch, nn


class MLPClassifier:
    """Feed-forward classifier for tabular data. Returned instance is a
    genuine ``torch.nn.Module`` — the wrapping just defers the torch import.
    """

    def __new__(
        cls,
        input_dim: int,
        n_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        torch, nn = _nn()
        hidden_dims = hidden_dims or [64, 32]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes if n_classes > 2 else 1))

        class _MLPClassifierImpl(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.net = nn.Sequential(*layers)
                self_inner.n_classes = n_classes

            def forward(self_inner, x):
                return self_inner.net(x)

        return _MLPClassifierImpl()


class MLPRegressor:
    """Feed-forward regressor for tabular data (single continuous output)."""

    def __new__(cls, input_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.1):
        torch, nn = _nn()
        hidden_dims = hidden_dims or [64, 32]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        class _MLPRegressorImpl(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.net = nn.Sequential(*layers)

            def forward(self_inner, x):
                return self_inner.net(x).squeeze(-1)

        return _MLPRegressorImpl()


class Autoencoder:
    """Simple symmetric autoencoder — useful for anomaly detection in
    questionnaire responses or as an unsupervised feature-compression step
    ahead of clustering."""

    def __new__(cls, input_dim: int, latent_dim: int = 8, hidden_dim: int = 32):
        torch, nn = _nn()

        class _AutoencoderImpl(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim),
                )
                self_inner.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )

            def forward(self_inner, x):
                z = self_inner.encoder(x)
                return self_inner.decoder(z)

            def encode(self_inner, x):
                return self_inner.encoder(x)

        return _AutoencoderImpl()


class LSTMTextClassifier:
    """A small embedding + bidirectional-LSTM classifier for short survey
    free-text responses (open-ended answers, sentiment labeling, etc.)."""

    def __new__(
        cls,
        vocab_size: int,
        n_classes: int = 2,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.2,
    ):
        torch, nn = _nn()

        class _LSTMTextClassifierImpl(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self_inner.lstm = nn.LSTM(
                    embed_dim,
                    hidden_dim,
                    num_layers=n_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if n_layers > 1 else 0.0,
                )
                out_dim = n_classes if n_classes > 2 else 1
                self_inner.classifier = nn.Linear(hidden_dim * 2, out_dim)
                self_inner.dropout = nn.Dropout(dropout)

            def forward(self_inner, x):
                embedded = self_inner.embedding(x)
                _, (hidden, _) = self_inner.lstm(embedded)
                # concat last layer's forward + backward hidden states
                final = torch.cat((hidden[-2], hidden[-1]), dim=1)
                return self_inner.classifier(self_inner.dropout(final))

        return _LSTMTextClassifierImpl()


class TinyTransformerClassifier:
    """A minimal transformer-encoder text classifier: embedding +
    positional encoding + a couple of ``nn.TransformerEncoderLayer``s +
    mean pooling + linear head. Small enough to train on CPU for
    demonstration / coursework purposes; for production-grade NLP use
    :mod:`psyinsight.transformers` (Hugging Face pretrained models) instead.
    """

    def __new__(
        cls,
        vocab_size: int,
        n_classes: int = 2,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        torch, nn = _nn()

        class _TinyTransformerImpl(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self_inner.pos_embedding = nn.Embedding(max_len, embed_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self_inner.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                out_dim = n_classes if n_classes > 2 else 1
                self_inner.classifier = nn.Linear(embed_dim, out_dim)

            def forward(self_inner, x):
                positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
                embedded = self_inner.embedding(x) + self_inner.pos_embedding(positions)
                padding_mask = x == 0
                encoded = self_inner.encoder(embedded, src_key_padding_mask=padding_mask)
                pooled = encoded.mean(dim=1)
                return self_inner.classifier(pooled)

        return _TinyTransformerImpl()
