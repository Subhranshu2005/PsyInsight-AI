"""
PsyInsight AI — Trainer
=========================

A small, framework-agnostic training loop wrapper (``fit`` / ``evaluate`` /
``predict`` / early stopping / checkpointing) shared by every model in
:mod:`psyinsight.dl.models`, so users never have to hand-write a PyTorch
training loop.

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ._torch_guard import require_torch
from psyinsight.utils import get_logger

__all__ = ["Trainer"]

_logger = get_logger(__name__)


class Trainer:
    """Generic supervised-training loop for any ``torch.nn.Module``.

    Parameters
    ----------
    model: a torch ``nn.Module`` (e.g. from :mod:`psyinsight.dl.models`).
    task: ``"binary"``, ``"multiclass"`` or ``"regression"`` — controls the
        default loss function and metric computation.
    lr: learning rate for the Adam optimizer.
    device: ``"cpu"`` / ``"cuda"``; auto-detected if ``None``.
    """

    def __init__(self, model, task: str = "binary", lr: float = 1e-3, device: Optional[str] = None):
        torch, nn = require_torch()
        self._torch = torch
        self._nn = nn

        self.model = model
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = self._default_criterion(task)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def _default_criterion(self, task: str):
        nn = self._nn
        if task == "binary":
            return nn.BCEWithLogitsLoss()
        if task == "multiclass":
            return nn.CrossEntropyLoss()
        if task == "regression":
            return nn.MSELoss()
        raise ValueError("task must be one of 'binary', 'multiclass', 'regression'")

    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 20,
        patience: int = 5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train with optional early stopping on validation loss.

        Both loaders should yield ``(X_batch, y_batch)`` tuples (see
        :mod:`psyinsight.dl.datasets`).
        """
        torch = self._torch
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self._compute_loss(output, y_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                n_batches += 1

            train_loss = running_loss / max(n_batches, 1)
            self.history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader)
                self.history["val_loss"].append(val_loss)

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    epochs_without_improvement += 1

            if verbose:
                msg = f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" - val_loss={val_loss:.4f}"
                _logger.info(msg)

            if val_loader is not None and epochs_without_improvement >= patience:
                _logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    def _compute_loss(self, output, y_batch):
        if self.task == "binary":
            return self.criterion(output.squeeze(-1), y_batch.float())
        if self.task == "multiclass":
            return self.criterion(output, y_batch.long())
        return self.criterion(output.squeeze(-1), y_batch.float())

    def _evaluate_loss(self, loader) -> float:
        self.model.eval()
        total, n_batches = 0.0, 0
        with self._torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                output = self.model(X_batch)
                loss = self._compute_loss(output, y_batch)
                total += loss.item()
                n_batches += 1
        return total / max(n_batches, 1)

    # ------------------------------------------------------------------

    def predict(self, loader) -> np.ndarray:
        """Return raw model outputs (logits for classification tasks,
        predicted values for regression) for every sample in ``loader``."""
        self.model.eval()
        torch = self._torch
        outputs = []
        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
                X_batch = X_batch.to(self.device)
                outputs.append(self.model(X_batch).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, loader) -> np.ndarray:
        """Sigmoid/softmax-normalized probabilities, for classification tasks."""
        torch = self._torch
        raw = self.predict(loader)
        raw_t = torch.as_tensor(raw)
        if self.task == "binary":
            return torch.sigmoid(raw_t).numpy()
        if self.task == "multiclass":
            return torch.softmax(raw_t, dim=-1).numpy()
        raise ValueError("predict_proba is only defined for classification tasks")

    # ------------------------------------------------------------------

    def save(self, path: str) -> str:
        self._torch.save(self.model.state_dict(), path)
        return path

    def load(self, path: str) -> "Trainer":
        state = self._torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        return self
