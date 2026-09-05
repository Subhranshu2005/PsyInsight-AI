"""
PsyInsight AI — PsyMetrics
==========================

Unified metrics utility for the whole ML module: a single place that knows
how to score classification, regression *and* clustering results, plus a
few psychology-research-flavoured extras (Cohen's d, reliability, effect
size interpretation) that plain scikit-learn does not provide.

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics as skm

__all__ = ["PsyMetrics"]


class PsyMetrics:
    """Namespace-style helper class. All methods are ``staticmethod`` so the
    class can be used either as ``PsyMetrics.accuracy(...)`` or instantiated
    if a caller prefers an object-oriented style.
    """

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def classification_report(y_true, y_pred, y_proba: Optional[np.ndarray] = None,
                               average: str = "weighted") -> Dict[str, Any]:
        """Return a dict of the most common classification metrics.

        ``y_proba`` (predicted probabilities, shape ``(n_samples, n_classes)``
        or ``(n_samples,)`` for binary) is optional; when supplied, ROC-AUC
        is added to the report.
        """
        report: Dict[str, Any] = {
            "accuracy": skm.accuracy_score(y_true, y_pred),
            "precision": skm.precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": skm.recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": skm.f1_score(y_true, y_pred, average=average, zero_division=0),
            "confusion_matrix": skm.confusion_matrix(y_true, y_pred).tolist(),
            "n_classes": int(len(np.unique(y_true))),
        }

        if y_proba is not None:
            try:
                n_classes = report["n_classes"]
                if n_classes == 2:
                    proba = y_proba[:, 1] if np.ndim(y_proba) == 2 else y_proba
                    report["roc_auc"] = skm.roc_auc_score(y_true, proba)
                else:
                    report["roc_auc"] = skm.roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average=average
                    )
            except Exception:  # noqa: BLE001
                report["roc_auc"] = None

        return report

    @staticmethod
    def per_class_report(y_true, y_pred) -> pd.DataFrame:
        """Precision / recall / f1 / support broken down per class as a DataFrame."""
        report_dict = skm.classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return pd.DataFrame(report_dict).transpose()

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    @staticmethod
    def regression_report(y_true, y_pred) -> Dict[str, float]:
        """Return MAE, MSE, RMSE, R^2, MAPE and explained variance."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mse = skm.mean_squared_error(y_true, y_pred)
        non_zero = y_true != 0
        mape = (
            float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
            if non_zero.any()
            else float("nan")
        )

        return {
            "mae": skm.mean_absolute_error(y_true, y_pred),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": skm.r2_score(y_true, y_pred),
            "mape": mape,
            "explained_variance": skm.explained_variance_score(y_true, y_pred),
            "max_error": skm.max_error(y_true, y_pred),
        }

    @staticmethod
    def residual_stats(y_true, y_pred) -> Dict[str, float]:
        """Descriptive statistics of the residuals — useful for diagnosing
        heteroscedasticity or bias in a regression model."""
        residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        return {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
            "min_residual": float(np.min(residuals)),
            "max_residual": float(np.max(residuals)),
        }

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    @staticmethod
    def clustering_report(X, labels) -> Dict[str, Any]:
        """Internal clustering-validity metrics (no ground truth required)."""
        labels = np.asarray(labels)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        report: Dict[str, Any] = {"n_clusters": int(n_clusters)}

        if n_clusters >= 2 and n_clusters < len(X):
            try:
                report["silhouette"] = float(skm.silhouette_score(X, labels))
                report["calinski_harabasz"] = float(skm.calinski_harabasz_score(X, labels))
                report["davies_bouldin"] = float(skm.davies_bouldin_score(X, labels))
            except Exception:  # noqa: BLE001
                pass
        return report

    # ------------------------------------------------------------------
    # Psychology-flavoured effect sizes / reliability
    # ------------------------------------------------------------------

    @staticmethod
    def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
        """Cohen's d effect size for the difference between two independent
        samples (pooled standard deviation)."""
        a, b = np.asarray(group_a, dtype=float), np.asarray(group_b, dtype=float)
        n_a, n_b = len(a), len(b)
        pooled_std = np.sqrt(
            ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
        )
        if pooled_std == 0:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / pooled_std)

    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        """Standard (Cohen, 1988) verbal interpretation of an effect size."""
        magnitude = abs(d)
        if magnitude < 0.2:
            return "negligible"
        if magnitude < 0.5:
            return "small"
        if magnitude < 0.8:
            return "medium"
        return "large"

    @staticmethod
    def cronbach_alpha(item_scores: pd.DataFrame) -> float:
        """Cronbach's alpha internal-consistency reliability coefficient for
        a DataFrame of items (columns) x respondents (rows) — the standard
        reliability check for psychometric questionnaires / scales."""
        df = item_scores.dropna()
        k = df.shape[1]
        if k < 2:
            return float("nan")
        item_variances = df.var(axis=0, ddof=1)
        total_variance = df.sum(axis=1).var(ddof=1)
        if total_variance == 0:
            return 0.0
        alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
        return float(alpha)

    @staticmethod
    def interpret_alpha(alpha: float) -> str:
        """Common rule-of-thumb interpretation of Cronbach's alpha."""
        if np.isnan(alpha):
            return "undefined (need >= 2 items)"
        if alpha >= 0.9:
            return "excellent"
        if alpha >= 0.8:
            return "good"
        if alpha >= 0.7:
            return "acceptable"
        if alpha >= 0.6:
            return "questionable"
        if alpha >= 0.5:
            return "poor"
        return "unacceptable"
