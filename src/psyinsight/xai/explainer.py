"""
PsyInsight AI — PsyExplainer (Explainable AI module)
=======================================================

Model-agnostic explainability toolkit: permutation importance, partial
dependence, a lightweight LIME-style local surrogate explanation, and an
optional SHAP wrapper (used automatically when the ``shap`` package is
installed). Every method works with *any* fitted scikit-learn-compatible
estimator (including the PsyInsight ``PsyClassifier`` / ``PsyRegressor``
wrappers, since they expose ``.predict`` / ``.model``).

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import get_scorer

from psyinsight.utils import get_logger

__all__ = ["PsyExplainer"]

_logger = get_logger(__name__)


def _resolve_predict_fn(model) -> Callable:
    """Accept either a plain sklearn estimator or a PsyInsight wrapper
    (``PsyClassifier`` / ``PsyRegressor``) and return a ``.predict``-style
    callable + the underlying raw estimator for sklearn-native utilities."""
    if hasattr(model, "predict") and not hasattr(model, "model"):
        return model.predict, model
    if hasattr(model, "model"):
        return model.predict, model  # PsyInsight wrappers implement .predict themselves
    raise TypeError("model must expose a .predict method")


class PsyExplainer:
    """Explain the predictions of a fitted model.

    Parameters
    ----------
    model: fitted estimator (sklearn-compatible, or a PsyInsight
        ``PsyClassifier`` / ``PsyRegressor`` instance).
    feature_names: optional list of feature names (inferred from
        ``model.feature_names_`` when available).
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        self.model = model
        self._predict, self._estimator_for_sklearn_utils = _resolve_predict_fn(model)
        self.feature_names = feature_names or getattr(model, "feature_names_", None)

    # ------------------------------------------------------------------
    # Global importance
    # ------------------------------------------------------------------

    def global_feature_importance(self) -> pd.DataFrame:
        """Return native feature importance (coefficients / Gini importance)
        from the model when available — the cheapest, fastest explanation."""
        if hasattr(self.model, "feature_importance"):
            try:
                result = self.model.feature_importance(feature_names=self.feature_names)
            except TypeError:
                result = self.model.feature_importance()
            if isinstance(result, pd.DataFrame):
                return result
            names = self.feature_names or [f"feature_{i}" for i in range(len(result))]
            return pd.DataFrame({"feature": names, "importance": result}).sort_values(
                "importance", key=abs, ascending=False
            ).reset_index(drop=True)

        est = self.model.model if hasattr(self.model, "model") else self.model
        names = self.feature_names or [f"feature_{i}" for i in range(_n_features(est))]

        if hasattr(est, "feature_importances_"):
            values = est.feature_importances_
            col = "importance"
        elif hasattr(est, "coef_"):
            values = np.ravel(est.coef_)
            col = "coefficient"
        else:
            raise AttributeError("Model does not expose native importances; use permutation_importance instead")

        return pd.DataFrame({"feature": names, col: values}).sort_values(
            col, key=abs, ascending=False
        ).reset_index(drop=True)

    def permutation_importance(
        self, X, y, scoring: Optional[str] = None, n_repeats: int = 10, random_state: int = 42
    ) -> pd.DataFrame:
        """Model-agnostic permutation feature importance: how much a metric
        degrades when each feature's values are randomly shuffled. Works for
        *any* model, unlike native importances."""
        est = self.model.model if hasattr(self.model, "model") else self.model
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        names = self.feature_names or (X.columns.tolist() if isinstance(X, pd.DataFrame) else
                                        [f"feature_{i}" for i in range(X_arr.shape[1])])

        result = permutation_importance(
            est, X_arr, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state
        )
        return pd.DataFrame(
            {
                "feature": names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Partial dependence (marginal effect of one feature)
    # ------------------------------------------------------------------

    def partial_dependence(self, X, feature: str, grid_resolution: int = 30) -> pd.DataFrame:
        """Return the partial-dependence curve for a single feature — how
        the model's average prediction changes as that feature varies,
        holding everything else at its observed distribution."""
        est = self.model.model if hasattr(self.model, "model") else self.model
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        feature_idx = X_df.columns.get_loc(feature)

        pd_result = partial_dependence(
            est, X_df, features=[feature_idx], grid_resolution=grid_resolution, kind="average"
        )
        return pd.DataFrame(
            {"value": pd_result["grid_values"][0], "avg_prediction": pd_result["average"][0]}
        )

    # ------------------------------------------------------------------
    # Local explanation (LIME-style linear surrogate)
    # ------------------------------------------------------------------

    def explain_instance(
        self, X, instance_index: int, n_samples: int = 500, kernel_width: float = 0.75
    ) -> pd.DataFrame:
        """Explain a single prediction with a local linear surrogate model
        (a simplified LIME): perturb the instance's numeric features,
        query the black-box model, and fit a weighted ridge regression to
        the perturbations. The surrogate's coefficients approximate each
        feature's local contribution to that one prediction.
        """
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        instance = X_df.iloc[instance_index].values.astype(float)
        feature_names = X_df.columns.tolist()
        std = X_df.std(axis=0).replace(0, 1.0).values

        rng = np.random.RandomState(42)
        noise = rng.normal(0, 1, size=(n_samples, len(instance))) * std
        perturbed = instance + noise
        perturbed_df = pd.DataFrame(perturbed, columns=feature_names)

        predictions = np.asarray(self._predict(perturbed_df), dtype=float).reshape(-1)

        distances = np.linalg.norm((perturbed - instance) / std, axis=1)
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        surrogate = Ridge(alpha=1.0)
        surrogate.fit(perturbed, predictions, sample_weight=weights)

        return pd.DataFrame(
            {"feature": feature_names, "local_contribution": surrogate.coef_, "instance_value": instance}
        ).sort_values("local_contribution", key=abs, ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Optional SHAP integration
    # ------------------------------------------------------------------

    def shap_values(self, X, max_samples: int = 200):
        """Compute SHAP values if the optional ``shap`` package is
        installed; raises a clear ``ImportError`` otherwise."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP values require the optional 'shap' package: pip install shap"
            ) from exc

        est = self.model.model if hasattr(self.model, "model") else self.model
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        sample = X_df.sample(min(max_samples, len(X_df)), random_state=42)

        try:
            explainer = shap.TreeExplainer(est)
        except Exception:  # noqa: BLE001
            explainer = shap.KernelExplainer(est.predict, shap.sample(X_df, 50))

        return explainer.shap_values(sample)

    # ------------------------------------------------------------------
    # Convenience report
    # ------------------------------------------------------------------

    def full_report(self, X, y, top_k: int = 10) -> Dict[str, Any]:
        """One-call explainability report: global importance, permutation
        importance and a single-instance local explanation for the
        highest-confidence and lowest-confidence sample."""
        report: Dict[str, Any] = {}
        try:
            report["global_importance"] = self.global_feature_importance().head(top_k)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("global_feature_importance unavailable: %s", exc)

        try:
            report["permutation_importance"] = self.permutation_importance(X, y).head(top_k)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("permutation_importance unavailable: %s", exc)

        try:
            report["local_explanation_sample_0"] = self.explain_instance(X, 0).head(top_k)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("local explanation unavailable: %s", exc)

        return report


def _n_features(estimator) -> int:
    if hasattr(estimator, "n_features_in_"):
        return int(estimator.n_features_in_)
    if hasattr(estimator, "coef_"):
        return int(np.ravel(estimator.coef_).shape[0])
    raise AttributeError("Could not infer number of features from estimator")
