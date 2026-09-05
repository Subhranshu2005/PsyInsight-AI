"""
PsyInsight AI — PsyRegressor
============================

Single-file regression counterpart to ``PsyClassifier``: a registry-driven
regression engine that supports algorithm switching, AutoML model
comparison, evaluation, importance / coefficient introspection and
persistence, all through one consistent API.

Quick start
-----------
    from psyinsight.ml.regressor import PsyRegressor

    reg = PsyRegressor()
    X_train, X_test, y_train, y_test = reg.split_data(X, y)
    result = reg.auto_train(X_train, X_test, y_train, y_test)
    reg.evaluate(X_test, y_test)

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from psyinsight.ml.metrics import PsyMetrics
from psyinsight.utils import ModelNotFittedError, get_logger, timeit

warnings.filterwarnings("ignore")
_logger = get_logger(__name__)

__all__ = ["PsyRegressor"]


# =============================================================================
# 1. Model registry
# =============================================================================


class RegressorRegistryMixin:
    """Catalogue of available regression algorithms + factory method."""

    MODEL_REGISTRY = {
        "linear_regression": lambda: LinearRegression(),
        "ridge": lambda: Ridge(alpha=1.0, random_state=42),
        "lasso": lambda: Lasso(alpha=0.1, random_state=42),
        "elastic_net": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "decision_tree": lambda: DecisionTreeRegressor(random_state=42),
        "random_forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
        "gradient_boosting": lambda: GradientBoostingRegressor(random_state=42),
        "svr": lambda: SVR(kernel="rbf"),
        "knn": lambda: KNeighborsRegressor(n_neighbors=5),
    }

    @classmethod
    def available_models(cls) -> List[str]:
        return list(cls.MODEL_REGISTRY.keys())

    def _build_model(self, name: str):
        if name not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{name}'. Available models: {self.available_models()}"
            )
        return self.MODEL_REGISTRY[name]()


# =============================================================================
# 2. Core: data / training / evaluation / persistence
# =============================================================================


class PsyRegressorCore(RegressorRegistryMixin):
    def __init__(self, model_name: str = "linear_regression", scale_features: bool = True):
        self.model_name = model_name
        self.model = self._build_model(model_name)
        self.scale_features = scale_features
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted_: bool = False
        self.metadata_: Dict[str, Any] = {"created_at": datetime.now().isoformat()}

    def set_model(self, name: str, **kwargs) -> "PsyRegressorCore":
        """Switch the active algorithm. Extra ``kwargs`` are forwarded to the
        estimator's constructor when supported."""
        self.model_name = name
        self.model = self._build_model(name)
        if kwargs:
            self.model.set_params(**kwargs)
        self.is_fitted_ = False
        return self

    # ---------------- data ----------------

    def split_data(self, X, y, test_size: float = 0.2, random_state: int = 42):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def _prepare(self, X, fit: bool = False) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if not self.scale_features:
            return X_arr
        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X_arr)
        if self.scaler is None:
            return X_arr
        return self.scaler.transform(X_arr)

    # ---------------- training ----------------

    @timeit
    def fit(self, X, y) -> "PsyRegressorCore":
        if isinstance(X, pd.DataFrame) and self.feature_names_ is None:
            self.feature_names_ = X.columns.tolist()
        X_prepared = self._prepare(X, fit=True)
        self.model.fit(X_prepared, y)
        self.is_fitted_ = True
        self.metadata_["fitted_at"] = datetime.now().isoformat()
        self.metadata_["n_samples"] = len(y)
        return self

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(self._prepare(X))

    def _check_fitted(self):
        if not self.is_fitted_:
            raise ModelNotFittedError("Call .fit(X, y) before using this method.")

    # ---------------- evaluation ----------------

    def evaluate(self, X, y) -> Dict[str, float]:
        preds = self.predict(X)
        report = PsyMetrics.regression_report(y, preds)
        report.update(PsyMetrics.residual_stats(y, preds))
        self.metadata_["last_evaluation"] = report
        return report

    def cross_validate(self, X, y, cv: int = 5, scoring: str = "r2") -> Dict[str, float]:
        """
        K-fold cross-validation.

        Leakage note: earlier versions of this method called
        ``self._prepare(X, fit=True)`` *before* cross-validating, which
        fits the ``StandardScaler`` on the full dataset -- including
        every fold's held-out data -- before any split happens. That is
        classic preprocessing/cross-validation leakage: each fold's
        "unseen" data has already influenced the scaler's mean/variance.
        Fixed by fitting the scaler *inside* each fold via an
        sklearn ``Pipeline``, exactly like ``cross_val_score`` expects.
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        if self.scale_features:
            estimator = make_pipeline(StandardScaler(), clone(self.model))
        else:
            estimator = clone(self.model)

        scores = cross_val_score(estimator, X_arr, y, cv=cv, scoring=scoring)
        return {"mean": float(scores.mean()), "std": float(scores.std()), "scores": scores.tolist()}

    # ---------------- introspection ----------------

    def feature_importance(self) -> pd.DataFrame:
        """Coefficients (linear models) or ``feature_importances_`` (tree
        ensembles) as a tidy, sorted DataFrame."""
        self._check_fitted()
        names = self.feature_names_ or [f"feature_{i}" for i in range(len(self._get_raw_importance()))]

        if hasattr(self.model, "coef_"):
            values = np.ravel(self.model.coef_)
            col = "coefficient"
        elif hasattr(self.model, "feature_importances_"):
            values = self.model.feature_importances_
            col = "importance"
        else:
            raise AttributeError(
                f"Model '{self.model_name}' exposes neither coef_ nor feature_importances_"
            )

        return pd.DataFrame({"feature": names, col: values}).sort_values(
            col, key=abs, ascending=False
        ).reset_index(drop=True)

    def _get_raw_importance(self):
        if hasattr(self.model, "coef_"):
            return np.ravel(self.model.coef_)
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return []

    # ---------------- persistence ----------------

    def save(self, path: str) -> str:
        self._check_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "model_name": self.model_name,
                "feature_names": self.feature_names_,
                "metadata": self.metadata_,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str) -> "PsyRegressorCore":
        """Load a model previously saved with :meth:`save`.

        SECURITY WARNING: this deserializes with ``joblib.load``, which
        (like ``pickle``) can execute arbitrary code embedded in the
        file. Only call this on files you created yourself or that came
        from a fully trusted source -- never on a file uploaded by an
        untrusted user.
        """
        payload = joblib.load(path)
        instance = cls(model_name=payload["model_name"])
        instance.model = payload["model"]
        instance.scaler = payload["scaler"]
        instance.feature_names_ = payload["feature_names"]
        instance.metadata_ = payload["metadata"]
        instance.is_fitted_ = True
        return instance


# =============================================================================
# 3. AutoML: compare every registered algorithm
# =============================================================================


class RegressorAutoMLMixin:
    def compare_models(self, X_train, X_test, y_train, y_test, models: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit every candidate algorithm and rank them by test R^2."""
        models = models or self.available_models()
        rows = []
        failures = []
        for name in models:
            try:
                candidate = PsyRegressorCore(model_name=name, scale_features=self.scale_features)
                candidate.fit(X_train, y_train)
                report = candidate.evaluate(X_test, y_test)
                rows.append({"model": name, **report})
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Model '%s' failed during comparison: %s", name, exc)
                failures.append(f"{name}: {exc}")

        if not rows:
            detail = "; ".join(failures) if failures else "no candidate models were tried"
            raise ValueError(
                "Every candidate model failed to fit -- nothing to compare. "
                f"Check your data (e.g. NaNs in the target, too few rows). Details: {detail}"
            )

        leaderboard = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
        self.leaderboard_ = leaderboard
        return leaderboard

    def auto_train(self, X_train, X_test, y_train, y_test, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare candidates, then refit the best one as ``self.model``."""
        leaderboard = self.compare_models(X_train, X_test, y_train, y_test, models=models)
        best_name = leaderboard.iloc[0]["model"]
        self.set_model(best_name)
        self.fit(X_train, y_train)
        report = self.evaluate(X_test, y_test)
        return {"best_model": best_name, "leaderboard": leaderboard, "test_report": report}


# =============================================================================
# 4. Public class
# =============================================================================


class PsyRegressor(RegressorAutoMLMixin, PsyRegressorCore):
    """Public, fully-composed regression engine for PsyInsight AI."""
