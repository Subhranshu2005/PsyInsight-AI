"""
PsyInsight AI — PsyModelSelector
================================

Unified helpers for splitting data, cross-validating and tuning
hyperparameters, shared by :class:`~psyinsight.ml.classifier.PsyClassifier`,
:class:`~psyinsight.ml.regressor.PsyRegressor` and any user code that wants
consistent, reproducible model-selection utilities.

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    learning_curve,
    train_test_split,
)

from psyinsight.utils import get_logger

__all__ = ["PsyModelSelector"]

_logger = get_logger(__name__)


class PsyModelSelector:
    """A small, composable toolkit for splitting data and searching for the
    best model / hyperparameters.

    Every method is stateless (``staticmethod``) except when it makes sense
    to remember the *last* search result (``self.best_estimator_`` etc.),
    so the class can be used functionally or as a stateful helper.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_estimator_: Optional[BaseEstimator] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.cv_results_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def train_val_test_split(
        self,
        X,
        y=None,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify: bool = False,
    ) -> Tuple[Any, ...]:
        """Split data into train / validation / test sets in one call.

        Returns ``X_train, X_val, X_test, y_train, y_val, y_test`` (the
        ``y_*`` values are ``None`` if ``y`` is ``None``).
        """
        strat = y if (stratify and y is not None) else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=strat
        )

        relative_val = val_size / (1 - test_size)
        strat_temp = y_temp if (stratify and y_temp is not None) else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val, random_state=self.random_state, stratify=strat_temp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate_model(
        self,
        estimator: BaseEstimator,
        X,
        y,
        cv: int = 5,
        scoring: Optional[str] = None,
        stratified: bool = True,
        return_train_score: bool = True,
    ) -> pd.DataFrame:
        """Run k-fold cross-validation and return a tidy results DataFrame."""
        splitter = (
            StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            if stratified
            else KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        )
        results = cross_validate(
            estimator,
            X,
            y,
            cv=splitter,
            scoring=scoring,
            return_train_score=return_train_score,
        )
        self.cv_results_ = pd.DataFrame(results)
        return self.cv_results_

    def quick_cv_score(
        self, estimator: BaseEstimator, X, y, cv: int = 5, scoring: Optional[str] = None
    ) -> Dict[str, float]:
        """Return mean/std of ``cross_val_score`` — a one-liner sanity check."""
        scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
        return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "scores": scores.tolist()}

    # ------------------------------------------------------------------
    # Hyperparameter search
    # ------------------------------------------------------------------

    def grid_search(
        self,
        estimator: BaseEstimator,
        param_grid: Dict[str, List[Any]],
        X,
        y,
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
    ) -> BaseEstimator:
        """Exhaustive grid search. Stores the winner on ``self.best_estimator_``."""
        search = GridSearchCV(
            estimator, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, refit=True
        )
        search.fit(X, y)
        self._store_search_results(search)
        return search.best_estimator_

    def random_search(
        self,
        estimator: BaseEstimator,
        param_distributions: Dict[str, Any],
        X,
        y,
        n_iter: int = 25,
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
    ) -> BaseEstimator:
        """Randomized hyperparameter search — much cheaper than a full grid
        when the search space is large."""
        search = RandomizedSearchCV(
            estimator,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=self.random_state,
            refit=True,
        )
        search.fit(X, y)
        self._store_search_results(search)
        return search.best_estimator_

    def _store_search_results(self, search) -> None:
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = float(search.best_score_)
        self.cv_results_ = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_score"
        ).reset_index(drop=True)
        _logger.info("Best score=%.4f params=%s", self.best_score_, self.best_params_)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def learning_curve_data(
        self,
        estimator: BaseEstimator,
        X,
        y,
        cv: int = 5,
        train_sizes: Optional[Sequence[float]] = None,
        scoring: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute learning-curve data (train size vs. train/val score) for
        diagnosing over/under-fitting."""
        sizes = np.linspace(0.1, 1.0, 5) if train_sizes is None else np.asarray(train_sizes)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=sizes, scoring=scoring,
            random_state=self.random_state,
        )
        return {
            "train_sizes": train_sizes_abs,
            "train_scores_mean": train_scores.mean(axis=1),
            "train_scores_std": train_scores.std(axis=1),
            "val_scores_mean": val_scores.mean(axis=1),
            "val_scores_std": val_scores.std(axis=1),
        }

    @staticmethod
    def detect_overfitting(train_score: float, val_score: float, threshold: float = 0.1) -> Dict[str, Any]:
        """Simple heuristic diagnosis based on the train/validation score gap."""
        gap = train_score - val_score
        if gap > threshold:
            verdict = "overfitting"
        elif train_score < 0.6 and val_score < 0.6:
            verdict = "underfitting"
        else:
            verdict = "good fit"
        return {"train_score": train_score, "val_score": val_score, "gap": gap, "verdict": verdict}
