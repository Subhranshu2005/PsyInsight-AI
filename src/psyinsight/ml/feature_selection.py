"""
PsyInsight AI — PsyFeatureSelector
==================================

Feature-selection / dimensionality-reduction toolkit: variance filtering,
correlation-based redundancy removal, univariate statistical tests,
model-based importance ranking, recursive feature elimination and PCA —
each exposed as a single well-documented method so the pipeline stays
transparent (important for research reproducibility).

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import StandardScaler

from psyinsight.utils import ensure_dataframe, get_logger

__all__ = ["PsyFeatureSelector"]

_logger = get_logger(__name__)


class PsyFeatureSelector:
    """Feature-selection toolkit operating on a pandas DataFrame.

    All methods return the *names* of the selected/removed features (rather
    than mutating the DataFrame in place) so callers stay in full control of
    their pipeline.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = ensure_dataframe(dataframe)
        self.report_: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Variance-based filtering
    # ------------------------------------------------------------------

    def low_variance_features(self, threshold: float = 0.0) -> List[str]:
        """Return numeric columns whose variance is at or below ``threshold``
        (constant / near-constant columns carry no information)."""
        numeric = self.df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        if numeric.empty:
            return []
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(numeric.fillna(numeric.mean()))
        mask = ~selector.get_support()
        removed = numeric.columns[mask].tolist()
        self.report_["low_variance_removed"] = removed
        return removed

    # ------------------------------------------------------------------
    # Correlation-based redundancy removal
    # ------------------------------------------------------------------

    def highly_correlated_features(self, threshold: float = 0.9) -> List[str]:
        """Identify redundant numeric features using pairwise Pearson
        correlation; for each highly-correlated pair, the second feature is
        flagged for removal (keeping the first, arbitrary but deterministic)."""
        numeric = self.df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return []
        corr_matrix = numeric.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        self.report_["correlation_removed"] = to_drop
        return to_drop

    def correlation_matrix(self) -> pd.DataFrame:
        """Full pairwise Pearson correlation matrix of numeric columns."""
        return self.df.select_dtypes(include=[np.number]).corr()

    # ------------------------------------------------------------------
    # Univariate statistical selection
    # ------------------------------------------------------------------

    def univariate_scores(
        self, target: str, task: str = "classification", k: str | int = "all"
    ) -> pd.DataFrame:
        """Rank numeric features by an ANOVA-F (classification) or
        F-regression (regression) statistical test against ``target``.
        Returns a DataFrame sorted by descending score."""
        X, y = self._xy(target)
        score_func = f_classif if task == "classification" else f_regression
        selector = SelectKBest(score_func=score_func, k="all")
        selector.fit(X, y)
        result = pd.DataFrame(
            {"feature": X.columns, "score": selector.scores_, "p_value": selector.pvalues_}
        ).sort_values("score", ascending=False).reset_index(drop=True)
        if k != "all":
            result = result.head(int(k))
        return result

    def mutual_information(self, target: str, task: str = "classification") -> pd.DataFrame:
        """Mutual-information based feature ranking, which — unlike ANOVA-F —
        also captures non-linear dependencies between a feature and the
        target."""
        X, y = self._xy(target)
        func = mutual_info_classif if task == "classification" else mutual_info_regression
        scores = func(X, y, random_state=42)
        return pd.DataFrame({"feature": X.columns, "mutual_info": scores}).sort_values(
            "mutual_info", ascending=False
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Model-based importance
    # ------------------------------------------------------------------

    def model_based_importance(self, target: str, task: str = "classification") -> pd.DataFrame:
        """Feature importances from a Random Forest — captures interactions
        that univariate tests miss."""
        X, y = self._xy(target)
        model = (
            RandomForestClassifier(n_estimators=200, random_state=42)
            if task == "classification"
            else RandomForestRegressor(n_estimators=200, random_state=42)
        )
        model.fit(X, y)
        return pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

    def recursive_feature_elimination(
        self, target: str, n_features: int = 5, task: str = "classification"
    ) -> List[str]:
        """Recursive Feature Elimination (RFE) using a Random Forest as the
        base estimator; returns the names of the ``n_features`` selected."""
        X, y = self._xy(target)
        estimator = (
            RandomForestClassifier(n_estimators=100, random_state=42)
            if task == "classification"
            else RandomForestRegressor(n_estimators=100, random_state=42)
        )
        n_features = max(1, min(n_features, X.shape[1]))
        selector = RFE(estimator, n_features_to_select=n_features)
        selector.fit(X, y)
        return X.columns[selector.support_].tolist()

    # ------------------------------------------------------------------
    # Dimensionality reduction
    # ------------------------------------------------------------------

    def pca_reduce(
        self, n_components: Optional[int] = None, variance_threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Reduce numeric features with PCA.

        If ``n_components`` is ``None``, enough components are kept to reach
        ``variance_threshold`` cumulative explained variance.
        """
        numeric = self.df.select_dtypes(include=[np.number]).dropna()
        scaled = StandardScaler().fit_transform(numeric)

        if n_components is None:
            pca_full = PCA().fit(scaled)
            cumulative = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)

        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(scaled)
        columns = [f"PC{i + 1}" for i in range(n_components)]
        result_df = pd.DataFrame(transformed, columns=columns, index=numeric.index)

        info = {
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": float(np.sum(pca.explained_variance_ratio_)),
            "loadings": pd.DataFrame(
                pca.components_.T, index=numeric.columns, columns=columns
            ),
        }
        return result_df, info

    # ------------------------------------------------------------------
    # Convenience: run everything and get a consolidated recommendation
    # ------------------------------------------------------------------

    def auto_select(
        self, target: str, task: str = "classification", top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """One-call pipeline: drop low-variance & redundant features, then
        rank the remainder by model-based importance. Returns a report dict
        with a final recommended feature list."""
        low_var = self.low_variance_features()
        redundant = self.highly_correlated_features()
        dropped = set(low_var) | set(redundant)

        remaining_df = self.df.drop(columns=[c for c in dropped if c != target], errors="ignore")
        selector = PsyFeatureSelector(remaining_df)
        importance = selector.model_based_importance(target=target, task=task)

        recommended = importance["feature"].tolist()
        if top_k is not None:
            recommended = recommended[:top_k]

        return {
            "dropped_low_variance": low_var,
            "dropped_redundant": redundant,
            "importance_ranking": importance,
            "recommended_features": recommended,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _xy(self, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        if target not in self.df.columns:
            raise KeyError(f"target column '{target}' not found in dataframe")
        working = self.df.dropna()
        X = working.drop(columns=[target]).select_dtypes(include=[np.number])
        y = working[target]
        return X, y
