"""
PsyInsight AI — PsyClustering
=============================

Advanced Unsupervised Learning Framework

Author:
Subhranshu Ranjan Sahoo

Part 1A
--------
✓ Imports
✓ Cluster Registry
✓ Core Initialization
✓ Model Management
✓ Data Preprocessing
✓ Auto-clustering / model comparison
✓ Visualization utilities
✓ Reporting, export & project persistence

Part 1B (this revision)
--------
✓ Extended reporting (markdown / html / dict / csv export)
✓ Advanced visualizations (profiles, radar, parallel coords,
  boxplots, violin, density, correlation heatmap, pairplot)
✓ External-validation metrics (ARI, NMI, homogeneity,
  completeness, V-measure, Fowlkes-Mallows, entropy, contingency)
✓ Production utilities (cloning, params, introspection,
  batch operations, benchmarking, JSON (de)serialization)
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import sys
import time
import warnings
from copy import deepcopy
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd

# ==========================
# Clustering Algorithms
# ==========================

from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    DBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
    Birch,
    OPTICS,
)
from sklearn.mixture import GaussianMixture

# ==========================
# Preprocessing / Decomposition
# ==========================

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# ==========================
# Evaluation (internal)
# ==========================

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# ==========================
# Evaluation (external / requires ground truth)
# ==========================

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    homogeneity_completeness_v_measure,
    fowlkes_mallows_score,
    pair_confusion_matrix,
    confusion_matrix as sk_confusion_matrix,
)

# ==========================
# Plotting
# ==========================

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

__version__ = "1.2.0"

# ==========================
# Logging
# ==========================

logger = logging.getLogger("psyinsight.clustering")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[PsyClustering] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.WARNING)  # silent by default; raised to INFO when verbose=True


# ==========================
# Custom Exceptions
# ==========================
#
# All of these subclass ValueError so existing `except ValueError` call
# sites (including any in this module) keep working unchanged, while
# callers who want finer-grained handling can catch the specific type.

class PsyClusteringError(ValueError):
    """Base class for all PsyClustering-specific errors."""


class ModelNotSelectedError(PsyClusteringError):
    """Raised when an operation requires a model but none was set via set_model()."""

    def __init__(self, message: str = "No model selected.\nCall set_model() first."):
        super().__init__(message)


class ClusterNotFittedError(PsyClusteringError):
    """Raised when an operation requires a fitted model but fit() hasn't been called."""

    def __init__(self, message: str = "Model has not been fitted.\nCall fit() or fit_predict() first."):
        super().__init__(message)


class InvalidClusterError(PsyClusteringError):
    """Raised for invalid cluster references (e.g. an out-of-range cluster id)."""


# ==========================
# Registry
# ==========================


class ClusterRegistryMixin:
    """
    Registry of every clustering algorithm.

    Adding one algorithm here automatically
    makes it available throughout the framework.
    """

    _REGISTRY = {

        "kmeans": {
            "label": "K-Means",
            "builder": lambda rs, **kw:
                KMeans(
                    n_clusters=kw.pop("n_clusters", 3),
                    random_state=rs,
                    n_init=kw.pop("n_init", "auto"),
                    **kw,
                ),
        },

        "minibatch_kmeans": {
            "label": "MiniBatch K-Means",
            "builder": lambda rs, **kw:
                MiniBatchKMeans(
                    n_clusters=kw.pop("n_clusters", 3),
                    random_state=rs,
                    **kw,
                ),
        },

        "dbscan": {
            "label": "DBSCAN",
            "builder": lambda rs, **kw:
                DBSCAN(
                    eps=kw.pop("eps", 0.5),
                    min_samples=kw.pop("min_samples", 5),
                    **kw,
                ),
        },

        "agglomerative": {
            "label": "Agglomerative Clustering",
            "builder": lambda rs, **kw:
                AgglomerativeClustering(
                    n_clusters=kw.pop("n_clusters", 3),
                    **kw,
                ),
        },

        "spectral": {
            "label": "Spectral Clustering",
            "builder": lambda rs, **kw:
                SpectralClustering(
                    n_clusters=kw.pop("n_clusters", 3),
                    random_state=rs,
                    **kw,
                ),
        },

        "birch": {
            "label": "Birch",
            "builder": lambda rs, **kw:
                Birch(
                    n_clusters=kw.pop("n_clusters", 3),
                    **kw,
                ),
        },

        "optics": {
            "label": "OPTICS",
            "builder": lambda rs, **kw:
                OPTICS(
                    min_samples=kw.pop("min_samples", 5),
                    **kw,
                ),
        },

        "gmm": {
            "label": "Gaussian Mixture Model",
            "builder": lambda rs, **kw:
                GaussianMixture(
                    n_components=kw.pop("n_components", 3),
                    random_state=rs,
                    **kw,
                ),
        },

    }

    @classmethod
    def available_models(cls):
        return list(cls._REGISTRY.keys())

    @classmethod
    def model_label(cls, key):
        if key not in cls._REGISTRY:
            raise ValueError(f"Unknown model '{key}'.")
        return cls._REGISTRY[key]["label"]

    def _build(self, key, **kwargs):
        if key not in self._REGISTRY:
            raise ValueError(f"Unknown model '{key}'.")
        return self._REGISTRY[key]["builder"](self.random_state, **kwargs)


# ======================================================
# Core
# ======================================================

class PsyClusteringCore(ClusterRegistryMixin):
    """
    Core clustering engine.

    Handles
    • model management
    • preprocessing
    • fitting
    • prediction
    • persistence
    """

    def __init__(self, random_state: int = 42, verbose: bool = False):
        self.random_state = random_state
        self.model = None
        self.model_name = None
        self.model_key = None
        self.scaler = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.is_fitted = False
        self.set_verbose(verbose)

    # ==================================================
    # Reproducibility / logging toggles
    # ==================================================

    def set_random_state(self, random_state: int):
        """
        Update the random_state used for any future set_model()/fit() calls.
        Does not retroactively change an already-built estimator's own
        random_state — call set_model() again (or set_params()) for that.
        """
        self.random_state = random_state
        logger.info(f"random_state set to {random_state}")
        return self

    def set_verbose(self, verbose: bool = True):
        """
        Toggle verbose logging for this instance. When True, INFO-level
        messages (model selection, fit timing, warnings) are emitted via
        the 'psyinsight.clustering' logger.
        """
        self.verbose = verbose
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        return self

    # ==================================================
    # Model Management
    # ==================================================

    def set_model(self, model_name: str, **parameters):
        self.model = self._build(model_name, **parameters)
        self.model_name = self.model_label(model_name)
        self.model_key = model_name
        self.is_fitted = False
        logger.info(f"Model set to {self.model_name} ({model_name}) with params {parameters}")
        return self

    def current_model(self):
        return self.model_name

    def available_algorithms(self):
        return {key: self.model_label(key) for key in self.available_models()}

    def model_info(self):
        if self.model is None:
            raise ValueError("No model selected.")
        return {
            "Model": self.model_name,
            "Type": type(self.model).__name__,
            "Parameters": self.model.get_params(),
            "Fitted": self.is_fitted,
        }

    def model_parameters(self):
        if self.model is None:
            raise ValueError("No model selected.")
        return self.model.get_params()

    # ==================================================
    # Data
    # ==================================================

    def split_data(self, X, test_size=0.2, random_state=None):
        """
        Mostly used for semi-supervised experiments.
        """
        return train_test_split(
            X,
            test_size=test_size,
            random_state=(self.random_state if random_state is None else random_state),
        )

    # ==================================================
    # Scaling
    # ==================================================

    def preprocess(self, X_train, X_test=None, scaler="standard"):
        if scaler is None:
            return (X_train, X_test) if X_test is not None else X_train

        scaler = scaler.lower()

        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError("Scaler must be 'standard', 'minmax', 'robust' or None.")

        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_test is None:
            return X_train_scaled

        X_test_scaled = self.scaler.transform(X_test)
        return (X_train_scaled, X_test_scaled)

    def apply_scaler(self, X):
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted.")
        return self.scaler.transform(X)

    # ==================================================
    # Internal Helpers
    # ==================================================

    def _require_model(self):
        if self.model is None:
            raise ModelNotSelectedError()

    def _require_fitted(self):
        self._require_model()
        if not self.is_fitted:
            raise ClusterNotFittedError()

    # ==================================================
    # Utilities
    # ==================================================

    def framework(self):
        return {
            "Framework": "PsyInsight AI",
            "Module": "Machine Learning",
            "Component": "PsyClustering",
            "Version": __version__,
            "Supported Models": len(self.available_models()),
        }

    def version(self):
        return __version__

    def reset(self):
        self.model = None
        self.model_name = None
        self.model_key = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.scaler = None
        self.is_fitted = False
        return self

    clear = reset

    # ==================================================
    # Training / Prediction
    # ==================================================

    def fit(self, X):
        """
        Fit the selected clustering model.
        """
        self._require_model()

        start = time.perf_counter()

        if self.model_key == "gmm":
            self.model.fit(X)
            self.labels_ = self.model.predict(X)
            self.cluster_centers_ = self.model.means_
        else:
            self.labels_ = self.model.fit_predict(X)
            if hasattr(self.model, "cluster_centers_"):
                self.cluster_centers_ = self.model.cluster_centers_
            else:
                self.cluster_centers_ = None

        self.is_fitted = True

        elapsed = time.perf_counter() - start
        logger.info(
            f"Fitted {self.model_name} on {len(np.asarray(X))} samples in "
            f"{elapsed:.4f}s — {self.number_of_clusters()} cluster(s) found."
        )

        return self

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.
        """
        self.fit(X)
        return self.labels_

    def predict(self, X):
        """
        Predict cluster labels for new samples.
        """
        self._require_fitted()
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise AttributeError(f"{self.model_name} does not support predict().")

    def transform(self, X):
        """
        Distance / similarity transformation.
        """
        self._require_fitted()
        if hasattr(self.model, "transform"):
            return self.model.transform(X)
        raise AttributeError(f"{self.model_name} does not support transform().")

    # ==================================================
    # Cluster Information
    # ==================================================

    def cluster_labels(self):
        self._require_fitted()
        return self.labels_

    def number_of_clusters(self):
        self._require_fitted()
        labels = np.asarray(self.labels_)
        unique = np.unique(labels)
        if -1 in unique:
            unique = unique[unique != -1]
        return len(unique)

    def cluster_centers(self):
        self._require_fitted()
        return self.cluster_centers_

    def cluster_sizes(self):
        self._require_fitted()
        labels = np.asarray(self.labels_)
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    # ==================================================
    # Evaluation
    # ==================================================

    def evaluate(self, X):
        """
        Evaluate clustering performance.
        """
        self._require_fitted()

        labels = np.asarray(self.labels_)
        valid = labels != -1

        metrics = {}

        X_valid = np.asarray(X)[valid]
        labels_valid = labels[valid]

        if len(np.unique(labels_valid)) > 1:
            metrics["Silhouette Score"] = silhouette_score(X_valid, labels_valid)
            metrics["Davies Bouldin Score"] = davies_bouldin_score(X_valid, labels_valid)
            metrics["Calinski Harabasz Score"] = calinski_harabasz_score(X_valid, labels_valid)
        else:
            metrics["Silhouette Score"] = np.nan
            metrics["Davies Bouldin Score"] = np.nan
            metrics["Calinski Harabasz Score"] = np.nan

        metrics["Inertia"] = getattr(self.model, "inertia_", np.nan)

        return metrics

    def print_evaluation(self, X):
        """
        Pretty-print clustering metrics.
        """
        metrics = self.evaluate(X)

        print("\n===========================")
        print("CLUSTERING EVALUATION")
        print("===========================\n")

        for key, value in metrics.items():
            if isinstance(value, float) and np.isnan(value):
                print(f"{key:<30} N/A")
            else:
                print(f"{key:<30} {value:.4f}")

    # ==================================================
    # Cluster Summary
    # ==================================================

    def cluster_summary(self):
        """
        Return a summary DataFrame describing each discovered cluster.
        """
        self._require_fitted()

        labels = np.asarray(self.labels_)
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        rows = []

        for label, count in zip(unique, counts):
            row = {
                "Cluster": int(label),
                "Samples": int(count),
                "Percentage": round(count * 100 / total, 2),
            }

            if (
                self.cluster_centers_ is not None
                and label >= 0
                and label < len(self.cluster_centers_)
            ):
                row["Centroid"] = self.cluster_centers_[label]
            else:
                row["Centroid"] = None

            rows.append(row)

        return pd.DataFrame(rows)

    # ==================================================
    # Persistence
    # ==================================================

    def save_model(self, filename):
        """
        Save clustering model.
        """
        self._require_model()

        payload = {
            "model": self.model,
            "model_name": self.model_name,
            "model_key": self.model_key,
            "labels": self.labels_,
            "cluster_centers": self.cluster_centers_,
            "scaler": self.scaler,
        }

        joblib.dump(payload, filename)

    def load_model(self, filename):
        """
        Load clustering model.
        """
        payload = joblib.load(filename)

        self.model = payload["model"]
        self.model_name = payload["model_name"]
        self.model_key = payload["model_key"]
        self.labels_ = payload.get("labels")
        self.cluster_centers_ = payload.get("cluster_centers")
        self.scaler = payload.get("scaler")
        self.is_fitted = True

        return self


# =============================================================================
# AutoClusterMixin
# =============================================================================

class AutoClusterMixin:
    """
    Automatic clustering comparison and model selection.

    Provides:
        • compare_clusters()
        • leaderboard()
        • best_clustering()
        • auto_cluster()

    This mixin assumes PsyClusteringCore is inherited.
    """

    # Metrics where a LOWER value is better.
    _LOWER_IS_BETTER = {"Davies-Bouldin Score"}

    def _resolve_results(self, results=None):
        if results is not None:
            return results

        if hasattr(self, "_last_results"):
            return self._last_results

        raise ValueError("No clustering results available.\nRun compare_clusters() first.")

    def _cluster_metrics(self, X):
        """
        Compute clustering metrics for the currently fitted model.
        """
        labels = np.asarray(self.labels_)
        valid = labels != -1

        X_valid = np.asarray(X)[valid]
        labels_valid = labels[valid]

        metrics = {}

        if len(np.unique(labels_valid)) > 1:
            metrics["Silhouette Score"] = silhouette_score(X_valid, labels_valid)
            metrics["Davies-Bouldin Score"] = davies_bouldin_score(X_valid, labels_valid)
            metrics["Calinski-Harabasz Score"] = calinski_harabasz_score(X_valid, labels_valid)
        else:
            metrics["Silhouette Score"] = np.nan
            metrics["Davies-Bouldin Score"] = np.nan
            metrics["Calinski-Harabasz Score"] = np.nan

        metrics["Inertia"] = getattr(self.model, "inertia_", np.nan)
        metrics["Clusters"] = self.number_of_clusters()

        return metrics

    def compare_clusters(self, X, algorithms: Optional[List[str]] = None, scale: bool = True):
        """
        Compare every clustering algorithm.

        Parameters
        ----------
        X : ndarray
        algorithms : list, optional subset of algorithms.
        scale : bool, apply preprocessing before clustering.

        Returns
        -------
        pandas.DataFrame
        """
        if algorithms is None:
            algorithms = [
                "kmeans",
                "minibatch_kmeans",
                "agglomerative",
                "birch",
                "gmm",
            ]

        X_use = self.preprocess(X) if scale else X

        rows = []
        self._candidate_models = {}

        for algorithm in algorithms:
            self.set_model(algorithm)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fit(X_use)

            metrics = self._cluster_metrics(X_use)

            row = {
                "Algorithm": self.model_name,
                "_key": self.model_key,
                "Clusters": metrics["Clusters"],
                "Silhouette Score": metrics["Silhouette Score"],
                "Davies-Bouldin Score": metrics["Davies-Bouldin Score"],
                "Calinski-Harabasz Score": metrics["Calinski-Harabasz Score"],
                "Inertia": metrics["Inertia"],
            }

            rows.append(row)

            # Store the fitted estimator itself (not a clone) so predict()
            # and transform() keep working once this candidate is selected
            # by best_clustering(). Each loop iteration calls set_model(),
            # which rebuilds self.model as a brand-new object, so there is
            # no aliasing risk between candidates.
            self._candidate_models[self.model_key] = {
                "model": self.model,
                "labels": np.copy(self.labels_),
                "centers": (
                    None if self.cluster_centers_ is None else np.copy(self.cluster_centers_)
                ),
            }

        results = pd.DataFrame(rows)

        if len(results) == 0:
            raise ValueError("No clustering algorithms were evaluated.")

        results = results.sort_values(
            by="Silhouette Score", ascending=False, na_position="last"
        ).reset_index(drop=True)

        self._last_results = results
        return results

    # ==================================================
    # Leaderboard
    # ==================================================

    def leaderboard(self, results: Optional[pd.DataFrame] = None, metric: str = "Silhouette Score"):
        """
        Return (and print) the clustering leaderboard, ranked by `metric`.
        """
        results = self._resolve_results(results)
        ascending = metric in self._LOWER_IS_BETTER

        ranked = results.sort_values(
            by=metric, ascending=ascending, na_position="last"
        ).reset_index(drop=True)

        print("\n==============================")
        print(" CLUSTERING LEADERBOARD")
        print("==============================\n")

        for i, row in ranked.iterrows():
            score = row[metric]
            score_text = "N/A" if pd.isna(score) else f"{score:.4f}"
            print(f"{i + 1}. {row['Algorithm']:<25}{score_text}")

        return ranked

    # ==================================================
    # Best Cluster Selection
    # ==================================================

    @staticmethod
    def _best_cluster_reason(ranked):
        """
        Generate human-readable reasons explaining why the
        best clustering algorithm was selected.
        """
        reasons = []
        top = ranked.iloc[0]

        if pd.notna(top["Silhouette Score"]):
            reasons.append(
                f"Highest Silhouette Score ({top['Silhouette Score']:.4f}), "
                "indicating well-separated clusters."
            )

        if pd.notna(top["Davies-Bouldin Score"]):
            if top["Davies-Bouldin Score"] == ranked["Davies-Bouldin Score"].min():
                reasons.append(
                    f"Lowest Davies-Bouldin Score ({top['Davies-Bouldin Score']:.4f}), "
                    "indicating compact clusters."
                )

        if pd.notna(top["Calinski-Harabasz Score"]):
            if top["Calinski-Harabasz Score"] == ranked["Calinski-Harabasz Score"].max():
                reasons.append(
                    f"Highest Calinski-Harabasz Score ({top['Calinski-Harabasz Score']:.2f})."
                )

        reasons.append(f"Detected {int(top['Clusters'])} cluster(s).")

        return reasons

    def best_clustering(self, results: Optional[pd.DataFrame] = None, metric: str = "Silhouette Score"):
        """
        Select the best clustering algorithm based on the given metric,
        and load it as the active model.

        Returns
        -------
        (algorithm_key, best_row)
        """
        results = self._resolve_results(results)
        ascending = metric in self._LOWER_IS_BETTER

        ranked = results.sort_values(
            by=metric, ascending=ascending, na_position="last"
        ).reset_index(drop=True)

        best = ranked.iloc[0]
        key = best["_key"]

        candidate = self._candidate_models[key]

        self.model = candidate["model"]
        self.model_name = self.model_label(key)
        self.model_key = key
        self.labels_ = candidate["labels"]
        self.cluster_centers_ = candidate["centers"]
        self.is_fitted = True

        return key, best

    # ==================================================
    # Automatic Clustering Pipeline
    # ==================================================

    def auto_cluster(
        self,
        X,
        algorithms: Optional[List[str]] = None,
        metric: str = "Silhouette Score",
        scale: bool = True,
        verbose: bool = True,
    ):
        """
        Automatically compare all clustering algorithms, select the best
        one, and load it as the active model.

        Returns
        -------
        dict
        """
        results = self.compare_clusters(X, algorithms=algorithms, scale=scale)

        ascending = metric in self._LOWER_IS_BETTER
        ranked = results.sort_values(
            by=metric, ascending=ascending, na_position="last"
        ).reset_index(drop=True)

        best_key, best_row = self.best_clustering(ranked, metric=metric)
        reasons = self._best_cluster_reason(ranked)

        if verbose:
            self.leaderboard(ranked, metric=metric)

            print("\n==============================")
            print("BEST CLUSTERING ALGORITHM")
            print("==============================\n")

            print(best_row["Algorithm"])
            print("\nReason(s)\n")

            for reason in reasons:
                print(f"• {reason}")

            print("\nRecommended for this dataset.\n")

        return {
            "results": ranked,
            "best_key": best_key,
            "best_row": best_row,
            "reasons": reasons,
        }

    # ==================================================
    # Standalone hyper-parameter optimizers
    # ==================================================

    def optimize_kmeans(self, X, k_range=range(2, 11), random_state=None):
        """
        Automatically determine the best number of clusters using Silhouette Score.
        """
        best_model = None
        best_labels = None
        best_score = -1
        best_k = None

        random_state = random_state if random_state is not None else self.random_state

        results = []

        for k in k_range:
            model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            labels = model.fit_predict(X)

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(X, labels)

            results.append({"K": k, "Silhouette": score, "Inertia": model.inertia_})

            if score > best_score:
                best_score = score
                best_model = model
                best_labels = labels
                best_k = k

        return {
            "Best K": best_k,
            "Best Score": best_score,
            "Best Model": best_model,
            "Labels": best_labels,
            "Results": pd.DataFrame(results),
        }

    def optimize_gmm(self, X, components=range(2, 11)):
        """
        Automatically determine the best Gaussian Mixture using BIC.
        """
        best_model = None
        best_labels = None
        best_bic = np.inf

        rows = []

        for n in components:
            model = GaussianMixture(n_components=n, random_state=self.random_state)
            model.fit(X)
            labels = model.predict(X)

            bic = model.bic(X)
            aic = model.aic(X)

            rows.append({"Components": n, "BIC": bic, "AIC": aic})

            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_labels = labels

        return {
            "Best Components": best_model.n_components,
            "Best BIC": best_bic,
            "Best Model": best_model,
            "Labels": best_labels,
            "Results": pd.DataFrame(rows),
        }

    def optimize_agglomerative(self, X, k_range=range(2, 11), linkage_methods=None):
        """
        Optimize Agglomerative Clustering by testing different linkage strategies.
        """
        if linkage_methods is None:
            linkage_methods = ["ward", "complete", "average", "single"]

        best_model = None
        best_labels = None
        best_score = -1
        best_linkage = None
        best_k = None

        rows = []

        for linkage in linkage_methods:
            for k in k_range:
                model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels = model.fit_predict(X)

                if len(np.unique(labels)) < 2:
                    continue

                score = silhouette_score(X, labels)

                rows.append({"Clusters": k, "Linkage": linkage, "Silhouette": score})

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_labels = labels
                    best_linkage = linkage
                    best_k = k

        return {
            "Best Linkage": best_linkage,
            "Best K": best_k,
            "Best Score": best_score,
            "Best Model": best_model,
            "Labels": best_labels,
            "Results": pd.DataFrame(rows),
        }

    def optimize_dbscan(self, X, eps_values=None, min_samples_values=None):
        """
        Automatically optimize DBSCAN using Silhouette Score.
        """
        if eps_values is None:
            eps_values = np.arange(0.1, 2.1, 0.1)

        if min_samples_values is None:
            min_samples_values = range(3, 11)

        best_model = None
        best_labels = None
        best_score = -1
        best_eps = None
        best_min_samples = None

        rows = []

        X = np.asarray(X)

        for eps in eps_values:
            for min_samples in min_samples_values:
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X)

                unique = np.unique(labels)

                if len(unique) <= 1:
                    continue
                if len(unique) == 1 and -1 in unique:
                    continue

                mask = labels != -1
                if np.sum(mask) < 2:
                    continue

                try:
                    score = silhouette_score(X[mask], labels[mask])
                except Exception:
                    continue

                rows.append({
                    "eps": eps,
                    "min_samples": min_samples,
                    "Silhouette": score,
                    "Clusters": len(unique) - (1 if -1 in unique else 0),
                })

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_labels = labels
                    best_eps = eps
                    best_min_samples = min_samples

        return {
            "Best eps": best_eps,
            "Best min_samples": best_min_samples,
            "Best Score": best_score,
            "Best Model": best_model,
            "Labels": best_labels,
            "Results": pd.DataFrame(rows),
        }

    def optimize_optics(self, X, min_samples_values=None, xi_values=None):
        """
        Optimize OPTICS clustering.
        """
        if min_samples_values is None:
            min_samples_values = range(3, 11)

        if xi_values is None:
            xi_values = [0.01, 0.03, 0.05, 0.1]

        best_model = None
        best_labels = None
        best_score = -1
        best_min_samples = None
        best_xi = None

        rows = []

        X = np.asarray(X)

        for min_samples in min_samples_values:
            for xi in xi_values:
                model = OPTICS(min_samples=min_samples, xi=xi)
                labels = model.fit_predict(X)

                unique = np.unique(labels)

                if len(unique) <= 1:
                    continue

                mask = labels != -1
                if np.sum(mask) < 2:
                    continue

                try:
                    score = silhouette_score(X[mask], labels[mask])
                except Exception:
                    continue

                rows.append({"min_samples": min_samples, "xi": xi, "Silhouette": score})

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_labels = labels
                    best_min_samples = min_samples
                    best_xi = xi

        return {
            "Best min_samples": best_min_samples,
            "Best xi": best_xi,
            "Best Score": best_score,
            "Best Model": best_model,
            "Labels": best_labels,
            "Results": pd.DataFrame(rows),
        }

    def optimize_birch(self, X, thresholds=None, k_range=range(2, 11)):
        """
        Optimize Birch clustering.
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.1, 0.1)

        best_model = None
        best_labels = None
        best_score = -1
        best_threshold = None
        best_clusters = None

        rows = []

        for threshold in thresholds:
            for k in k_range:
                model = Birch(threshold=threshold, n_clusters=k)
                labels = model.fit_predict(X)

                if len(np.unique(labels)) < 2:
                    continue

                score = silhouette_score(X, labels)

                rows.append({"Threshold": threshold, "Clusters": k, "Silhouette": score})

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_labels = labels
                    best_threshold = threshold
                    best_clusters = k

        return {
            "Best Threshold": best_threshold,
            "Best Clusters": best_clusters,
            "Best Score": best_score,
            "Best Model": best_model,
            "Labels": best_labels,
            "Results": pd.DataFrame(rows),
        }

    # ==================================================
    # Convenience Helpers
    # ==================================================

    def ranking(self, metric="Silhouette Score"):
        """
        Return leaderboard sorted by metric (without printing).
        """
        results = self._resolve_results()
        ascending = metric in self._LOWER_IS_BETTER
        return results.sort_values(by=metric, ascending=ascending, na_position="last").reset_index(drop=True)

    def best_algorithm_name(self, metric="Silhouette Score"):
        """
        Return only the name of the best algorithm.
        """
        return self.ranking(metric).iloc[0]["Algorithm"]

    def best_algorithm_key(self, metric="Silhouette Score"):
        """
        Return the internal registry key of the best algorithm.
        """
        return self.ranking(metric).iloc[0]["_key"]

    def best_metrics(self, metric="Silhouette Score"):
        """
        Return metrics row of the best algorithm.
        """
        return self.ranking(metric).iloc[0].to_dict()

    def comparison_summary(self):
        """
        Return the previous comparison DataFrame.
        """
        return self._resolve_results().copy()

    def print_summary(self):
        """
        Nicely print the AutoCluster comparison summary.
        """
        results = self._resolve_results()

        print("\n==============================")
        print("AUTO CLUSTER SUMMARY")
        print("==============================\n")

        print(results)
        print()

    def export_leaderboard(self, filename="cluster_leaderboard.csv"):
        """
        Export the comparison leaderboard to CSV.
        """
        results = self._resolve_results()
        results.to_csv(filename, index=False)
        return filename

    def comparison_metrics(self):
        """
        Metrics available for ranking.
        """
        return [
            "Silhouette Score",
            "Davies-Bouldin Score",
            "Calinski-Harabasz Score",
            "Inertia",
            "Clusters",
        ]


# =============================================================================
# VisualizationMixin
# =============================================================================

class VisualizationMixin:
    """
    Visualization utilities for clustering.

    Generates publication-quality figures for PsyInsight AI.
    Assumes PsyClusteringCore is inherited.
    """

    def _fig_path(self, save_dir, filename):
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, filename)

    def _as_dataframe(self, X, feature_names=None):
        """
        Coerce X into a DataFrame with sensible column names,
        used by the profile-style plots below.
        """
        X = np.asarray(X)
        if feature_names is None:
            feature_names = [f"Feature {i + 1}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=feature_names)

    # ==================================================
    # Core plots (Part 1A)
    # ==================================================

    def plot_clusters(self, X, labels=None, save_dir="psy_plots", feature_names=None):
        """
        2D Scatter Plot (PCA-reduced if more than 2 features).
        """
        self._require_fitted()

        if labels is None:
            labels = self.labels_

        X = np.asarray(X)

        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=self.random_state)
            X_plot = pca.fit_transform(X)
            xlabel = "Principal Component 1"
            ylabel = "Principal Component 2"
        else:
            X_plot = X
            xlabel = feature_names[0] if feature_names else "Feature 1"
            ylabel = feature_names[1] if feature_names else "Feature 2"

        fig, ax = plt.subplots(figsize=(7, 6))

        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap="tab10", s=40, edgecolor="black")

        ax.set_title(f"Clusters — {self.model_name}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.colorbar(scatter, ax=ax)

        path = self._fig_path(save_dir, "clusters.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_cluster_distribution(self, labels=None, save_dir="psy_plots"):
        """
        Cluster size distribution.
        """
        self._require_fitted()

        if labels is None:
            labels = self.labels_

        values = pd.Series(labels).value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(6, 5))
        values.plot.bar(ax=ax)

        ax.set_xlabel("Cluster")
        ax.set_ylabel("Samples")
        ax.set_title("Cluster Distribution")

        path = self._fig_path(save_dir, "cluster_distribution.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_centroids(self, feature_names=None, save_dir="psy_plots"):
        """
        Centroid heatmap.
        """
        self._require_fitted()

        if not hasattr(self.model, "cluster_centers_"):
            return None

        centers = self.model.cluster_centers_

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(centers, aspect="auto", cmap="viridis")

        plt.colorbar(im, ax=ax)

        ax.set_ylabel("Cluster")
        ax.set_xlabel("Features")
        ax.set_title("Centroid Heatmap")

        if feature_names is not None:
            ax.set_xticks(np.arange(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha="right")

        path = self._fig_path(save_dir, "centroid_heatmap.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_clusters_3d(self, X, labels=None, save_dir="psy_plots"):
        """
        3D cluster visualization. Uses PCA if features > 3.
        """
        self._require_fitted()

        if labels is None:
            labels = self.labels_

        X = np.asarray(X)

        if X.shape[1] > 3:
            pca = PCA(n_components=3, random_state=self.random_state)
            X_plot = pca.fit_transform(X)
        else:
            X_plot = X

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], c=labels, cmap="tab10", s=40)

        plt.colorbar(scatter, ax=ax)
        ax.set_title(f"3D Clusters — {self.model_name}")

        path = self._fig_path(save_dir, "clusters_3d.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_pca_projection(self, X, labels=None, save_dir="psy_plots"):
        """
        PCA projection.
        """
        self._require_fitted()

        if labels is None:
            labels = self.labels_

        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=40)

        plt.colorbar(scatter, ax=ax)
        ax.set_title("PCA Projection")

        path = self._fig_path(save_dir, "pca_projection.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_tsne_projection(self, X, labels=None, save_dir="psy_plots"):
        """
        t-SNE visualization.
        """
        self._require_fitted()

        from sklearn.manifold import TSNE

        if labels is None:
            labels = self.labels_

        tsne = TSNE(n_components=2, random_state=self.random_state)
        X_tsne = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="tab10", s=40)

        plt.colorbar(scatter, ax=ax)
        ax.set_title("t-SNE Projection")

        path = self._fig_path(save_dir, "tsne_projection.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_umap_projection(self, X, labels=None, save_dir="psy_plots"):
        """
        UMAP projection. Requires: pip install umap-learn
        """
        self._require_fitted()

        try:
            import umap
        except ImportError:
            return None

        if labels is None:
            labels = self.labels_

        reducer = umap.UMAP(random_state=self.random_state)
        X_umap = reducer.fit_transform(X)

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap="tab10", s=40)

        plt.colorbar(scatter, ax=ax)
        ax.set_title("UMAP Projection")

        path = self._fig_path(save_dir, "umap_projection.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_elbow_method(self, X, k_range=range(2, 11), save_dir="psy_plots"):
        """
        Elbow method for selecting K.
        """
        inertias = []

        for k in k_range:
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            model.fit(X)
            inertias.append(model.inertia_)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(list(k_range), inertias, marker="o", linewidth=2)

        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")

        path = self._fig_path(save_dir, "elbow_method.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_silhouette(self, X, labels=None, save_dir="psy_plots"):
        """
        Silhouette plot.
        """
        self._require_fitted()

        if labels is None:
            labels = self.labels_

        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            return None

        sample_values = silhouette_samples(X, labels)

        fig, ax = plt.subplots(figsize=(7, 5))
        y_lower = 10

        for cluster in unique_labels:
            vals = sample_values[labels == cluster]
            vals.sort()

            size = len(vals)
            y_upper = y_lower + size

            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
            ax.text(-0.05, y_lower + size / 2, str(cluster))

            y_lower = y_upper + 10

        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster")
        ax.set_title("Silhouette Plot")

        path = self._fig_path(save_dir, "silhouette_plot.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_cluster_distance(self, save_dir="psy_plots"):
        """
        Pairwise centroid distance matrix.
        """
        self._require_fitted()

        if not hasattr(self.model, "cluster_centers_"):
            return None

        from scipy.spatial.distance import cdist

        centers = self.model.cluster_centers_
        dist = cdist(centers, centers)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(dist, cmap="magma")

        plt.colorbar(im, ax=ax)
        ax.set_title("Cluster Distance Matrix")

        path = self._fig_path(save_dir, "cluster_distance_matrix.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_dendrogram(self, X, method: str = "ward", save_dir: str = "psy_plots"):
        """
        Plot a hierarchical clustering dendrogram.
        """
        from scipy.cluster.hierarchy import linkage, dendrogram

        linkage_matrix = linkage(X, method=method)

        fig, ax = plt.subplots(figsize=(10, 6))

        dendrogram(linkage_matrix, ax=ax, leaf_rotation=90)

        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Distance")

        path = self._fig_path(save_dir, "dendrogram.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    # ==================================================
    # Advanced plots (Part 1B / Chunk B)
    # ==================================================

    def plot_pairplot(self, X, feature_names=None, max_features=5, save_dir="psy_plots"):
        """
        Pairwise scatter matrix colored by cluster label.
        Falls back to pandas.plotting.scatter_matrix if seaborn is
        unavailable. Limits to `max_features` columns to stay readable.
        """
        self._require_fitted()

        df = self._as_dataframe(X, feature_names)
        cols = list(df.columns[:max_features])
        df_plot = df[cols].copy()
        df_plot["Cluster"] = self.labels_

        try:
            import seaborn as sns

            grid = sns.pairplot(df_plot, hue="Cluster", palette="tab10", diag_kind="kde")
            grid.fig.suptitle(f"Pairplot — {self.model_name}", y=1.02)

            path = self._fig_path(save_dir, "pairplot.png")
            grid.savefig(path, dpi=150)
            plt.close(grid.fig)
            return path

        except ImportError:
            axes = pd.plotting.scatter_matrix(
                df_plot[cols],
                c=self.labels_,
                cmap="tab10",
                figsize=(2.2 * len(cols), 2.2 * len(cols)),
                diagonal="hist",
                s=30,
            )
            fig = axes[0, 0].get_figure()
            fig.suptitle(f"Pairplot — {self.model_name}")

            path = self._fig_path(save_dir, "pairplot.png")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return path

    def plot_feature_means(self, X, feature_names=None, save_dir="psy_plots"):
        """
        Grouped bar chart of per-cluster feature means.
        """
        self._require_fitted()

        df = self._as_dataframe(X, feature_names)
        df["Cluster"] = self.labels_
        means = df.groupby("Cluster").mean()

        fig, ax = plt.subplots(figsize=(max(7, len(means.columns) * 0.9), 6))
        means.T.plot(kind="bar", ax=ax, cmap="tab10")

        ax.set_title("Per-Cluster Feature Means")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Mean Value")
        ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")

        path = self._fig_path(save_dir, "feature_means.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_cluster_profiles(self, X, feature_names=None, save_dir="psy_plots"):
        """
        Line "profile" plot: one line per cluster across all features,
        useful for spotting which features drive each cluster.
        """
        self._require_fitted()

        df = self._as_dataframe(X, feature_names)
        df["Cluster"] = self.labels_
        profiles = df.groupby("Cluster").mean()

        fig, ax = plt.subplots(figsize=(max(7, len(profiles.columns) * 0.7), 6))

        for cluster, row in profiles.iterrows():
            ax.plot(profiles.columns, row.values, marker="o", label=f"Cluster {cluster}")

        ax.set_title("Cluster Profiles")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Mean Value")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Cluster")

        path = self._fig_path(save_dir, "cluster_profiles.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_parallel_coordinates(self, X, feature_names=None, save_dir="psy_plots"):
        """
        Parallel-coordinates plot colored by cluster.
        """
        self._require_fitted()

        from pandas.plotting import parallel_coordinates

        df = self._as_dataframe(X, feature_names)
        df["Cluster"] = self.labels_.astype(str)

        fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 0.9), 6))
        parallel_coordinates(df, "Cluster", colormap="tab10", ax=ax, alpha=0.6)

        ax.set_title("Parallel Coordinates")
        ax.tick_params(axis="x", rotation=45)

        path = self._fig_path(save_dir, "parallel_coordinates.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_radar_chart(self, X, feature_names=None, save_dir="psy_plots"):
        """
        Radar / spider chart of per-cluster mean feature profiles
        (features are min-max normalized for comparability).
        """
        self._require_fitted()

        df = self._as_dataframe(X, feature_names)
        df["Cluster"] = self.labels_
        means = df.groupby("Cluster").mean()

        normalized = (means - means.min()) / (means.max() - means.min() + 1e-12)

        categories = list(normalized.columns)
        n_cats = len(categories)
        angles = [i / float(n_cats) * 2 * np.pi for i in range(n_cats)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        for cluster, row in normalized.iterrows():
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=f"Cluster {cluster}")
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title("Cluster Radar Chart (normalized)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        path = self._fig_path(save_dir, "radar_chart.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_boxplots(self, X, feature_names=None, max_features=6, save_dir="psy_plots"):
        """
        Boxplot of each feature, grouped by cluster.
        """
        self._require_fitted()

        df = self._as_dataframe(X, feature_names)
        cols = list(df.columns[:max_features])
        df["Cluster"] = self.labels_

        n = len(cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for idx, col in enumerate(cols):
            ax = axes[idx // ncols][idx % ncols]
            df.boxplot(column=col, by="Cluster", ax=ax)
            ax.set_title(col)
            ax.set_xlabel("Cluster")

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig.suptitle("Feature Distributions by Cluster (Boxplots)")

        path = self._fig_path(save_dir, "boxplots.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_violin(self, X, feature_names=None, max_features=6, save_dir="psy_plots"):
        """
        Violin plot of each feature, grouped by cluster.
        """
        self._require_fitted()

        df = self._as_dataframe(X, feature_names)
        cols = list(df.columns[:max_features])
        df["Cluster"] = self.labels_

        n = len(cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        clusters = sorted(df["Cluster"].unique())

        for idx, col in enumerate(cols):
            ax = axes[idx // ncols][idx % ncols]
            data = [df.loc[df["Cluster"] == c, col].values for c in clusters]
            ax.violinplot(data, showmeans=True)
            ax.set_xticks(range(1, len(clusters) + 1))
            ax.set_xticklabels(clusters)
            ax.set_title(col)
            ax.set_xlabel("Cluster")

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig.suptitle("Feature Distributions by Cluster (Violin)")

        path = self._fig_path(save_dir, "violin.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_density(self, X, save_dir="psy_plots"):
        """
        KDE density plot of the first principal component, split by cluster.
        Useful as a 1D overview of cluster separation.
        """
        self._require_fitted()

        X = np.asarray(X)

        if X.shape[1] > 1:
            pca = PCA(n_components=1, random_state=self.random_state)
            values = pca.fit_transform(X).ravel()
            xlabel = "Principal Component 1"
        else:
            values = X.ravel()
            xlabel = "Feature 1"

        fig, ax = plt.subplots(figsize=(7, 5))

        labels = np.asarray(self.labels_)
        for cluster in np.unique(labels):
            subset = values[labels == cluster]
            if len(subset) < 2:
                continue
            try:
                from scipy.stats import gaussian_kde

                kde = gaussian_kde(subset)
                grid = np.linspace(values.min(), values.max(), 200)
                ax.plot(grid, kde(grid), label=f"Cluster {cluster}")
                ax.fill_between(grid, kde(grid), alpha=0.15)
            except Exception:
                ax.hist(subset, bins=20, alpha=0.4, label=f"Cluster {cluster}", density=True)

        ax.set_title("Cluster Density (1D Projection)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()

        path = self._fig_path(save_dir, "density.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    def plot_heatmap(self, X, feature_names=None, save_dir="psy_plots"):
        """
        Feature correlation heatmap (independent of clustering result;
        useful for understanding what the clusters are built from).
        """
        df = self._as_dataframe(X, feature_names)
        corr = df.corr()

        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(corr.columns)), max(5, 0.6 * len(corr.columns))))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_yticklabels(corr.columns)

        plt.colorbar(im, ax=ax)
        ax.set_title("Feature Correlation Heatmap")

        path = self._fig_path(save_dir, "correlation_heatmap.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path

    # ==================================================
    # Batch generation
    # ==================================================

    def visualize_all(self, X, feature_names=None, include_advanced=True):
        """
        Generate the full standard set of visualizations for the
        currently fitted model in one call. Set `include_advanced=False`
        to skip the Chunk-B profile/statistical plots.
        """
        self.plot_clusters(X)
        self.plot_clusters_3d(X)
        self.plot_cluster_distribution()
        self.plot_centroids()
        self.plot_pca_projection(X)
        self.plot_tsne_projection(X)
        self.plot_elbow_method(X)
        self.plot_silhouette(X)

        try:
            self.plot_cluster_distance()
        except Exception:
            pass

        if include_advanced:
            feature_aware = (
                self.plot_feature_means,
                self.plot_cluster_profiles,
                self.plot_parallel_coordinates,
                self.plot_radar_chart,
                self.plot_boxplots,
                self.plot_violin,
                self.plot_heatmap,
            )
            for method in feature_aware:
                try:
                    method(X, feature_names=feature_names)
                except Exception:
                    pass

            try:
                self.plot_density(X)
            except Exception:
                pass

        return "All visualizations generated."


# =============================================================================
# ReportingMixin
# =============================================================================

class ReportingMixin:
    """
    Reporting, export, and project-persistence utilities.

    Assumes PsyClusteringCore is inherited.
    """

    def cluster_purity(self, y_true) -> float:
        """
        Compute cluster purity against known ground-truth labels.

        Noise points (label -1, e.g. from DBSCAN/OPTICS) are excluded
        from the calculation since they don't belong to any cluster.
        """
        self._require_fitted()

        labels = np.asarray(self.labels_)
        y_true = np.asarray(y_true)

        valid = labels != -1
        labels_valid = labels[valid]
        y_true_valid = y_true[valid]

        total = len(y_true_valid)

        if total == 0:
            return float("nan")

        purity = 0

        for cluster in np.unique(labels_valid):
            mask = labels_valid == cluster
            counts = np.bincount(y_true_valid[mask])
            purity += counts.max()

        return purity / total

    def prediction_summary(self, X) -> pd.DataFrame:
        """
        Summarize predicted cluster assignment counts for new samples.
        """
        preds = self.predict(X)
        unique, counts = np.unique(preds, return_counts=True)

        return pd.DataFrame({"Cluster": unique, "Samples": counts})

    def describe_model(self) -> dict:
        """
        Describe capabilities of the currently selected model.
        """
        self._require_model()

        return {
            "Model": self.model_name,
            "Clusters": self.number_of_clusters() if self.is_fitted else None,
            "Supports Prediction": hasattr(self.model, "predict"),
            "Supports Transform": hasattr(self.model, "transform"),
            "Supports Centroids": hasattr(self.model, "cluster_centers_"),
        }

    # ==================================================
    # Recommendations (used by generate_report / report_dict)
    # ==================================================

    def _recommendations(self, metrics: dict) -> List[str]:
        """
        Produce simple, human-readable recommendations based on the
        evaluation metrics of the currently fitted model.
        """
        recs = []

        n_clusters = self.number_of_clusters()
        sizes = self.cluster_sizes()
        total = sum(sizes.values())

        sil = metrics.get("Silhouette Score", np.nan)
        db = metrics.get("Davies Bouldin Score", metrics.get("Davies-Bouldin Score", np.nan))

        if pd.isna(sil):
            recs.append(
                "Silhouette Score could not be computed (fewer than 2 valid clusters). "
                "Consider adjusting the algorithm's parameters (e.g. eps/min_samples for "
                "density-based methods, or n_clusters for partition-based methods)."
            )
        elif sil < 0.25:
            recs.append(
                f"Silhouette Score is low ({sil:.3f}), suggesting overlapping or poorly "
                "separated clusters. Try rescaling features, reducing dimensionality "
                "(e.g. PCA), or comparing alternative algorithms with compare_clusters()."
            )
        elif sil < 0.5:
            recs.append(
                f"Silhouette Score is moderate ({sil:.3f}). Clusters are reasonably "
                "separated but may benefit from further tuning."
            )
        else:
            recs.append(
                f"Silhouette Score is strong ({sil:.3f}), indicating well-separated, "
                "cohesive clusters."
            )

        if not pd.isna(db):
            if db > 1.5:
                recs.append(
                    f"Davies-Bouldin Score is relatively high ({db:.3f}), indicating some "
                    "clusters may be too close together or not compact."
                )

        if -1 in sizes:
            noise_pct = sizes[-1] * 100 / total
            if noise_pct > 10:
                recs.append(
                    f"{noise_pct:.1f}% of samples were labeled as noise/outliers. "
                    "Consider loosening density parameters if this seems too aggressive."
                )

        smallest = min(sizes.values())
        if smallest * 100 / total < 3:
            recs.append(
                "At least one cluster contains fewer than 3% of all samples — verify "
                "this isn't an artifact of outliers or an unlucky initialization."
            )

        if n_clusters == 1:
            recs.append(
                "Only a single cluster was found. The data may not contain strong "
                "cluster structure, or parameters need adjustment."
            )

        return recs

    # ==================================================
    # Structured report data
    # ==================================================

    def report_dict(self, X, y_true=None) -> Dict[str, Any]:
        """
        Build a structured dictionary containing everything a report
        needs: metrics, cluster sizes, and recommendations. This is the
        single source of truth used by generate_report / markdown_report
        / html_report so all report formats stay consistent.
        """
        self._require_fitted()

        metrics = self.evaluate(X)
        sizes = self.cluster_sizes()

        payload = {
            "framework": "PsyInsight AI",
            "version": __version__,
            "algorithm": self.model_name,
            "n_clusters": self.number_of_clusters(),
            "n_samples": int(len(self.labels_)),
            "metrics": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()},
            "cluster_sizes": {str(k): v for k, v in sizes.items()},
            "recommendations": self._recommendations(metrics),
        }

        if y_true is not None:
            payload["cluster_purity"] = self.cluster_purity(y_true)

        return payload

    # ==================================================
    # Text report (Part 1A, kept for backwards compatibility)
    # ==================================================

    def generate_report(self, X, y_true=None, save_path: str = "cluster_report.txt") -> str:
        """
        Write a plain-text clustering report to disk and return its path.
        """
        self._require_fitted()

        metrics = self.evaluate(X)

        with open(save_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("PsyInsight AI\n")
            f.write("Cluster Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Algorithm : {self.model_name}\n")
            f.write(f"Clusters  : {self.number_of_clusters()}\n\n")

            for k, v in metrics.items():
                f.write(f"{k:30}{v}\n")

            if y_true is not None:
                f.write("\n")
                f.write(f"Cluster Purity : {self.cluster_purity(y_true):.4f}\n")

            f.write("\n")
            f.write("Cluster Summary\n")
            f.write("-------------------------\n")
            f.write(self.cluster_summary().to_string())
            f.write("\n\n")

            f.write("Recommendations\n")
            f.write("-------------------------\n")
            for rec in self._recommendations(metrics):
                f.write(f"- {rec}\n")

        return save_path

    def save_report(self, X, y_true=None, save_path: str = "cluster_report.txt") -> str:
        """
        Alias of generate_report(), named to match the rest of the
        export_* / save_* utility naming convention.
        """
        return self.generate_report(X, y_true=y_true, save_path=save_path)

    # ==================================================
    # Markdown / HTML reports (Chunk A)
    # ==================================================

    def markdown_report(self, X, y_true=None, save_path: str = "cluster_report.md") -> str:
        """
        Write a Markdown clustering report (nice for READMEs/notebooks).
        """
        data = self.report_dict(X, y_true=y_true)

        lines = []
        lines.append(f"# {data['framework']} — Cluster Report\n")
        lines.append(f"**Algorithm:** {data['algorithm']}  ")
        lines.append(f"**Clusters found:** {data['n_clusters']}  ")
        lines.append(f"**Samples:** {data['n_samples']}\n")

        lines.append("## Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in data["metrics"].items():
            v_text = "N/A" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
            lines.append(f"| {k} | {v_text} |")

        if "cluster_purity" in data:
            lines.append(f"| Cluster Purity | {data['cluster_purity']:.4f} |")
        lines.append("")

        lines.append("## Cluster Sizes\n")
        lines.append("| Cluster | Samples |")
        lines.append("|---|---|")
        for k, v in data["cluster_sizes"].items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

        lines.append("## Recommendations\n")
        for rec in data["recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")

        content = "\n".join(lines)

        with open(save_path, "w") as f:
            f.write(content)

        return save_path

    def html_report(self, X, y_true=None, save_path: str = "cluster_report.html") -> str:
        """
        Write a self-contained HTML clustering report.
        """
        data = self.report_dict(X, y_true=y_true)

        def esc(text):
            return str(text).replace("<", "&lt;").replace(">", "&gt;")

        metric_rows = ""
        for k, v in data["metrics"].items():
            v_text = "N/A" if v is None else (f"{v:.4f}" if isinstance(v, float) else esc(v))
            metric_rows += f"<tr><td>{esc(k)}</td><td>{v_text}</td></tr>\n"

        if "cluster_purity" in data:
            metric_rows += f"<tr><td>Cluster Purity</td><td>{data['cluster_purity']:.4f}</td></tr>\n"

        size_rows = "".join(
            f"<tr><td>{esc(k)}</td><td>{v}</td></tr>\n" for k, v in data["cluster_sizes"].items()
        )

        rec_items = "".join(f"<li>{esc(rec)}</li>\n" for rec in data["recommendations"])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{esc(data['framework'])} — Cluster Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 40px; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0; }}
  .meta {{ color: #555; margin-top: 4px; margin-bottom: 24px; }}
  table {{ border-collapse: collapse; margin-bottom: 24px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 14px; text-align: left; }}
  th {{ background: #f4f4f4; }}
  h2 {{ margin-top: 32px; }}
</style>
</head>
<body>
  <h1>{esc(data['framework'])}</h1>
  <div class="meta">Cluster Report &middot; v{esc(data['version'])}</div>

  <p><strong>Algorithm:</strong> {esc(data['algorithm'])}<br>
     <strong>Clusters found:</strong> {data['n_clusters']}<br>
     <strong>Samples:</strong> {data['n_samples']}</p>

  <h2>Metrics</h2>
  <table><tr><th>Metric</th><th>Value</th></tr>{metric_rows}</table>

  <h2>Cluster Sizes</h2>
  <table><tr><th>Cluster</th><th>Samples</th></tr>{size_rows}</table>

  <h2>Recommendations</h2>
  <ul>{rec_items}</ul>
</body>
</html>
"""

        with open(save_path, "w") as f:
            f.write(html)

        return save_path

    # ==================================================
    # Export helpers
    # ==================================================

    def export_clusters(self, X, filename: str = "clustered_data.csv") -> str:
        """
        Export the original data with an appended cluster-label column.
        """
        self._require_fitted()

        df = pd.DataFrame(X)
        df["Cluster"] = self.labels_
        df.to_csv(filename, index=False)

        return filename

    def export_summary(self, filename: str = "cluster_summary.csv") -> str:
        """
        Export cluster_summary() (sizes/percentages/centroids) to CSV.
        """
        summary = self.cluster_summary()
        summary.to_csv(filename, index=False)
        return filename

    # ==================================================
    # Human-readable summary
    # ==================================================

    def summary(self, X=None):
        """
        Print a human-readable summary of the current model.

        Pass `X` to also include evaluation metrics for the fitted data.
        """
        self._require_fitted()

        print("=" * 60)
        print("PsyInsight AI")
        print("Clustering Framework")
        print("=" * 60)
        print()

        print("Model    :", self.model_name)
        print("Clusters :", self.number_of_clusters())
        print()

        print(self.cluster_summary())
        print()

        if X is not None:
            print("Evaluation Metrics")
            print("-------------------------")
            for k, v in self.evaluate(X).items():
                if isinstance(v, float) and np.isnan(v):
                    print(f"{k:<30} N/A")
                else:
                    print(f"{k:<30} {v}")

    # ==================================================
    # Project persistence
    # ==================================================

    def save_project(self, X, folder: str = "psy_project") -> str:
        """
        Persist the fitted model, clustered data, and a text report
        together in one project folder.
        """
        self._require_fitted()

        os.makedirs(folder, exist_ok=True)

        self.save_model(os.path.join(folder, "model.pkl"))
        self.export_clusters(X, os.path.join(folder, "clusters.csv"))
        self.generate_report(X, save_path=os.path.join(folder, "report.txt"))

        return folder


# =============================================================================
# ExternalMetricsMixin
# =============================================================================

class ExternalMetricsMixin:
    """
    External-validation metrics: everything that requires ground-truth
    labels (y_true) in addition to the discovered cluster labels.

    Assumes PsyClusteringCore is inherited (uses self.labels_).
    """

    def _external_labels(self, y_true):
        """
        Align predicted / true labels, dropping noise points (-1) so
        external metrics aren't distorted by unassigned samples.
        """
        self._require_fitted()

        labels = np.asarray(self.labels_)
        y_true = np.asarray(y_true)

        if len(labels) != len(y_true):
            raise ValueError("y_true must be the same length as the fitted data.")

        valid = labels != -1
        return labels[valid], y_true[valid]

    # ==================================================
    # Individual metrics
    # ==================================================

    def entropy(self, y_true) -> float:
        """
        Weighted average entropy of the true-label distribution within
        each discovered cluster. Lower is better (0 = perfectly pure).
        """
        labels, y_true = self._external_labels(y_true)
        total = len(labels)

        if total == 0:
            return float("nan")

        weighted_entropy = 0.0

        for cluster in np.unique(labels):
            mask = labels == cluster
            counts = np.bincount(y_true[mask])
            probs = counts[counts > 0] / counts.sum()
            cluster_entropy = -np.sum(probs * np.log2(probs))
            weighted_entropy += (mask.sum() / total) * cluster_entropy

        return float(weighted_entropy)

    def homogeneity(self, y_true) -> float:
        labels, y_true = self._external_labels(y_true)
        return float(homogeneity_score(y_true, labels))

    def completeness(self, y_true) -> float:
        labels, y_true = self._external_labels(y_true)
        return float(completeness_score(y_true, labels))

    def v_measure(self, y_true, beta: float = 1.0) -> float:
        labels, y_true = self._external_labels(y_true)
        return float(v_measure_score(y_true, labels, beta=beta))

    def homogeneity_completeness_vmeasure(self, y_true) -> Dict[str, float]:
        labels, y_true = self._external_labels(y_true)
        h, c, v = homogeneity_completeness_v_measure(y_true, labels)
        return {"Homogeneity": float(h), "Completeness": float(c), "V-Measure": float(v)}

    def adjusted_rand_index(self, y_true) -> float:
        """
        Adjusted Rand Index (ARI). Ranges roughly [-1, 1]; 1 is a
        perfect match, ~0 is random labeling.
        """
        labels, y_true = self._external_labels(y_true)
        return float(adjusted_rand_score(y_true, labels))

    def normalized_mutual_information(self, y_true, average_method: str = "arithmetic") -> float:
        """
        Normalized Mutual Information (NMI). Ranges [0, 1].
        """
        labels, y_true = self._external_labels(y_true)
        return float(normalized_mutual_info_score(y_true, labels, average_method=average_method))

    def fowlkes_mallows(self, y_true) -> float:
        """
        Fowlkes-Mallows Index: geometric mean of pairwise precision
        and recall. Ranges [0, 1].
        """
        labels, y_true = self._external_labels(y_true)
        return float(fowlkes_mallows_score(y_true, labels))

    def pair_confusion(self, y_true) -> np.ndarray:
        """
        2x2 pair-confusion matrix (see sklearn.metrics.pair_confusion_matrix):
        counts of sample pairs agreeing/disagreeing between the true
        and predicted clusterings.
        """
        labels, y_true = self._external_labels(y_true)
        return pair_confusion_matrix(y_true, labels)

    def contingency(self, y_true) -> pd.DataFrame:
        """
        Contingency table: rows = true classes, columns = discovered
        clusters, cells = sample counts.
        """
        labels, y_true = self._external_labels(y_true)
        matrix = sk_confusion_matrix(y_true, labels)

        true_names = [f"True {c}" for c in np.unique(y_true)]
        cluster_names = [f"Cluster {c}" for c in np.unique(labels)]

        return pd.DataFrame(matrix, index=true_names, columns=cluster_names)

    # ==================================================
    # Combined evaluation
    # ==================================================

    def evaluate_external(self, y_true) -> Dict[str, float]:
        """
        Compute every external-validation metric in one call.
        """
        hcv = self.homogeneity_completeness_vmeasure(y_true)

        metrics = {
            "Cluster Purity": self.cluster_purity(y_true) if hasattr(self, "cluster_purity") else None,
            "Entropy": self.entropy(y_true),
            "Homogeneity": hcv["Homogeneity"],
            "Completeness": hcv["Completeness"],
            "V-Measure": hcv["V-Measure"],
            "Adjusted Rand Index": self.adjusted_rand_index(y_true),
            "Normalized Mutual Information": self.normalized_mutual_information(y_true),
            "Fowlkes-Mallows Index": self.fowlkes_mallows(y_true),
        }

        return metrics

    def print_external_evaluation(self, y_true):
        """
        Pretty-print the external-validation metrics.
        """
        metrics = self.evaluate_external(y_true)

        print("\n===========================")
        print("EXTERNAL VALIDATION METRICS")
        print("===========================\n")

        for key, value in metrics.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                print(f"{key:<32} N/A")
            else:
                print(f"{key:<32} {value:.4f}")


# =============================================================================
# UtilityMixin
# =============================================================================

class UtilityMixin:
    """
    Production utilities: cloning, parameter access, introspection,
    batch operations, benchmarking, and JSON (de)serialization.

    Assumes PsyClusteringCore (+ ClusterRegistryMixin) is inherited.
    """

    # ==================================================
    # Cloning / copying
    # ==================================================

    def clone_model(self):
        """
        Return an unfitted clone of the current estimator (same
        hyperparameters, fresh internal state). Useful for re-running
        on new data without disturbing the current fitted model.
        """
        self._require_model()
        from sklearn.base import clone

        return clone(self.model)

    def copy(self):
        """
        Return a deep copy of this entire PsyClustering instance,
        including fitted state, labels, and scaler.
        """
        return deepcopy(self)

    # ==================================================
    # Parameters
    # ==================================================

    def get_params(self) -> dict:
        """
        Return the current estimator's hyperparameters.
        """
        self._require_model()
        return self.model.get_params()

    def set_params(self, **params):
        """
        Update the current estimator's hyperparameters in place.
        Marks the model as unfitted since parameters changed.
        """
        self._require_model()
        self.model.set_params(**params)
        self.is_fitted = False
        return self

    # ==================================================
    # Introspection
    # ==================================================

    def describe(self) -> dict:
        """
        High-level description of the current framework state.
        """
        info = {
            "framework": "PsyInsight AI",
            "version": __version__,
            "random_state": self.random_state,
            "model": self.model_name,
            "model_key": self.model_key,
            "is_fitted": self.is_fitted,
            "scaler": type(self.scaler).__name__ if self.scaler is not None else None,
        }
        if self.is_fitted:
            info["n_clusters"] = self.number_of_clusters()
            info["n_samples"] = int(len(self.labels_))
        return info

    def explain(self) -> str:
        """
        Plain-English explanation of the currently selected algorithm,
        its strengths, and typical use cases.
        """
        explanations = {
            "kmeans": (
                "K-Means partitions data into a fixed number (k) of spherical "
                "clusters by minimizing within-cluster variance. Fast and scalable, "
                "but assumes roughly equal-sized, convex clusters and requires k "
                "to be chosen in advance."
            ),
            "minibatch_kmeans": (
                "MiniBatch K-Means is a faster, approximate variant of K-Means that "
                "updates centroids using small random batches — well suited to "
                "large datasets where full K-Means is too slow."
            ),
            "dbscan": (
                "DBSCAN groups points that are densely packed together, marking "
                "points in low-density regions as noise (-1). It doesn't require "
                "specifying the number of clusters and can find arbitrarily shaped "
                "clusters, but is sensitive to the eps/min_samples parameters."
            ),
            "agglomerative": (
                "Agglomerative Clustering builds a hierarchy by iteratively merging "
                "the closest clusters. Produces a dendrogram-friendly structure "
                "and can use different linkage criteria (ward, average, etc.)."
            ),
            "spectral": (
                "Spectral Clustering uses the eigenvectors of a similarity graph "
                "to reduce dimensionality before clustering, which lets it capture "
                "non-convex cluster shapes at the cost of higher computational cost."
            ),
            "birch": (
                "Birch incrementally builds a compact tree summary of the data, "
                "making it memory-efficient for very large datasets."
            ),
            "optics": (
                "OPTICS is a density-based method similar to DBSCAN but capable of "
                "finding clusters of varying density by ordering points based on "
                "reachability distance."
            ),
            "gmm": (
                "Gaussian Mixture Models assume the data is generated from a "
                "mixture of Gaussian distributions and assign soft (probabilistic) "
                "cluster memberships rather than hard labels."
            ),
        }

        if self.model_key is None:
            return "No model selected. Call set_model() first."

        return explanations.get(self.model_key, f"No description available for '{self.model_key}'.")

    def algorithm_details(self, key: Optional[str] = None) -> dict:
        """
        Return registry metadata (label + explanation) for a given
        algorithm key, or the currently selected one if omitted.
        """
        key = key or self.model_key
        if key is None:
            raise ValueError("No algorithm specified and no model currently selected.")

        current_key = self.model_key
        label = self.model_label(key)

        # Temporarily borrow explain()'s lookup table without mutating state.
        self.model_key = key
        description = self.explain()
        self.model_key = current_key

        return {"key": key, "label": label, "description": description}

    def feature_importance_like(self, X, feature_names=None) -> pd.DataFrame:
        """
        Proxy "feature importance" for clustering: the variance of each
        feature's mean across clusters, normalized to sum to 1. Features
        that differ a lot between cluster centroids are more responsible
        for separating the clusters.
        """
        self._require_fitted()

        X = np.asarray(X)
        if feature_names is None:
            feature_names = [f"Feature {i + 1}" for i in range(X.shape[1])]

        df = pd.DataFrame(X, columns=feature_names)
        df["Cluster"] = self.labels_
        means = df.groupby("Cluster").mean()

        variance = means.var(axis=0)
        importance = variance / (variance.sum() + 1e-12)

        result = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance.values,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        return result

    # ==================================================
    # Sampling / examples
    # ==================================================

    def random_sample(self, X, cluster: Optional[int] = None, n: int = 5) -> pd.DataFrame:
        """
        Return n random rows from X, optionally restricted to one cluster.
        """
        self._require_fitted()

        X = np.asarray(X)
        labels = np.asarray(self.labels_)

        if cluster is not None:
            idx = np.where(labels == cluster)[0]
        else:
            idx = np.arange(len(labels))

        if len(idx) == 0:
            return pd.DataFrame()

        rng = np.random.RandomState(self.random_state)
        chosen = rng.choice(idx, size=min(n, len(idx)), replace=False)

        df = pd.DataFrame(X[chosen])
        df.insert(0, "Cluster", labels[chosen])
        df.insert(0, "Index", chosen)

        return df.reset_index(drop=True)

    def cluster_examples(self, X, n: int = 3) -> pd.DataFrame:
        """
        Return n representative examples (closest to centroid, if
        available; otherwise random) for every cluster.
        """
        self._require_fitted()

        X = np.asarray(X)
        labels = np.asarray(self.labels_)
        rows = []

        for cluster in np.unique(labels):
            idx = np.where(labels == cluster)[0]

            if self.cluster_centers_ is not None and cluster >= 0 and cluster < len(self.cluster_centers_):
                center = self.cluster_centers_[cluster]
                dists = np.linalg.norm(X[idx] - center, axis=1)
                closest = idx[np.argsort(dists)[:n]]
            else:
                rng = np.random.RandomState(self.random_state)
                closest = rng.choice(idx, size=min(n, len(idx)), replace=False)

            for i in closest:
                row = {"Cluster": int(cluster), "Index": int(i)}
                for j, val in enumerate(X[i]):
                    row[f"Feature {j + 1}"] = val
                rows.append(row)

        return pd.DataFrame(rows)

    # ==================================================
    # Probabilistic outputs
    # ==================================================

    def predict_proba(self, X):
        """
        Soft cluster-membership probabilities. Only supported by
        probabilistic models (currently: Gaussian Mixture).
        """
        self._require_fitted()

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        raise AttributeError(
            f"{self.model_name} does not support predict_proba(). "
            "Only probabilistic models such as GMM support this."
        )

    def decision_function(self, X):
        """
        Distance/score of each sample to the decision boundary, where
        supported by the underlying estimator (e.g. some density models).
        """
        self._require_fitted()

        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        if hasattr(self.model, "score_samples"):
            return self.model.score_samples(X)

        raise AttributeError(f"{self.model_name} does not support decision_function().")

    # ==================================================
    # Batch operations
    # ==================================================

    def batch_predict(self, X, batch_size: int = 1000):
        """
        Predict cluster labels in batches (useful for memory-constrained
        environments or very large inference sets).
        """
        self._require_fitted()

        X = np.asarray(X)
        results = []

        for start in range(0, len(X), batch_size):
            chunk = X[start:start + batch_size]
            results.append(self.predict(chunk))

        return np.concatenate(results)

    def batch_fit(self, X, batch_size: int = 1000):
        """
        Incrementally fit using partial_fit, for estimators that support
        it (currently: MiniBatch K-Means). Falls back to a single full
        fit() for estimators without partial_fit support.
        """
        self._require_model()

        X = np.asarray(X)

        if not hasattr(self.model, "partial_fit"):
            warnings.warn(
                f"{self.model_name} does not support partial_fit(); "
                "falling back to a single full fit() call."
            )
            return self.fit(X)

        for start in range(0, len(X), batch_size):
            chunk = X[start:start + batch_size]
            self.model.partial_fit(chunk)

        self.labels_ = self.model.predict(X)
        if hasattr(self.model, "cluster_centers_"):
            self.cluster_centers_ = self.model.cluster_centers_
        self.is_fitted = True

        return self

    # ==================================================
    # Pipelines
    # ==================================================

    def build_pipeline(self, steps: Optional[list] = None):
        """
        Return an sklearn Pipeline chaining any preprocessing steps with
        the currently selected clustering estimator. `steps` should be a
        list of (name, transformer) tuples, e.g. [("scale", StandardScaler())].
        The clustering model is always appended as the final step.
        """
        self._require_model()
        from sklearn.pipeline import Pipeline

        steps = list(steps) if steps else []
        steps.append((self.model_key or "cluster", self.model))

        return Pipeline(steps)

    # ==================================================
    # Performance / benchmarking
    # ==================================================

    def memory_usage(self) -> int:
        """
        Approximate serialized size (in bytes) of the fitted model,
        via an in-memory joblib dump.
        """
        self._require_model()

        buffer = io.BytesIO()
        joblib.dump(self.model, buffer)
        return buffer.tell()

    def time_training(self, X, **fit_kwargs) -> Dict[str, Any]:
        """
        Time how long fit() takes on the currently selected model.
        """
        self._require_model()

        start = time.perf_counter()
        self.fit(X, **fit_kwargs)
        elapsed = time.perf_counter() - start

        return {"Model": self.model_name, "Seconds": elapsed, "Samples": len(np.asarray(X))}

    def benchmark(self, X, algorithms: Optional[List[str]] = None, scale: bool = True) -> pd.DataFrame:
        """
        Time fit() for each algorithm in `algorithms` (defaults to all
        registered models) and report elapsed seconds plus cluster count.
        """
        if algorithms is None:
            algorithms = self.available_models()

        X_use = self.preprocess(X) if scale else np.asarray(X)

        rows = []

        for algorithm in algorithms:
            self.set_model(algorithm)

            start = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.fit(X_use)
                elapsed = time.perf_counter() - start
                clusters = self.number_of_clusters()
                status = "OK"
            except Exception as exc:
                elapsed = time.perf_counter() - start
                clusters = None
                status = f"Failed: {exc}"

            rows.append({
                "Algorithm": self.model_label(algorithm),
                "_key": algorithm,
                "Seconds": elapsed,
                "Clusters": clusters,
                "Status": status,
            })

        return pd.DataFrame(rows).sort_values("Seconds").reset_index(drop=True)

    # ==================================================
    # Reproducibility
    # ==================================================

    @staticmethod
    def seed_everything(seed: int = 42):
        """
        Seed Python's `random` and NumPy's global RNG for reproducibility.
        Note: this does not retroactively change self.random_state on any
        already-built estimator; pass random_state explicitly for that.
        """
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def config(self) -> dict:
        """
        Snapshot of the current configuration (not the fitted state) —
        useful for logging or reproducing a run.
        """
        return {
            "framework": "PsyInsight AI",
            "version": __version__,
            "random_state": self.random_state,
            "model_key": self.model_key,
            "model_params": self.model.get_params() if self.model is not None else None,
            "scaler": type(self.scaler).__name__ if self.scaler is not None else None,
        }

    # ==================================================
    # JSON (de)serialization
    # ==================================================

    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Serialize lightweight, JSON-safe state (config + metrics-friendly
        summaries — not the fitted estimator object itself, which is
        binary; use save_model()/save_project() for that) to a JSON string,
        optionally also writing it to `filepath`.
        """
        payload = {
            "framework": "PsyInsight AI",
            "version": __version__,
            "random_state": self.random_state,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "model_params": self.model.get_params() if self.model is not None else None,
            "is_fitted": self.is_fitted,
        }

        if self.is_fitted:
            payload["n_clusters"] = self.number_of_clusters()
            payload["cluster_sizes"] = {str(k): v for k, v in self.cluster_sizes().items()}
            payload["labels"] = np.asarray(self.labels_).tolist()

        text = json.dumps(payload, indent=2, default=str)

        if filepath:
            with open(filepath, "w") as f:
                f.write(text)

        return text

    def from_json(self, filepath_or_text: str):
        """
        Restore configuration (model choice + hyperparameters + labels,
        if present) from a JSON file path or JSON string produced by
        to_json(). Rebuilds the estimator via set_model() but does NOT
        re-fit it automatically — call fit(X) again if you need a live
        estimator, since raw JSON can't carry the fitted sklearn object.
        """
        if os.path.exists(filepath_or_text):
            with open(filepath_or_text, "r") as f:
                payload = json.load(f)
        else:
            payload = json.loads(filepath_or_text)

        self.random_state = payload.get("random_state", self.random_state)

        model_key = payload.get("model_key")
        model_params = payload.get("model_params") or {}

        if model_key:
            # Only pass through parameters relevant to this algorithm's
            # builder signature; unknown/incompatible kwargs are dropped
            # rather than raising, since sklearn params differ per model.
            try:
                self.set_model(model_key, **model_params)
            except TypeError:
                self.set_model(model_key)

        if payload.get("is_fitted") and "labels" in payload:
            self.labels_ = np.array(payload["labels"])
            self.is_fitted = True

        return self


# =============================================================================
# ClusterStatsMixin
# =============================================================================

class ClusterStatsMixin:
    """
    Per-cluster statistics, noise analysis (DBSCAN/OPTICS), safe
    prediction, dataset diagnostics, outlier detection, cluster
    renaming, and lightweight export helpers.

    Assumes PsyClusteringCore is inherited.
    """

    # ==================================================
    # Internal helper
    # ==================================================

    def _as_dataframe(self, X, feature_names=None):
        """
        Coerce X into a DataFrame with sensible column names.
        (Shared naming convention with VisualizationMixin; duplicated
        here so this mixin has no hard dependency on it.)
        """
        X = np.asarray(X)
        if feature_names is None:
            feature_names = [f"Feature {i + 1}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=feature_names)

    def _validate_cluster_id(self, cluster):
        sizes = self.cluster_sizes()
        if cluster not in sizes:
            raise InvalidClusterError(
                f"Cluster '{cluster}' does not exist. Valid clusters: {sorted(sizes.keys())}"
            )

    # ==================================================
    # Cluster statistics
    # ==================================================

    def cluster_means(self, X, feature_names=None) -> pd.DataFrame:
        """
        Mean of each feature, per cluster.
        """
        self._require_fitted()
        df = self._as_dataframe(X, feature_names)
        df["Cluster"] = self.labels_
        return df.groupby("Cluster").mean()

    def cluster_variances(self, X, feature_names=None) -> pd.DataFrame:
        """
        Variance of each feature, per cluster.
        """
        self._require_fitted()
        df = self._as_dataframe(X, feature_names)
        df["Cluster"] = self.labels_
        return df.groupby("Cluster").var()

    def cluster_std(self, X, feature_names=None) -> pd.DataFrame:
        """
        Standard deviation of each feature, per cluster.
        """
        self._require_fitted()
        df = self._as_dataframe(X, feature_names)
        df["Cluster"] = self.labels_
        return df.groupby("Cluster").std()

    def cluster_feature_summary(self, X, feature_names=None) -> pd.DataFrame:
        """
        Combined mean/std/min/max per feature per cluster, in one
        long-format DataFrame — handy for quick per-cluster profiling.
        """
        self._require_fitted()
        df = self._as_dataframe(X, feature_names)
        df["Cluster"] = self.labels_

        agg = df.groupby("Cluster").agg(["mean", "std", "min", "max"])
        agg.columns = [f"{feat} ({stat})" for feat, stat in agg.columns]
        return agg

    def between_cluster_variance(self, X, feature_names=None) -> pd.DataFrame:
        """
        Explicit between-cluster variance per feature — the variance of
        each feature's cluster means, weighted by cluster size. Larger
        values indicate a feature that more strongly differentiates the
        clusters (a simple ANOVA-style importance proxy for clustering,
        where no single ground-truth label exists to compute normal
        feature-importance against).
        """
        self._require_fitted()
        df = self._as_dataframe(X, feature_names)
        df["Cluster"] = self.labels_

        grand_mean = df.drop(columns="Cluster").mean()
        grouped = df.groupby("Cluster")
        sizes = grouped.size()

        weighted_sq_dev = pd.DataFrame(
            {
                cluster: (group.drop(columns="Cluster").mean() - grand_mean) ** 2 * len(group)
                for cluster, group in grouped
            }
        ).T

        between_var = weighted_sq_dev.sum() / sizes.sum()

        result = pd.DataFrame({
            "Feature": between_var.index,
            "Between-Cluster Variance": between_var.values,
        }).sort_values("Between-Cluster Variance", ascending=False).reset_index(drop=True)

        return result

    # ==================================================
    # Noise analysis (DBSCAN / OPTICS)
    # ==================================================

    def noise_ratio(self) -> float:
        """
        Fraction of samples labeled as noise (-1). Zero for algorithms
        that don't produce a noise label.
        """
        self._require_fitted()
        labels = np.asarray(self.labels_)
        if len(labels) == 0:
            return 0.0
        return float(np.sum(labels == -1) / len(labels))

    def noise_samples(self, X=None):
        """
        Indices (and optionally the corresponding rows of X) labeled
        as noise (-1).
        """
        self._require_fitted()
        labels = np.asarray(self.labels_)
        idx = np.where(labels == -1)[0]

        if X is None:
            return idx

        X = np.asarray(X)
        df = pd.DataFrame(X[idx])
        df.insert(0, "Index", idx)
        return df.reset_index(drop=True)

    def core_samples(self):
        """
        Indices of core samples, for algorithms that expose
        `core_sample_indices_` (currently: DBSCAN).
        """
        self._require_fitted()
        if hasattr(self.model, "core_sample_indices_"):
            return self.model.core_sample_indices_
        raise AttributeError(
            f"{self.model_name} does not expose core sample indices "
            "(only DBSCAN does)."
        )

    # ==================================================
    # Safe prediction
    # ==================================================

    def safe_predict(self, X, default=None):
        """
        Like predict(), but instead of raising AttributeError for
        algorithms that don't support out-of-sample prediction (e.g.
        Agglomerative, DBSCAN, OPTICS, Spectral), logs a friendly
        warning and returns `default` (None by default).
        """
        self._require_fitted()

        if hasattr(self.model, "predict"):
            return self.model.predict(X)

        logger.warning(
            f"{self.model_name} does not support predicting on new/unseen data "
            "(this is a limitation of the algorithm itself, not a bug). "
            "Returning default value instead. Consider using a centroid- or "
            "distribution-based algorithm (KMeans, MiniBatchKMeans, Birch, GMM) "
            "if you need to predict on new samples."
        )
        return default

    # ==================================================
    # Cluster renaming
    # ==================================================

    def rename_clusters(self, mapping: Dict[Any, Any]):
        """
        Relabel discovered clusters in place using `mapping`
        (e.g. {0: "Anxious", 1: "Resilient", -1: "Unclassified"}).
        Useful for turning numeric cluster ids into meaningful names
        for psychology/behavioral datasets. Returns self for chaining.
        """
        self._require_fitted()

        labels = pd.Series(self.labels_)
        unmapped = set(labels.unique()) - set(mapping.keys())
        if unmapped:
            raise InvalidClusterError(
                f"mapping is missing entries for cluster id(s): {sorted(unmapped, key=str)}"
            )

        self.labels_ = labels.map(mapping).to_numpy()
        self._cluster_name_mapping = dict(mapping)
        return self

    def cluster_mapping(self) -> Optional[Dict[Any, Any]]:
        """
        Return the mapping last passed to rename_clusters(), or None if
        clusters have not been renamed.
        """
        return getattr(self, "_cluster_name_mapping", None)

    # ==================================================
    # Outlier detection
    # ==================================================

    def distance_from_centroid(self, X) -> np.ndarray:
        """
        Euclidean distance of each sample to its own cluster's centroid.
        Requires a centroid-based model (has cluster_centers_).
        """
        self._require_fitted()

        if self.cluster_centers_ is None:
            raise AttributeError(
                f"{self.model_name} has no centroids to measure distance from."
            )

        X = np.asarray(X)
        labels = np.asarray(self.labels_)
        distances = np.full(len(X), np.nan)

        for cluster in np.unique(labels):
            if cluster == -1 or cluster >= len(self.cluster_centers_):
                continue
            mask = labels == cluster
            center = self.cluster_centers_[cluster]
            distances[mask] = np.linalg.norm(X[mask] - center, axis=1)

        return distances

    def centroid_outliers(self, X, threshold: float = 2.5) -> pd.DataFrame:
        """
        Flag samples whose distance from their cluster centroid exceeds
        `threshold` standard deviations above that cluster's mean
        distance (a simple z-score rule on top of distance_from_centroid).
        """
        distances = self.distance_from_centroid(X)
        labels = np.asarray(self.labels_)

        z_scores = np.full(len(distances), np.nan)

        for cluster in np.unique(labels):
            if cluster == -1:
                continue
            mask = labels == cluster
            cluster_dists = distances[mask]
            mu, sigma = np.nanmean(cluster_dists), np.nanstd(cluster_dists)
            if sigma > 0:
                z_scores[mask] = (cluster_dists - mu) / sigma
            else:
                z_scores[mask] = 0.0

        df = pd.DataFrame({
            "Index": np.arange(len(distances)),
            "Cluster": labels,
            "Distance": distances,
            "Z-Score": z_scores,
            "Is Outlier": z_scores > threshold,
        })

        return df[df["Is Outlier"]].reset_index(drop=True)

    def mahalanobis_outliers(self, X, threshold: float = 3.0) -> pd.DataFrame:
        """
        Flag samples with a large Mahalanobis distance from their own
        cluster's mean, using that cluster's covariance. More sensitive
        to correlated features than a simple Euclidean distance.
        """
        self._require_fitted()
        from scipy.spatial.distance import mahalanobis

        X = np.asarray(X)
        labels = np.asarray(self.labels_)

        results = []

        for cluster in np.unique(labels):
            if cluster == -1:
                continue

            mask = labels == cluster
            cluster_points = X[mask]
            cluster_idx = np.where(mask)[0]

            if len(cluster_points) < 2:
                continue

            mean = cluster_points.mean(axis=0)
            cov = np.cov(cluster_points, rowvar=False)

            try:
                inv_cov = np.linalg.pinv(np.atleast_2d(cov))
            except np.linalg.LinAlgError:
                continue

            for i, point in zip(cluster_idx, cluster_points):
                dist = mahalanobis(point, mean, inv_cov)
                results.append({"Index": int(i), "Cluster": cluster, "Mahalanobis Distance": dist})

        df = pd.DataFrame(results)
        if df.empty:
            return df

        return df[df["Mahalanobis Distance"] > threshold].reset_index(drop=True)

    # ==================================================
    # Dataset diagnostics
    # ==================================================

    def dataset_summary(self, X) -> Dict[str, Any]:
        """
        Basic diagnostics about a dataset before/after clustering:
        shape, missing values, memory footprint, and per-feature variance.
        """
        df = pd.DataFrame(X)

        buffer = io.BytesIO()
        joblib.dump(df, buffer)

        return {
            "Samples": int(df.shape[0]),
            "Features": int(df.shape[1]),
            "Missing Values": int(df.isna().sum().sum()),
            "Missing Values (%)": float(df.isna().sum().sum() * 100 / df.size) if df.size else 0.0,
            "Memory (bytes)": buffer.tell(),
            "Shape": tuple(df.shape),
            "Per-Feature Variance": df.var(numeric_only=True).to_dict(),
        }

    # ==================================================
    # Export helpers
    # ==================================================

    def save_labels(self, filename: str = "cluster_labels.csv") -> str:
        """
        Export just the cluster label for every sample.
        """
        self._require_fitted()
        pd.DataFrame({"Cluster": self.labels_}).to_csv(filename, index=False)
        return filename

    def save_centroids(self, feature_names=None, filename: str = "cluster_centroids.csv") -> str:
        """
        Export cluster centroids (only available for centroid-based models).
        """
        self._require_fitted()

        if self.cluster_centers_ is None:
            raise AttributeError(f"{self.model_name} has no centroids to export.")

        centers = np.asarray(self.cluster_centers_)
        if feature_names is None:
            feature_names = [f"Feature {i + 1}" for i in range(centers.shape[1])]

        df = pd.DataFrame(centers, columns=feature_names)
        df.insert(0, "Cluster", range(len(df)))
        df.to_csv(filename, index=False)
        return filename

    def export_results(self, X, feature_names=None, filename: str = "cluster_results.csv") -> str:
        """
        Export the original data plus cluster label, per-sample distance
        from its centroid (where available), in one combined CSV.
        """
        self._require_fitted()

        df = self._as_dataframe(X, feature_names)
        df["Cluster"] = self.labels_

        if self.cluster_centers_ is not None:
            try:
                df["Distance From Centroid"] = self.distance_from_centroid(X)
            except Exception:
                pass

        df.to_csv(filename, index=False)
        return filename


# =============================================================================
# Combined, ready-to-use class
# =============================================================================

class PsyClustering(
    ClusterStatsMixin,
    UtilityMixin,
    ExternalMetricsMixin,
    VisualizationMixin,
    ReportingMixin,
    AutoClusterMixin,
    PsyClusteringCore,
):
    """
    Full PsyInsight AI clustering framework:
    core model management + auto-comparison + visualization + reporting +
    external validation metrics + production utilities + cluster
    statistics/diagnostics, all in one class.
    """
    pass
