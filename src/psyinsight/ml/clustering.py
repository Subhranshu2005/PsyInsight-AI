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
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone

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
# Preprocessing
# ==========================

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from sklearn.model_selection import train_test_split

# ==========================
# Evaluation
# ==========================

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# ==========================
# Plotting
# ==========================

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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
            raise ValueError(
                f"Unknown model '{key}'."
            )

        return cls._REGISTRY[key]["label"]

    def _build(self, key, **kwargs):

        if key not in self._REGISTRY:
            raise ValueError(
                f"Unknown model '{key}'."
            )

        return self._REGISTRY[key]["builder"](
            self.random_state,
            **kwargs,
        )


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

    def __init__(
        self,
        random_state: int = 42,
    ):

        self.random_state = random_state

        self.model = None

        self.model_name = None

        self.model_key = None

        self.scaler = None

        self.labels_ = None

        self.cluster_centers_ = None

        self.is_fitted = False

    # ==================================================
    # Model Management
    # ==================================================

    def set_model(
        self,
        model_name: str,
        **parameters,
    ):

        self.model = self._build(
            model_name,
            **parameters,
        )

        self.model_name = self.model_label(model_name)

        self.model_key = model_name

        self.is_fitted = False

        return self

    def current_model(self):

        return self.model_name

    def available_algorithms(self):

        return {
            key: self.model_label(key)
            for key in self.available_models()
        }

    def model_info(self):

        if self.model is None:
            raise ValueError(
                "No model selected."
            )

        return {

            "Model": self.model_name,

            "Type": type(self.model).__name__,

            "Parameters": self.model.get_params(),

            "Fitted": self.is_fitted,

        }

    def model_parameters(self):

        if self.model is None:
            raise ValueError(
                "No model selected."
            )

        return self.model.get_params()

    # ==================================================
    # Data
    # ==================================================

    def split_data(

        self,

        X,

        test_size=0.2,

        random_state=None,

    ):

        """
        Mostly used for
        semi-supervised experiments.
        """

        return train_test_split(

            X,

            test_size=test_size,

            random_state=(
                self.random_state
                if random_state is None
                else random_state
            ),

        )

    # ==================================================
    # Scaling
    # ==================================================

    def preprocess(

        self,

        X_train,

        X_test=None,

        scaler="standard",

    ):

        if scaler is None:

            return (
                (X_train, X_test)
                if X_test is not None
                else X_train
            )

        scaler = scaler.lower()

        if scaler == "standard":

            self.scaler = StandardScaler()

        elif scaler == "minmax":

            self.scaler = MinMaxScaler()

        elif scaler == "robust":

            self.scaler = RobustScaler()

        else:

            raise ValueError(
                "Scaler must be "
                "'standard', "
                "'minmax', "
                "'robust' "
                "or None."
            )

        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_test is None:

            return X_train_scaled

        X_test_scaled = self.scaler.transform(X_test)

        return (

            X_train_scaled,

            X_test_scaled,

        )

    def apply_scaler(

        self,

        X,

    ):

        if self.scaler is None:

            raise ValueError(

                "Scaler has not been fitted."

            )

        return self.scaler.transform(X)

    # ==================================================
    # Internal Helpers
    # ==================================================

    def _require_model(self):

        if self.model is None:

            raise ValueError(

                "No model selected.\n"

                "Call\n"

                "set_model() first."

            )

    def _require_fitted(self):

        self._require_model()

        if not self.is_fitted:

            raise ValueError(

                "Model has not been fitted."

            )

    # ==================================================
    # Utilities
    # ==================================================

    def framework(self):

        return {

            "Framework": "PsyInsight AI",

            "Module": "Machine Learning",

            "Component": "PsyClustering",

            "Version": __version__,

            "Supported Models": len(

                self.available_models()

            ),

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
