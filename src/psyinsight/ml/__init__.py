"""
PsyInsight AI — Machine Learning module
========================================

Exposes the four core supervised/unsupervised engines plus the shared
metrics, model-selection and feature-selection toolkits under one
convenient namespace::

    from psyinsight.ml import PsyClassifier, PsyRegressor, PsyClustering
    from psyinsight.ml import PsyMetrics, PsyModelSelector, PsyFeatureSelector
"""

from .classifier import PsyClassifier
from .clustering import PsyClustering
from .feature_selection import PsyFeatureSelector
from .metrics import PsyMetrics
from .model_selection import PsyModelSelector
from .regressor import PsyRegressor

__all__ = [
    "PsyClassifier",
    "PsyRegressor",
    "PsyClustering",
    "PsyMetrics",
    "PsyModelSelector",
    "PsyFeatureSelector",
]
