"""
PsyInsight AI — Explainable AI (XAI) module
==============================================

Model-agnostic explainability: global feature importance, permutation
importance, partial dependence, LIME-style local explanations, and an
optional SHAP wrapper.

    from psyinsight.xai import PsyExplainer
"""

from .explainer import PsyExplainer

__all__ = ["PsyExplainer"]
