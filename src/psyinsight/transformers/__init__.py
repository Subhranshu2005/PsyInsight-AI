"""
PsyInsight AI — Transformers / NLP module
============================================

Text analysis for open-ended psychology survey and interview responses:
sentiment analysis, zero-shot theme classification, sentence embeddings
and unsupervised theme discovery, keyword extraction.

Gracefully degrades to lightweight, dependency-free fallbacks when
``transformers``/``torch`` are not installed (see :data:`HF_AVAILABLE`).

    from psyinsight.transformers import TextAnalyzer, HF_AVAILABLE
"""

from ._hf_guard import HF_AVAILABLE
from .text_analyzer import TextAnalyzer

__all__ = ["TextAnalyzer", "HF_AVAILABLE"]
