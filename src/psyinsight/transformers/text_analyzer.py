"""
PsyInsight AI — TextAnalyzer (NLP / Transformers module)
==========================================================

Analyzes open-ended survey / interview responses using Hugging Face
transformer pipelines when available (sentiment, zero-shot topic
classification, sentence embeddings), with a dependency-free fallback
lexicon-based sentiment scorer so the module still works in minimal
environments — useful for teaching and for quick offline exploration
before committing to a full transformer download.

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from psyinsight.utils import get_logger

from ._hf_guard import HF_AVAILABLE, require_hf

__all__ = ["TextAnalyzer"]

_logger = get_logger(__name__)

# A small, hand-curated affect lexicon used only as an offline fallback when
# `transformers` is not installed. This is intentionally simple — it exists
# for graceful degradation, not as a replacement for a real sentiment model.
_POSITIVE_WORDS = {
    "good", "great", "happy", "love", "excellent", "positive", "wonderful", "calm",
    "relaxed", "confident", "hopeful", "joy", "pleased", "grateful", "satisfied",
    "comfortable", "supported", "motivated", "content", "enjoy",
}
_NEGATIVE_WORDS = {
    "bad", "sad", "angry", "hate", "terrible", "negative", "awful", "anxious",
    "stressed", "depressed", "worried", "afraid", "fear", "lonely", "upset",
    "frustrated", "overwhelmed", "hopeless", "tired", "exhausted", "hurt",
}


class TextAnalyzer:
    """Unified interface over transformer-based and lexicon-based text
    analysis, tuned for short survey / questionnaire free-text responses.

    Parameters
    ----------
    sentiment_model: HF model id used for sentiment (only loaded on first use).
    embedding_model: HF model id used for sentence embeddings.
    use_transformers: if ``False`` (or transformers isn't installed), all
        methods fall back to lightweight, dependency-free implementations.
    """

    def __init__(
        self,
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_transformers: bool = True,
    ):
        self.sentiment_model_name = sentiment_model
        self.embedding_model_name = embedding_model
        self.use_transformers = use_transformers and HF_AVAILABLE
        self._sentiment_pipeline = None
        self._embedding_pipeline = None
        self._zero_shot_pipeline = None

        if use_transformers and not HF_AVAILABLE:
            _logger.warning(
                "transformers/torch not installed — TextAnalyzer will use the "
                "lexicon-based fallback. Install with `pip install transformers torch` "
                "for full pretrained-model accuracy."
            )

    # ------------------------------------------------------------------
    # Lazy pipeline loaders
    # ------------------------------------------------------------------

    def _get_sentiment_pipeline(self):
        if self._sentiment_pipeline is None:
            transformers = require_hf()
            self._sentiment_pipeline = transformers.pipeline(
                "sentiment-analysis", model=self.sentiment_model_name
            )
        return self._sentiment_pipeline

    def _get_zero_shot_pipeline(self):
        if self._zero_shot_pipeline is None:
            transformers = require_hf()
            self._zero_shot_pipeline = transformers.pipeline(
                "zero-shot-classification", model="facebook/bart-large-mnli"
            )
        return self._zero_shot_pipeline

    def _get_embedding_pipeline(self):
        if self._embedding_pipeline is None:
            transformers = require_hf()
            self._embedding_pipeline = transformers.pipeline(
                "feature-extraction", model=self.embedding_model_name
            )
        return self._embedding_pipeline

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------

    def sentiment(self, texts: Sequence[str]) -> pd.DataFrame:
        """Return a DataFrame with columns ``text``, ``label``, ``score``.

        Uses a pretrained transformer pipeline when available, otherwise a
        lexicon-based fallback (label in {POSITIVE, NEGATIVE, NEUTRAL}).
        """
        texts = [str(t) for t in texts]
        if self.use_transformers:
            pipeline = self._get_sentiment_pipeline()
            results = pipeline(texts, truncation=True)
            return pd.DataFrame(
                {
                    "text": texts,
                    "label": [r["label"] for r in results],
                    "score": [r["score"] for r in results],
                }
            )
        return self._fallback_sentiment(texts)

    def _fallback_sentiment(self, texts: Sequence[str]) -> pd.DataFrame:
        rows = []
        for text in texts:
            words = str(text).lower().split()
            pos = sum(1 for w in words if w.strip(".,!?") in _POSITIVE_WORDS)
            neg = sum(1 for w in words if w.strip(".,!?") in _NEGATIVE_WORDS)
            if pos == neg:
                label, score = "NEUTRAL", 0.5
            elif pos > neg:
                label, score = "POSITIVE", pos / max(pos + neg, 1)
            else:
                label, score = "NEGATIVE", neg / max(pos + neg, 1)
            rows.append({"text": text, "label": label, "score": round(score, 3)})
        return pd.DataFrame(rows)

    def sentiment_summary(self, texts: Sequence[str]) -> Dict[str, Any]:
        """Aggregate sentiment distribution across a batch of responses —
        the typical unit of analysis for an open-ended survey question."""
        result = self.sentiment(texts)
        counts = result["label"].value_counts(normalize=True).round(3).to_dict()
        return {
            "n_responses": len(result),
            "distribution": counts,
            "mean_confidence": float(result["score"].mean()),
            "detail": result,
        }

    # ------------------------------------------------------------------
    # Zero-shot topic / theme classification
    # ------------------------------------------------------------------

    def zero_shot_classify(self, texts: Sequence[str], candidate_labels: List[str]) -> pd.DataFrame:
        """Classify each text against a set of researcher-defined candidate
        themes/labels without any labeled training data (requires
        transformers)."""
        if not self.use_transformers:
            raise ImportError(
                "zero_shot_classify requires transformers/torch to be installed "
                "(no dependency-free fallback exists for this method)."
            )
        pipeline = self._get_zero_shot_pipeline()
        rows = []
        for text in texts:
            result = pipeline(str(text), candidate_labels)
            rows.append(
                {
                    "text": text,
                    "top_label": result["labels"][0],
                    "top_score": result["scores"][0],
                    "all_scores": dict(zip(result["labels"], result["scores"])),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Embeddings + theme discovery
    # ------------------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return dense vector embeddings for each text.

        Uses mean-pooled transformer hidden states when available, else a
        TF-IDF vector (dense) as a lightweight fallback.
        """
        texts = [str(t) for t in texts]
        if self.use_transformers:
            pipeline = self._get_embedding_pipeline()
            raw = pipeline(texts)
            return np.array([np.mean(np.array(vecs[0]), axis=0) for vecs in raw])

        vectorizer = TfidfVectorizer(max_features=256)
        return vectorizer.fit_transform(texts).toarray()

    def discover_themes(self, texts: Sequence[str], n_themes: int = 3) -> Dict[str, Any]:
        """Unsupervised theme discovery: embed responses, cluster them, and
        surface the most representative words per cluster via TF-IDF —
        a quick way to summarize what respondents are talking about."""
        texts = [str(t) for t in texts]
        embeddings = self.embed(texts)
        n_themes = max(1, min(n_themes, len(texts)))

        kmeans = KMeans(n_clusters=n_themes, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)

        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf = vectorizer.fit_transform(texts)
        terms = np.array(vectorizer.get_feature_names_out())

        themes = {}
        for cluster_id in range(n_themes):
            mask = clusters == cluster_id
            if not mask.any():
                themes[cluster_id] = {"keywords": [], "n_responses": 0}
                continue
            mean_tfidf = np.asarray(tfidf[mask].mean(axis=0)).ravel()
            top_terms = terms[np.argsort(mean_tfidf)[::-1][:8]]
            themes[cluster_id] = {
                "keywords": top_terms.tolist(),
                "n_responses": int(mask.sum()),
            }

        return {
            "cluster_assignments": clusters.tolist(),
            "themes": themes,
        }

    # ------------------------------------------------------------------
    # Keyword extraction (always dependency-free — sklearn only)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_keywords(texts: Sequence[str], top_k: int = 10) -> pd.DataFrame:
        """TF-IDF based keyword ranking across a corpus of responses."""
        texts = [str(t) for t in texts if str(t).strip()]
        if not texts:
            return pd.DataFrame(columns=["keyword", "score"])

        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf = vectorizer.fit_transform(texts)
        scores = np.asarray(tfidf.sum(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()

        result = pd.DataFrame({"keyword": terms, "score": scores}).sort_values(
            "score", ascending=False
        ).head(top_k).reset_index(drop=True)
        return result
