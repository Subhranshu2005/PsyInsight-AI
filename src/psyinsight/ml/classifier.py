"""
PsyInsight AI — PsyClassifier Pro
==================================
Single-file Machine Learning Classification Framework

Author: Subhranshu Ranjan Sahoo

A unified, extensible classification engine for PsyInsight AI.
Instead of one hand-written method per algorithm, models are
selected through a small registry (`set_model("random_forest")`),
so training, evaluation, AutoML comparison, visualization and PDF
reporting all share the exact same code path no matter which
algorithm is active.

Quick start
-----------
    from psyclassifier_pro import PsyClassifier

    clf = PsyClassifier()
    X_train, X_test, y_train, y_test = clf.split_data(X, y)
    result = clf.auto_train(X_train, X_test, y_train, y_test)
    clf.visualize_results(X_test, y_test, feature_names=feature_names)
    clf.generate_report(X_test, y_test, feature_names=feature_names)

Sections in this file
----------------------
    1. Imports
    2. ModelRegistryMixin      — algorithm catalogue / factory
    3. PsyClassifierCore       — model mgmt, data, training, metrics, explainability, persistence
    4. AutoMLMixin             — compare_models / leaderboard / best_model / auto_train
    5. VisualizationMixin      — ROC, confusion matrix, PR curve, importance, learning/validation curves
    6. ReportMixin             — generate_report() -> PDF
    7. PsyClassifier           — the public class, composed from all mixins above
"""

# =============================================================================
# 1. Imports
# =============================================================================

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Optional, Dict, Any, List

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_curve,
    auc,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    learning_curve,
    validation_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)


# =============================================================================
# 2. ModelRegistryMixin — algorithm catalogue / factory
# =============================================================================

class ModelRegistryMixin:
    """
    Holds the catalogue of supported algorithms and turns a plain
    string ("random_forest") into a configured, unfitted scikit-learn
    estimator. Adding a new algorithm to the whole framework (AutoML,
    leaderboard, etc.) means adding one line here — nothing else has
    to change.
    """

    #: name -> {label, build}
    _REGISTRY: Dict[str, Dict[str, Any]] = {
        "logistic_regression": {
            "label": "Logistic Regression",
            "build": lambda rs, **kw: LogisticRegression(
                random_state=rs, max_iter=kw.pop("max_iter", 1000), **kw
            ),
        },
        "decision_tree": {
            "label": "Decision Tree",
            "build": lambda rs, **kw: DecisionTreeClassifier(
                random_state=rs, **kw
            ),
        },
        "random_forest": {
            "label": "Random Forest",
            "build": lambda rs, **kw: RandomForestClassifier(
                n_estimators=kw.pop("n_estimators", 100),
                random_state=rs,
                **kw,
            ),
        },
        "svm": {
            "label": "Support Vector Machine",
            "build": lambda rs, **kw: SVC(
                kernel=kw.pop("kernel", "rbf"),
                C=kw.pop("C", 1.0),
                probability=True,
                random_state=rs,
                **kw,
            ),
        },
        "knn": {
            "label": "K Nearest Neighbors",
            "build": lambda rs, **kw: KNeighborsClassifier(
                n_neighbors=kw.pop("n_neighbors", 5), **kw
            ),
        },
        "naive_bayes": {
            "label": "Gaussian Naive Bayes",
            "build": lambda rs, **kw: GaussianNB(**kw),
        },
    }

    @classmethod
    def available_models(cls) -> List[str]:
        """Return the list of registered model keys."""
        return list(cls._REGISTRY.keys())

    @classmethod
    def model_label(cls, key: str) -> str:
        """Return the human-readable label for a model key."""
        cls._validate_key(key)
        return cls._REGISTRY[key]["label"]

    @classmethod
    def _validate_key(cls, key: str) -> None:
        if key not in cls._REGISTRY:
            valid = ", ".join(cls._REGISTRY.keys())
            raise ValueError(f"Unknown model '{key}'. Available models: {valid}")

    def _build(self, key: str, **kwargs):
        self._validate_key(key)
        return self._REGISTRY[key]["build"](self.random_state, **kwargs)


# =============================================================================
# 3. PsyClassifierCore — model mgmt, data, training, metrics, explainability
# =============================================================================

class PsyClassifierCore(ModelRegistryMixin):
    """
    Unified classification engine for PsyInsight AI.

    Example
    -------
    >>> clf = PsyClassifierCore()
    >>> clf.set_model("random_forest")
    >>> clf.fit(X_train, y_train)
    >>> clf.predict(X_test)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.model_name: Optional[str] = None
        self._model_key: Optional[str] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def set_model(self, name: str, **params):
        """
        Select and configure the active model by name.

        Parameters
        ----------
        name :
            One of `PsyClassifierCore.available_models()`, e.g.
            "random_forest", "svm", "logistic_regression".
        **params :
            Optional hyperparameters forwarded to the underlying
            scikit-learn estimator (e.g. n_estimators=200).
        """
        self.model = self._build(name, **params)
        self.model_name = self.model_label(name)
        self._model_key = name
        self.is_fitted = False
        return self

    def current_model(self) -> Optional[str]:
        """Return the display name of the currently selected model."""
        return self.model_name

    def _require_model(self):
        if self.model is None:
            raise ValueError(
                "No model selected. Call set_model(name) first, "
                "e.g. clf.set_model('random_forest')."
            )

    def _require_fitted(self):
        self._require_model()
        if not self.is_fitted:
            raise ValueError(
                f"{self.model_name} has not been trained yet. Call fit() first."
            )

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------

    def split_data(self, X, y, test_size: float = 0.2, random_state: Optional[int] = None):
        """Stratified train/test split."""
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state if random_state is not None else self.random_state,
            stratify=y,
        )

    def preprocess(self, X_train, X_test=None, scale: bool = True):
        """
        Fit a StandardScaler on X_train (and reuse it) so
        distance/gradient-based models (SVM, KNN, Logistic
        Regression) get standardized input. Tree-based models are
        unaffected by scaling but it does no harm.

        Returns X_train (transformed) if X_test is None, else a
        (X_train, X_test) tuple.
        """
        if not scale:
            return (X_train, X_test) if X_test is not None else X_train

        self.scaler = StandardScaler()
        X_train_t = self.scaler.fit_transform(X_train)

        if X_test is None:
            return X_train_t

        X_test_t = self.scaler.transform(X_test)
        return X_train_t, X_test_t

    def apply_scaler(self, X):
        """Transform new data using a previously-fit scaler."""
        if self.scaler is None:
            raise ValueError("No scaler has been fit yet. Call preprocess() first.")
        return self.scaler.transform(X)

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train):
        """Train the currently selected model."""
        self._require_model()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X_test):
        self._require_fitted()
        return self.model.predict(X_test)

    def predict_probability(self, X_test):
        self._require_fitted()
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{self.model_name} does not support predict_proba().")
        return self.model.predict_proba(X_test)

    def predict_log_probability(self, X_test):
        self._require_fitted()
        if not hasattr(self.model, "predict_log_proba"):
            raise AttributeError(f"{self.model_name} does not support predict_log_proba().")
        return self.model.predict_log_proba(X_test)

    def decision_function(self, X_test):
        self._require_fitted()
        if not hasattr(self.model, "decision_function"):
            raise AttributeError(f"{self.model_name} does not support decision_function().")
        return self.model.decision_function(X_test)

    def score(self, X_test, y_test):
        self._require_fitted()
        return self.model.score(X_test, y_test)

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def feature_importance(self, feature_names=None):
        """
        Return feature importances (Decision Tree / Random Forest)
        or |coefficient| magnitudes (Logistic Regression / linear SVM)
        as a fallback, so explainability isn't limited to tree models.
        """
        self._require_fitted()

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            coef = self.model.coef_
            importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            raise AttributeError(
                f"{self.model_name} does not provide feature importance. "
                "Available for tree-based and linear models."
            )

        if feature_names is None:
            return importance

        return (
            pd.DataFrame({"Feature": feature_names, "Importance": importance})
            .sort_values(by="Importance", ascending=False)
            .reset_index(drop=True)
        )

    def ranked_features(self, feature_names):
        """Alias for feature_importance() that always returns a ranked DataFrame."""
        return self.feature_importance(feature_names=feature_names)

    def explain_prediction(self, X_row, feature_names=None, top_n: int = 5):
        """
        Lightweight, model-agnostic explanation for a single row:
        combines the model's global feature importance with that
        row's own feature values so a reviewer can see which features
        likely drove the prediction, without pulling in a SHAP
        dependency.
        """
        self._require_fitted()
        row = np.asarray(X_row).reshape(1, -1)
        prediction = self.model.predict(row)[0]

        try:
            importance_df = self.feature_importance(feature_names=feature_names)
        except AttributeError:
            importance_df = None

        result = {"Prediction": prediction}

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(row)[0]
            result["Confidence"] = float(np.max(proba))

        if importance_df is not None:
            top = importance_df.head(top_n).copy()
            if feature_names is not None:
                name_to_idx = {name: i for i, name in enumerate(feature_names)}
                top["Value"] = [row[0][name_to_idx[f]] for f in top["Feature"]]
            result["Top Contributing Features"] = top

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, filename: str):
        self._require_model()
        joblib.dump(
            {
                "model": self.model,
                "model_name": self.model_name,
                "model_key": self._model_key,
                "scaler": self.scaler,
            },
            filename,
        )

    def load_model(self, filename: str):
        payload = joblib.load(filename)
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            self.model_name = payload.get("model_name", type(self.model).__name__)
            self._model_key = payload.get("model_key")
            self.scaler = payload.get("scaler")
        else:
            # Backward compatible with plain-estimator dumps.
            self.model = payload
            self.model_name = type(self.model).__name__
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred, average: str = "weighted"):
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def recall(y_true, y_pred, average: str = "weighted"):
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def f1(y_true, y_pred, average: str = "weighted"):
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def confusion(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def report_text(y_true, y_pred):
        return classification_report(y_true, y_pred, zero_division=0)

    @staticmethod
    def roc_auc(y_true, y_probability, multi_class: str = "ovr"):
        y_probability_arr = np.asarray(y_probability)
        n_classes = y_probability_arr.shape[1] if y_probability_arr.ndim > 1 else 2
        if n_classes == 2 and y_probability_arr.ndim > 1:
            return roc_auc_score(y_true, y_probability_arr[:, 1])
        return roc_auc_score(y_true, y_probability_arr, multi_class=multi_class)

    @staticmethod
    def matthews_correlation(y_true, y_pred):
        return matthews_corrcoef(y_true, y_pred)

    @staticmethod
    def cohen_kappa(y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred)

    def cross_validation(self, X, y, cv: int = 5, scoring: str = "accuracy"):
        self._require_model()
        return cross_val_score(self.model, X, y, cv=cv, scoring=scoring)

    def stratified_cross_validation(self, X, y, splits: int = 5, scoring: str = "accuracy"):
        self._require_model()
        stratified = StratifiedKFold(n_splits=splits, shuffle=True, random_state=self.random_state)
        return cross_val_score(self.model, X, y, cv=stratified, scoring=scoring)

    def evaluate(self, y_true, y_pred) -> Dict[str, float]:
        """Core set of metrics used everywhere (leaderboard, reports, summary)."""
        return {
            "Accuracy": self.accuracy(y_true, y_pred),
            "Precision": self.precision(y_true, y_pred),
            "Recall": self.recall(y_true, y_pred),
            "F1 Score": self.f1(y_true, y_pred),
            "Matthews Correlation": self.matthews_correlation(y_true, y_pred),
            "Cohen Kappa": self.cohen_kappa(y_true, y_pred),
        }

    def print_evaluation(self, y_true, y_pred):
        metrics = self.evaluate(y_true, y_pred)
        print("\n========== MODEL ==========")
        print(self.model_name)
        print()
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

    @staticmethod
    def interpretation(accuracy: float) -> str:
        if accuracy >= 0.95:
            return "Outstanding model performance."
        elif accuracy >= 0.90:
            return "Excellent model performance."
        elif accuracy >= 0.80:
            return "Good model performance."
        elif accuracy >= 0.70:
            return "Acceptable model performance."
        else:
            return "Model requires improvement."

    def model_parameters(self):
        self._require_model()
        return self.model.get_params()

    def model_summary(self):
        self._require_model()
        return {
            "Model Name": self.model_name,
            "Model Type": type(self.model).__name__,
            "Parameters": self.model.get_params(),
        }

    def summary(self, y_true, y_pred):
        accuracy = self.accuracy(y_true, y_pred)
        return {
            "Model": self.model_name,
            "Accuracy": accuracy,
            "Precision": self.precision(y_true, y_pred),
            "Recall": self.recall(y_true, y_pred),
            "F1 Score": self.f1(y_true, y_pred),
            "Interpretation": self.interpretation(accuracy),
        }
        # ------------------------------------------------------------------
    # Information Utilities
    # ------------------------------------------------------------------

    def available_algorithms(self):
        """
        Return all supported classification algorithms.
        """
        return {
            key: self.model_label(key)
            for key in self.available_models()
        }

    def model_info(self):
        """
        Return information about the currently selected model.
        """
        self._require_model()

        return {
            "Model Name": self.model_name,
            "Model Type": type(self.model).__name__,
            "Parameters": self.model.get_params(),
            "Fitted": self.is_fitted,
        }

    def framework(self):
        """
        Return framework information.
        """
        return {
            "Framework": "PsyInsight AI",
            "Module": "Machine Learning",
            "Component": "PsyClassifier",
            "Version": __version__,
            "Supported Models": len(self.available_models()),
        }

    def version(self):
        """
        Return current classifier version.
        """
        return __version__
    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        self.model = None
        self.model_name = None
        self._model_key = None
        self.is_fitted = False
        return self

    clear = reset


# =============================================================================
# 4. AutoMLMixin — compare_models / leaderboard / best_model / auto_train
# =============================================================================

class AutoMLMixin:
    """
    Mixin providing multi-model comparison and selection on top of
    `PsyClassifierCore`. Assumes `self` also has `set_model`, `fit`,
    `predict`, `predict_probability`, `evaluate`, `available_models`,
    `model_label`.
    """

    def compare_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        models: Optional[List[str]] = None,
        scale: bool = True,
    ) -> pd.DataFrame:
        """
        Train and evaluate every model in `models` (default: all
        registered models) on the same train/test split.

        Returns a DataFrame — one row per model — with Accuracy,
        Precision, Recall, F1, MCC, Cohen Kappa and, where available,
        ROC AUC. The underlying fitted estimators are kept internally
        so `best_model` can retrieve the winner without retraining.
        """
        models = models or self.available_models()

        if scale:
            X_train_use, X_test_use = self.preprocess(X_train, X_test, scale=True)
        else:
            X_train_use, X_test_use = X_train, X_test

        rows = []
        self._fitted_candidates: Dict[str, object] = {}

        for key in models:
            self.set_model(key)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fit(X_train_use, y_train)

            y_pred = self.predict(X_test_use)
            metrics = self.evaluate(y_test, y_pred)

            try:
                y_proba = self.predict_probability(X_test_use)
                metrics["ROC AUC"] = self.roc_auc(y_test, y_proba)
            except (AttributeError, ValueError):
                metrics["ROC AUC"] = np.nan

            metrics_row = {"Model": self.model_label(key), "_key": key}
            metrics_row.update(metrics)
            rows.append(metrics_row)

            # Keep a fitted, ready-to-use copy of this candidate.
            self._fitted_candidates[key] = self.model

        results = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
        self._last_results = results
        return results

    def leaderboard(self, results: Optional[pd.DataFrame] = None, metric: str = "Accuracy") -> pd.DataFrame:
        """
        Return (and pretty-print) a ranked leaderboard from the
        results of `compare_models`. Defaults to the last comparison
        run if `results` isn't supplied.
        """
        results = self._resolve_results(results)
        ranked = results.sort_values(by=metric, ascending=False).reset_index(drop=True)

        print("\n=========================")
        print("MODEL LEADERBOARD")
        print("=========================\n")
        for _, row in ranked.iterrows():
            print(f"{row['Model']:<20} {row[metric] * 100:5.1f}%")

        return ranked

    def best_model(self, results: Optional[pd.DataFrame] = None, metric: str = "Accuracy"):
        """
        Select the top model by `metric`, load its fitted estimator
        back onto `self` (so `self.model`/`self.predict` point at the
        winner), and return (key, row) describing the pick.
        """
        results = self._resolve_results(results)
        ranked = results.sort_values(by=metric, ascending=False).reset_index(drop=True)
        top_row = ranked.iloc[0]
        top_key = top_row["_key"]

        self.model = self._fitted_candidates[top_key]
        self.model_name = self.model_label(top_key)
        self._model_key = top_key
        self.is_fitted = True

        return top_key, top_row

    def _resolve_results(self, results: Optional[pd.DataFrame]) -> pd.DataFrame:
        if results is not None:
            return results
        if getattr(self, "_last_results", None) is not None:
            return self._last_results
        raise ValueError("No comparison results available. Call compare_models() or auto_train() first.")

    @staticmethod
    def _best_model_reason(ranked: pd.DataFrame) -> List[str]:
        """Derive a short, data-backed rationale rather than a canned string."""
        reasons = []
        top = ranked.iloc[0]

        reasons.append(f"Highest accuracy ({top['Accuracy'] * 100:.1f}%).")

        if len(ranked) > 1 and top["Recall"] >= ranked.iloc[1:]["Recall"].max():
            reasons.append(f"Best recall ({top['Recall'] * 100:.1f}%), important for not missing true cases.")

        if not np.isnan(top.get("ROC AUC", np.nan)):
            reasons.append(f"Strong ROC AUC ({top['ROC AUC']:.3f}) indicating good class separation.")

        reasons.append(f"Cohen's Kappa of {top['Cohen Kappa']:.3f} shows agreement beyond chance.")

        return reasons

    def auto_train(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        models: Optional[List[str]] = None,
        metric: str = "Accuracy",
        scale: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        One call to train every model, rank them, and load the winner:

        >>> clf = PsyClassifier()
        >>> clf.auto_train(X_train, X_test, y_train, y_test)

        Returns a dict with the full `results` table, the `best_key`,
        `best_row`, and the human-readable `reasons` for the pick.
        """
        results = self.compare_models(X_train, X_test, y_train, y_test, models=models, scale=scale)
        ranked = results.sort_values(by=metric, ascending=False).reset_index(drop=True)
        best_key, best_row = self.best_model(ranked, metric=metric)
        reasons = self._best_model_reason(ranked)

        if verbose:
            self.leaderboard(ranked, metric=metric)
            print(f"\nBest Model\n\n{best_row['Model']}\n")
            print("Reason\n")
            for r in reasons:
                print(r)
            print("\nRecommended for psychological classification.\n")

        return {
            "results": ranked,
            "best_key": best_key,
            "best_row": best_row,
            "reasons": reasons,
        }


# =============================================================================
# 5. VisualizationMixin — ROC, confusion matrix, PR curve, importance, curves
# =============================================================================

class VisualizationMixin:
    """
    Mixin providing `visualize_results()` and its component plots.
    Assumes `self` has a fitted `self.model` (from `PsyClassifierCore`).
    """

    def _fig_path(self, save_dir: str, name: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, name)

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self, X_test, y_test, save_dir: str = "psy_plots", class_names=None):
        self._require_fitted()
        y_pred = self.predict(X_test)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=class_names, cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(f"Confusion Matrix — {self.model_name}")
        path = self._fig_path(save_dir, "confusion_matrix.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_roc_curve(self, X_test, y_test, save_dir: str = "psy_plots"):
        self._require_fitted()
        if not hasattr(self.model, "predict_proba"):
            return None

        y_proba = self.predict_probability(X_test)
        classes = np.unique(y_test)
        fig, ax = plt.subplots(figsize=(5.5, 5))

        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
        else:
            y_bin = label_binarize(y_test, classes=classes)
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"Class {cls} (AUC = {roc_auc_val:.3f})")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {self.model_name}")
        ax.legend(loc="lower right", fontsize=8)
        path = self._fig_path(save_dir, "roc_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_precision_recall_curve(self, X_test, y_test, save_dir: str = "psy_plots"):
        self._require_fitted()
        if not hasattr(self.model, "predict_proba"):
            return None

        y_proba = self.predict_probability(X_test)
        classes = np.unique(y_test)
        fig, ax = plt.subplots(figsize=(5.5, 5))

        if len(classes) == 2:
            precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
            ax.plot(recall, precision, label=self.model_name)
        else:
            y_bin = label_binarize(y_test, classes=classes)
            for i, cls in enumerate(classes):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
                ax.plot(recall, precision, label=f"Class {cls}")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision–Recall Curve — {self.model_name}")
        ax.legend(loc="lower left", fontsize=8)
        path = self._fig_path(save_dir, "precision_recall_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_feature_importance(self, feature_names, save_dir: str = "psy_plots", top_n: int = 15):
        self._require_fitted()
        try:
            ranked = self.feature_importance(feature_names=feature_names)
        except AttributeError:
            return None

        top = ranked.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(top))))
        ax.barh(top["Feature"], top["Importance"], color="#4C72B0")
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {self.model_name}")
        path = self._fig_path(save_dir, "feature_importance.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_learning_curve(self, X, y, save_dir: str = "psy_plots", cv: int = 5):
        self._require_model()
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=cv, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 6), random_state=self.random_state,
        )
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training score")
        ax.plot(train_sizes, test_scores.mean(axis=1), "o-", label="Cross-validation score")
        ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                         train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
        ax.fill_between(train_sizes, test_scores.mean(axis=1) - test_scores.std(axis=1),
                         test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.15)
        ax.set_xlabel("Training Examples")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Learning Curve — {self.model_name}")
        ax.legend(loc="best", fontsize=8)
        path = self._fig_path(save_dir, "learning_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_validation_curve(self, X, y, param_name: str, param_range, save_dir: str = "psy_plots", cv: int = 5):
        self._require_model()
        train_scores, test_scores = validation_curve(
            self.model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring="accuracy",
        )
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(param_range, train_scores.mean(axis=1), "o-", label="Training score")
        ax.plot(param_range, test_scores.mean(axis=1), "o-", label="Cross-validation score")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Validation Curve — {self.model_name}")
        ax.legend(loc="best", fontsize=8)
        path = self._fig_path(save_dir, "validation_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_decision_boundary_2d(self, X, y, save_dir: str = "psy_plots", feature_names=None):
        """
        2D decision boundary. If X has more than 2 features, the
        first two columns are used (label this clearly — it's a
        projection, not the full decision surface).
        """
        self._require_fitted()
        X = np.asarray(X)
        if X.shape[1] > 2:
            X = X[:, :2]

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

        # Refit a lightweight clone on just these 2 dims for a true 2D boundary.
        model_2d = clone(self.model)
        model_2d.fit(X, y)
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=20)
        f1_label = feature_names[0] if feature_names else "Feature 1"
        f2_label = feature_names[1] if feature_names else "Feature 2"
        ax.set_xlabel(f1_label)
        ax.set_ylabel(f2_label)
        ax.set_title(f"Decision Boundary (2D projection) — {self.model_name}")
        path = self._fig_path(save_dir, "decision_boundary.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Everything at once
    # ------------------------------------------------------------------

    def visualize_results(
        self,
        X_test,
        y_test,
        X_train=None,
        y_train=None,
        feature_names=None,
        save_dir: str = "psy_plots",
        include: Optional[List[str]] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Generate the standard PsyInsight AI visual suite in one call:
        ROC curve, confusion matrix, precision-recall curve, feature
        importance, learning curve, and (if X_train/y_train given) a
        2D decision boundary projection. Returns a dict of plot name
        -> saved file path (None if that plot wasn't applicable, e.g.
        no predict_proba).
        """
        self._require_fitted()
        include = include or [
            "confusion_matrix", "roc_curve", "precision_recall_curve",
            "feature_importance", "learning_curve", "decision_boundary",
        ]

        paths: Dict[str, Optional[str]] = {}

        if "confusion_matrix" in include:
            paths["confusion_matrix"] = self.plot_confusion_matrix(X_test, y_test, save_dir=save_dir)
        if "roc_curve" in include:
            paths["roc_curve"] = self.plot_roc_curve(X_test, y_test, save_dir=save_dir)
        if "precision_recall_curve" in include:
            paths["precision_recall_curve"] = self.plot_precision_recall_curve(X_test, y_test, save_dir=save_dir)
        if "feature_importance" in include and feature_names is not None:
            paths["feature_importance"] = self.plot_feature_importance(feature_names, save_dir=save_dir)
        if "learning_curve" in include and X_train is not None and y_train is not None:
            paths["learning_curve"] = self.plot_learning_curve(X_train, y_train, save_dir=save_dir)
        if "decision_boundary" in include and X_train is not None and y_train is not None:
            paths["decision_boundary"] = self.plot_decision_boundary_2d(
                X_train, y_train, save_dir=save_dir, feature_names=feature_names
            )

        print(f"\nSaved {sum(1 for p in paths.values() if p)} plot(s) to '{save_dir}/'")
        return paths


# =============================================================================
# 6. ReportMixin — generate_report() -> PDF
# =============================================================================

class ReportMixin:
    """
    Mixin providing `generate_report()`. Assumes `self` has a fitted
    model (`PsyClassifierCore`) and, optionally, AutoML results
    (`self._last_results`) and visualization output.
    """

    def export_results(self, y_true, y_pred, filename: str = "results.csv"):
        """Export per-metric evaluation results to CSV for archiving."""
        metrics = self.evaluate(y_true, y_pred)
        pd.DataFrame([metrics]).to_csv(filename, index=False)
        return filename

    def generate_report(
        self,
        X_test,
        y_test,
        feature_names=None,
        output_path: str = "report.pdf",
        plot_dir: str = "psy_plots",
        leaderboard: Optional[pd.DataFrame] = None,
        dataset_name: str = "Dataset",
    ) -> str:
        """
        Build a PDF research report containing:
          - Title page / run metadata
          - Model summary
          - Evaluation metrics table
          - Model leaderboard (if AutoML was run)
          - Confusion matrix, ROC curve, PR curve, feature importance plots

        Returns the path to the generated PDF.
        """
        self._require_fitted()

        # Make sure the plots referenced exist.
        plots = self.visualize_results(
            X_test, y_test, feature_names=feature_names, save_dir=plot_dir,
            include=["confusion_matrix", "roc_curve", "precision_recall_curve", "feature_importance"],
        )

        y_pred = self.predict(X_test)
        metrics = self.evaluate(y_test, y_pred)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("PsyTitle", parent=styles["Title"], textColor=colors.HexColor("#1F3B57"))
        heading_style = ParagraphStyle("PsyHeading", parent=styles["Heading2"], textColor=colors.HexColor("#1F3B57"),
                                       spaceBefore=14, spaceAfter=8)
        body_style = styles["Normal"]

        story = []

        # --- Title ---
        story.append(Paragraph("PsyInsight AI — Model Evaluation Report", title_style))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Dataset: {dataset_name}", body_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
        story.append(Paragraph(f"Model: {self.model_name}", body_style))
        story.append(Spacer(1, 16))

        # --- Metrics table ---
        story.append(Paragraph("Evaluation Metrics", heading_style))
        metric_rows = [["Metric", "Value"]] + [[k, f"{v:.4f}"] for k, v in metrics.items()]
        metric_table = Table(metric_rows, colWidths=[2.5 * inch, 2.5 * inch])
        metric_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3B57")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ]))
        story.append(metric_table)

        accuracy = metrics["Accuracy"]
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"<b>Interpretation:</b> {self.interpretation(accuracy)}", body_style))

        # --- Leaderboard (if available) ---
        leaderboard = leaderboard if leaderboard is not None else getattr(self, "_last_results", None)
        if leaderboard is not None:
            story.append(Paragraph("Model Leaderboard", heading_style))
            display_cols = [c for c in leaderboard.columns if c != "_key"]
            lb = leaderboard[display_cols].copy()
            header = list(lb.columns)
            rows = [header] + [
                [row[c] if isinstance(row[c], str) else f"{row[c]:.3f}" for c in header]
                for _, row in lb.iterrows()
            ]
            lb_table = Table(rows, repeatRows=1)
            lb_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3B57")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]))
            story.append(lb_table)

        story.append(PageBreak())

        # --- Plots ---
        story.append(Paragraph("Diagnostic Plots", heading_style))
        for key, label in [
            ("confusion_matrix", "Confusion Matrix"),
            ("roc_curve", "ROC Curve"),
            ("precision_recall_curve", "Precision-Recall Curve"),
            ("feature_importance", "Feature Importance"),
        ]:
            path = plots.get(key)
            if path and os.path.exists(path):
                story.append(Paragraph(label, styles["Heading3"]))
                story.append(Image(path, width=4.5 * inch, height=4.1 * inch, kind="proportional"))
                story.append(Spacer(1, 10))

        doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
        doc.build(story)

        print(f"Report saved to {output_path}")
        return output_path


# =============================================================================
# 7. PsyClassifier — the public class, composed from all mixins above
# =============================================================================

class PsyClassifier(AutoMLMixin, VisualizationMixin, ReportMixin, PsyClassifierCore):
    """
    The single public entry point for PsyInsight AI's classification
    engine. Combines:

      - PsyClassifierCore   : model management, training, metrics
      - AutoMLMixin         : compare_models / leaderboard / best_model / auto_train
      - VisualizationMixin  : visualize_results and individual plots
      - ReportMixin         : generate_report (PDF)
    """
    pass


__all__ = ["PsyClassifier", "PsyClassifierCore"]
__version__ = "1.0.0"
