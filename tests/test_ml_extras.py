import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from psyinsight.ml import (
    PsyClustering,
    PsyFeatureSelector,
    PsyMetrics,
    PsyModelSelector,
    PsyRegressor,
)


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=150, n_features=5, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), y


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=150, n_features=5, noise=5, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), y


def test_regressor_auto_train(regression_data):
    X, y = regression_data
    reg = PsyRegressor()
    X_train, X_test, y_train, y_test = reg.split_data(X, y, test_size=0.2)
    result = reg.auto_train(X_train, X_test, y_train, y_test)
    assert "best_model" in result
    assert result["test_report"]["r2"] > -1


def test_regressor_feature_importance(regression_data):
    X, y = regression_data
    reg = PsyRegressor(model_name="random_forest")
    reg.fit(X, y)
    importance = reg.feature_importance()
    assert len(importance) == X.shape[1]


def test_feature_selector_auto_select(classification_data):
    X, y = classification_data
    df = X.assign(target=y)
    selector = PsyFeatureSelector(df)
    result = selector.auto_select("target", task="classification")
    assert len(result["recommended_features"]) > 0


def test_feature_selector_pca(classification_data):
    X, _ = classification_data
    selector = PsyFeatureSelector(X)
    reduced, info = selector.pca_reduce(n_components=2)
    assert reduced.shape[1] == 2
    assert info["n_components"] == 2


def test_model_selector_split_shapes(classification_data):
    X, y = classification_data
    selector = PsyModelSelector()
    X_train, X_val, X_test, y_train, y_val, y_test = selector.train_val_test_split(X, y)
    assert len(X_train) + len(X_val) + len(X_test) == len(X)


def test_metrics_classification_report(classification_data):
    X, y = classification_data
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(random_state=42).fit(X, y)
    preds = model.predict(X)
    report = PsyMetrics.classification_report(y, preds)
    assert 0 <= report["accuracy"] <= 1


def test_metrics_cohens_d_and_alpha():
    a = np.random.normal(0, 1, 100)
    b = np.random.normal(1, 1, 100)
    d = PsyMetrics.cohens_d(a, b)
    assert PsyMetrics.interpret_cohens_d(d) in {"negligible", "small", "medium", "large"}

    items = pd.DataFrame(np.random.rand(50, 5))
    alpha = PsyMetrics.cronbach_alpha(items)
    assert isinstance(PsyMetrics.interpret_alpha(alpha), str)


def test_clustering_smoke():
    X = pd.DataFrame(np.random.rand(80, 3), columns=["a", "b", "c"])
    engine = PsyClustering()
    engine.set_model("kmeans", n_clusters=2)
    labels = engine.fit_predict(X)
    assert len(labels) == len(X)
    report = PsyMetrics.clustering_report(X.values, labels)
    assert report["n_clusters"] >= 1
