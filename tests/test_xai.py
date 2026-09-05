import pandas as pd
from sklearn.datasets import make_classification

from psyinsight.ml import PsyClassifier
from psyinsight.xai import PsyExplainer


def _fitted_classifier():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    clf = PsyClassifier()
    clf.set_model("random_forest")
    X_train, X_test, y_train, y_test = clf.split_data(df, y)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test, df.columns.tolist()


def test_global_feature_importance():
    clf, X_test, y_test, names = _fitted_classifier()
    explainer = PsyExplainer(clf, feature_names=names)
    importance = explainer.global_feature_importance()
    assert len(importance) == len(names)


def test_permutation_importance():
    clf, X_test, y_test, names = _fitted_classifier()
    explainer = PsyExplainer(clf, feature_names=names)
    result = explainer.permutation_importance(X_test, y_test, n_repeats=3)
    assert len(result) == len(names)


def test_partial_dependence():
    clf, X_test, y_test, names = _fitted_classifier()
    explainer = PsyExplainer(clf, feature_names=names)
    pdp = explainer.partial_dependence(X_test, names[0])
    assert "avg_prediction" in pdp.columns


def test_local_explanation():
    clf, X_test, y_test, names = _fitted_classifier()
    explainer = PsyExplainer(clf, feature_names=names)
    local = explainer.explain_instance(X_test, 0, n_samples=100)
    assert set(local["feature"]) == set(names)


def test_full_report_runs_without_crashing():
    clf, X_test, y_test, names = _fitted_classifier()
    explainer = PsyExplainer(clf, feature_names=names)
    report = explainer.full_report(X_test, y_test)
    assert "global_importance" in report
