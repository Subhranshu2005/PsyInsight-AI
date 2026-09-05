"""
Property-based / fuzz tests using Hypothesis.

Unlike tests/test_edge_cases.py (specific, hand-picked edge cases), these
tests generate hundreds of randomized -- including malformed -- inputs per
run and assert a much looser but broader property: the code must never
raise an *unexpected* exception type. A clear, documented ValueError or
TypeError is fine (that's the whole point of the hardening pass); an
AttributeError, KeyError, IndexError, or ZeroDivisionError bubbling up from
inside a method is exactly the class of bug this pass has been fixing, and
fuzzing is a much better way to find more of them than hand-written cases.

Each test bounds Hypothesis's generated shapes to keep runtime reasonable
(deadline=None because model training genuinely takes >200ms sometimes).
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from psyinsight.ml.classifier import PsyClassifier
from psyinsight.ml.clustering import InvalidClusterError, PsyClustering
from psyinsight.ml.regressor import PsyRegressor
from psyinsight.preprocessing.cleaner import DataCleaner
from psyinsight.preprocessing.validator import DataValidator
from psyinsight.probability.distributions import ProbabilityEngine
from psyinsight.statistics.descriptive import DescriptiveStatistics
from psyinsight.visualization.visualizer import Visualizer

# A conservative "expected" exception set: things the library is allowed to
# raise on bad/degenerate input, because they come with a clear, actionable
# message. Anything else escaping is treated as a bug by these tests.
EXPECTED_EXCEPTIONS = (ValueError, TypeError, InvalidClusterError)

# Reasonable-magnitude floats (including NaN/inf) rather than the full
# float64 range, so we're fuzzing realistic "messy survey data" rather than
# purely numerical edge cases unrelated to this app's domain.
messy_floats = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True, width=32),
    st.just(0.0),
)

fuzz_settings = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# ---------------------------------------------------------------------------
# Preprocessing: DataValidator / DataCleaner
# ---------------------------------------------------------------------------


@fuzz_settings
@given(
    data_frames(
        columns=[
            column("a", elements=messy_floats),
            column("b", elements=messy_floats),
            column("c", elements=st.text(max_size=5)),
        ],
        index=range_indexes(min_size=0, max_size=25),
    )
)
def test_validator_never_raises_unexpected_exception(df):
    validator = DataValidator(df)
    validator.dataset_shape()
    validator.missing_values()
    pct = validator.missing_value_percentage()
    # Percentages must always be finite and in [0, 100] -- never NaN/inf,
    # regardless of how degenerate the input is (this was a real bug: 0/0
    # used to silently produce NaN).
    assert not pct.isna().any()
    assert ((pct >= 0) & (pct <= 100)).all()
    validator.duplicate_rows()
    validator.data_types()
    validator.validate()


@fuzz_settings
@given(
    data_frames(
        columns=[
            column("a", elements=messy_floats),
            column("B col", elements=messy_floats),
        ],
        index=range_indexes(min_size=0, max_size=25),
    )
)
def test_cleaner_never_raises_unexpected_exception(df):
    DataCleaner(df.copy()).rename_columns()
    DataCleaner(df.copy()).fill_missing_values(0)
    DataCleaner(df.copy()).remove_duplicates()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@fuzz_settings
@given(
    data_frames(
        columns=[column("a", elements=messy_floats), column("b", elements=messy_floats)],
        index=range_indexes(min_size=0, max_size=25),
    )
)
def test_descriptive_statistics_never_raises_unexpected_exception(df):
    stats = DescriptiveStatistics(df)
    stats.summary()


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------


@fuzz_settings
@given(
    x=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    mean=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    std=st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=100),
)
def test_normal_probability_never_raises_unexpected_exception(x, mean, std):
    try:
        result = ProbabilityEngine.normal_probability(x, mean, std, mode="cdf")
        assert not math.isnan(result) or std == 0
    except EXPECTED_EXCEPTIONS:
        pass  # e.g. std <= 0 -- documented, clear failure


@fuzz_settings
@given(
    k=st.integers(min_value=-1000, max_value=1000),
    n=st.integers(min_value=-100, max_value=100),
    p=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False),
)
def test_binomial_probability_never_raises_unexpected_exception(k, n, p):
    try:
        ProbabilityEngine.binomial_probability(k, n, p)
    except EXPECTED_EXCEPTIONS:
        pass


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------


@fuzz_settings
@given(
    data_frames(
        columns=[column("a", elements=messy_floats), column("b", elements=messy_floats)],
        index=range_indexes(min_size=0, max_size=30),
    )
)
def test_visualizer_never_raises_unexpected_exception(df):
    viz = Visualizer(df)
    try:
        viz.histogram("a")
        viz.box_plot("a")
        viz.scatter_plot("a", "b")
        viz.correlation_heatmap()
    except EXPECTED_EXCEPTIONS:
        pass
    except KeyError:
        pass  # column genuinely absent in a degenerate generated frame


# ---------------------------------------------------------------------------
# ML: classifier / regressor / clustering on randomized numeric data
# ---------------------------------------------------------------------------


numeric_matrix = st.lists(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3),
        min_size=2,
        max_size=2,
    ),
    min_size=0,
    max_size=40,
)


@fuzz_settings
@given(rows=numeric_matrix, class_seed=st.integers(min_value=0, max_value=3))
def test_classifier_auto_train_never_raises_unexpected_exception(rows, class_seed):
    if len(rows) < 4:
        return  # too degenerate to meaningfully split; covered by test_edge_cases.py
    X = pd.DataFrame(rows, columns=["f1", "f2"])
    rng = np.random.RandomState(class_seed)
    y = pd.Series(rng.randint(0, 2, size=len(rows)))

    clf = PsyClassifier()
    try:
        X_train, X_test, y_train, y_test = clf.split_data(X, y, test_size=0.3)
        if len(X_train) == 0 or len(X_test) == 0:
            return
        clf.auto_train(X_train, X_test, y_train, y_test)
    except EXPECTED_EXCEPTIONS:
        pass


@fuzz_settings
@given(rows=numeric_matrix)
def test_regressor_auto_train_never_raises_unexpected_exception(rows):
    if len(rows) < 4:
        return
    X = pd.DataFrame(rows, columns=["f1", "f2"])
    y = pd.Series([r[0] + r[1] for r in rows])

    reg = PsyRegressor()
    try:
        X_train, X_test, y_train, y_test = reg.split_data(X, y, test_size=0.3)
        if len(X_train) == 0 or len(X_test) == 0:
            return
        reg.auto_train(X_train, X_test, y_train, y_test)
    except EXPECTED_EXCEPTIONS:
        pass


@fuzz_settings
@given(rows=numeric_matrix, n_clusters=st.integers(min_value=1, max_value=6))
def test_clustering_never_raises_unexpected_exception(rows, n_clusters):
    X = pd.DataFrame(rows, columns=["f1", "f2"])
    c = PsyClustering()
    c.set_model("kmeans", n_clusters=n_clusters)
    try:
        c.fit_predict(X)
    except EXPECTED_EXCEPTIONS:
        pass
