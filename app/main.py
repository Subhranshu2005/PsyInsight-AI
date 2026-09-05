"""
PsyInsight AI — Streamlit Application
========================================

The unified research-assistant UI: upload data, validate & clean it, run
descriptive/inferential statistics and probability calculations, train and
compare ML models, explain them, mine open-ended text responses, check
survey response-pattern quality with automata, and export a research
report — all from one app.

Run with:
    streamlit run app/main.py

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make `psyinsight` importable whether the app is launched from the repo
# root or from within `app/`.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from psyinsight.automata import ResponsePatternAnalyzer
from psyinsight.ml import (
    PsyClassifier,
    PsyClustering,
    PsyFeatureSelector,
    PsyMetrics,
    PsyModelSelector,
    PsyRegressor,
)
from psyinsight.preprocessing import DataCleaner, DataLoader, DataValidator
from psyinsight.probability.distributions import ProbabilityEngine
from psyinsight.reporting import ResearchReportGenerator
from psyinsight.statistics.descriptive import DescriptiveStatistics
from psyinsight.statistics.inferential import InferentialStatistics
from psyinsight.transformers import HF_AVAILABLE, TextAnalyzer
from psyinsight.visualization.visualizer import Visualizer
from psyinsight.xai import PsyExplainer

st.set_page_config(page_title="PsyInsight AI", page_icon="🧠", layout="wide")

# Guardrails for uploaded datasets: keep the app responsive and prevent a
# single oversized/degenerate file from exhausting memory or hanging the
# server for every user sharing the process.
MAX_UPLOAD_MB = 50
MAX_ROWS = 200_000
MAX_COLS = 500

# Minimum seconds between expensive operations (model training, clustering,
# benchmarking) per browser session, to keep one user from starving CPU for
# everyone else sharing this process by mashing a button in a loop. This is
# *not* a substitute for infrastructure-level rate limiting (a reverse
# proxy or WAF), which this application-level check cannot provide.
RATE_LIMIT_SECONDS = 3.0


# =============================================================================
# Optional access gate
# =============================================================================
#
# If an operator sets an `app_password` in Streamlit secrets
# (.streamlit/secrets.toml or the deployment's secrets manager), the app
# requires it before showing any content. If no password is configured
# (the default -- e.g. for local research/coursework use), the gate is
# skipped entirely so existing usage is unaffected.
#
# This is a basic shared-password gate, not a full auth system: there's no
# per-user identity, no session expiry beyond the browser tab, and the
# password travels in the request like any Streamlit widget input (relying
# on the deployment's own TLS termination for transport security, which is
# outside this application's control).


def _configured_password() -> str | None:
    try:
        return st.secrets.get("app_password")
    except Exception:  # noqa: BLE001 -- no secrets.toml at all is normal/expected
        return None


def check_access_gate() -> bool:
    """Return True if the app should render for this session."""
    required_password = _configured_password()
    if not required_password:
        return True  # no password configured -- gate is a no-op

    if st.session_state.get("_authenticated"):
        return True

    st.title("🧠 PsyInsight AI")
    st.info("This deployment is password-protected.")
    entered = st.text_input("Password", type="password")
    if st.button("Unlock"):
        if entered == required_password:
            st.session_state["_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def rate_limit_ok(action_key: str, min_interval: float = RATE_LIMIT_SECONDS) -> bool:
    """
    Return True if `action_key` hasn't been triggered in this session within
    the last `min_interval` seconds; otherwise show a warning and return
    False. Meant to guard expensive actions (training, clustering,
    benchmarking) behind their button click, e.g.::

        if st.button("Train") and rate_limit_ok("train"):
            ...
    """
    now = time.monotonic()
    last_times = st.session_state.setdefault("_last_action_time", {})
    last = last_times.get(action_key, 0.0)
    if now - last < min_interval:
        wait = min_interval - (now - last)
        st.warning(f"Please wait {wait:.1f}s before doing that again.")
        return False
    last_times[action_key] = now
    return True


def sniff_mismatched_upload(uploaded) -> str | None:
    """
    Cheaply check that an uploaded file's content plausibly matches what
    its extension claims, beyond trusting the filename alone. Returns an
    error message if there's a clear mismatch, else None.

    This isn't a full malware/content scanner -- it catches the common
    "renamed file" confusion case (e.g. an .xlsx saved with a .csv
    extension, which would otherwise reach pd.read_csv and fail with a
    confusing binary-garbage error, or worse, be misinterpreted).
    """
    name = uploaded.name.lower()
    header = uploaded.getvalue()[:8]
    uploaded.seek(0)

    is_zip_magic = header[:2] == b"PK"  # .xlsx/.xls(x)/.docx are zip containers
    is_ole_magic = header[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .xls/.doc

    if name.endswith(".csv") or name.endswith(".json"):
        if is_zip_magic or is_ole_magic:
            return (
                f"'{uploaded.name}' has a .csv/.json extension but its content looks "
                "like an Excel file. Please check the file and re-upload it with the "
                "correct extension (or re-export it as CSV/JSON)."
            )
    return None


# =============================================================================
# Session-state helpers
# =============================================================================


def _init_state():
    defaults = {
        "df": None,
        "df_name": None,
        "report": ResearchReportGenerator(title="PsyInsight AI Research Report"),
        "trained_model": None,
        "trained_model_kind": None,
        "trained_features": None,
        "trained_target": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _sample_dataset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "datasets" / "sample_psychology_data.csv"


# =============================================================================
# Sidebar: data loading
# =============================================================================


def sidebar_data_loader():
    st.sidebar.title("🧠 PsyInsight AI")
    st.sidebar.caption("An AI-powered psychology research assistant")
    st.sidebar.divider()
    st.sidebar.subheader("📂 Dataset")

    uploaded = st.sidebar.file_uploader("Upload CSV / Excel / JSON", type=["csv", "xlsx", "xls", "json"])
    use_sample = st.sidebar.button("Load bundled sample dataset")

    if uploaded is not None:
        size_mb = uploaded.size / (1024 * 1024)
        mismatch = sniff_mismatched_upload(uploaded)
        if size_mb > MAX_UPLOAD_MB:
            st.sidebar.error(
                f"'{uploaded.name}' is {size_mb:.1f} MB, which is over the "
                f"{MAX_UPLOAD_MB} MB limit for this app. Please upload a smaller file "
                "or a sample of your data."
            )
        elif mismatch:
            st.sidebar.error(mismatch)
        else:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                elif uploaded.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(uploaded)
                elif uploaded.name.endswith(".json"):
                    df = pd.read_json(uploaded)
                else:
                    raise ValueError(
                        "Unsupported file type. Please upload a .csv, .xlsx, .xls, or .json file."
                    )
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"Could not load file: {exc}")
            else:
                if df.empty:
                    st.sidebar.error(f"'{uploaded.name}' contains no data rows.")
                elif df.shape[1] > MAX_COLS:
                    st.sidebar.error(
                        f"'{uploaded.name}' has {df.shape[1]} columns, which is over the "
                        f"{MAX_COLS}-column limit for this app."
                    )
                else:
                    truncated = False
                    if df.shape[0] > MAX_ROWS:
                        df = df.head(MAX_ROWS)
                        truncated = True
                    st.session_state.df = df
                    st.session_state.df_name = uploaded.name
                    st.sidebar.success(
                        f"Loaded {uploaded.name} ({df.shape[0]} rows x {df.shape[1]} cols)"
                    )
                    if truncated:
                        st.sidebar.warning(
                            f"Dataset was truncated to the first {MAX_ROWS:,} rows to keep "
                            "the app responsive."
                        )

    if use_sample:
        sample_path = _sample_dataset_path()
        try:
            df = DataLoader.load(str(sample_path))
            st.session_state.df = df
            st.session_state.df_name = sample_path.name
            st.sidebar.success(f"Loaded sample dataset ({df.shape[0]} rows x {df.shape[1]} cols)")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Could not load sample dataset: {exc}")

    if st.session_state.df is not None:
        st.sidebar.info(f"Active dataset: **{st.session_state.df_name}**")
        st.sidebar.write(f"{st.session_state.df.shape[0]} rows x {st.session_state.df.shape[1]} columns")

    st.sidebar.divider()
    st.sidebar.caption("PsyInsight AI v0.2 -- open-source research platform")


# =============================================================================
# Tab: Overview / Preprocessing
# =============================================================================


def tab_preprocessing():
    st.header("Data Preprocessing & Validation")
    df = st.session_state.df
    if df is None:
        st.info("Upload a dataset or load the bundled sample from the sidebar to get started.")
        return

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Validation Report")
    validator = DataValidator(df)
    report = validator.validate()
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", report["Dataset Shape"][0])
    col2.metric("Columns", report["Dataset Shape"][1])
    col3.metric("Duplicate Rows", report["Duplicate Rows"])

    with st.expander("Missing values by column"):
        st.dataframe(
            pd.DataFrame(
                {
                    "missing_count": report["Missing Values"],
                    "missing_pct": report["Missing Value Percentage"],
                    "dtype": report["Data Types"],
                }
            ),
            use_container_width=True,
        )

    st.subheader("Cleaning")
    c1, c2, c3 = st.columns(3)
    if c1.button("Remove duplicate rows"):
        cleaner = DataCleaner(df)
        st.session_state.df = cleaner.remove_duplicates()
        st.success("Duplicates removed.")
        st.rerun()
    fill_value = c2.text_input("Fill missing values with", value="0")
    if c2.button("Fill missing values"):
        cleaner = DataCleaner(df)
        try:
            value: object = float(fill_value)
        except ValueError:
            value = fill_value
        st.session_state.df = cleaner.fill_missing_values(value=value)
        st.success("Missing values filled.")
        st.rerun()
    if c3.button("Standardize column names"):
        cleaner = DataCleaner(df)
        st.session_state.df = cleaner.rename_columns()
        st.success("Column names standardized (lowercase, underscores).")
        st.rerun()

    if st.button("Save validation report to research report"):
        st.session_state.report.add_section("Data Validation", report)
        st.success("Added to research report.")


# =============================================================================
# Tab: Statistics
# =============================================================================


def tab_statistics():
    st.header("Statistics Engine")
    df = st.session_state.df
    if df is None:
        st.info("Load a dataset first.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.subheader("Descriptive Statistics")
    desc = DescriptiveStatistics(df)
    st.dataframe(desc.summary(), use_container_width=True)

    with st.expander("Skewness, kurtosis & IQR"):
        st.dataframe(
            pd.DataFrame(
                {
                    "skewness": desc.skewness(),
                    "kurtosis": desc.kurtosis(),
                    "IQR": desc.interquartile_range(),
                }
            ),
            use_container_width=True,
        )

    if st.button("Save descriptive stats to research report"):
        st.session_state.report.add_section("Descriptive Statistics", desc.summary())
        st.success("Added to research report.")

    st.divider()
    st.subheader("Inferential Statistics")
    infer = InferentialStatistics(df)

    test = st.selectbox(
        "Choose a test",
        [
            "One-sample t-test",
            "Independent t-test",
            "Paired t-test",
            "Pearson correlation",
            "Spearman correlation",
            "Chi-square test",
            "One-way ANOVA",
        ],
    )

    result = None
    if test == "One-sample t-test":
        col = st.selectbox("Column", numeric_cols, key="t1_col")
        pop_mean = st.number_input("Population mean (H0)", value=0.0)
        if st.button("Run test"):
            result = infer.one_sample_t_test(col, pop_mean)
    elif test == "Independent t-test":
        value_col = st.selectbox("Value column", numeric_cols, key="ti_value")
        group_col = st.selectbox("Group column", df.columns.tolist(), key="ti_group")
        group_values = df[group_col].dropna().unique().tolist()
        if len(group_values) < 2:
            st.warning("The selected group column needs at least 2 distinct values.")
        else:
            g1 = st.selectbox("Group A value", group_values, key="ti_g1")
            g2 = st.selectbox(
                "Group B value", group_values, key="ti_g2", index=min(1, len(group_values) - 1)
            )
            if st.button("Run test"):
                if g1 == g2:
                    st.warning("Group A and Group B must be different values.")
                else:
                    result = infer.independent_t_test(value_col, group_col, g1, g2)
    elif test == "Paired t-test":
        c1 = st.selectbox("Column A", numeric_cols, key="tp_a")
        c2 = st.selectbox("Column B", numeric_cols, key="tp_b", index=min(1, len(numeric_cols) - 1))
        if st.button("Run test"):
            result = infer.paired_t_test(c1, c2)
    elif test == "Pearson correlation":
        c1 = st.selectbox("Column A", numeric_cols, key="pc_a")
        c2 = st.selectbox("Column B", numeric_cols, key="pc_b", index=min(1, len(numeric_cols) - 1))
        if st.button("Run test"):
            result = infer.pearson_correlation(c1, c2)
    elif test == "Spearman correlation":
        c1 = st.selectbox("Column A", numeric_cols, key="sc_a")
        c2 = st.selectbox("Column B", numeric_cols, key="sc_b", index=min(1, len(numeric_cols) - 1))
        if st.button("Run test"):
            result = infer.spearman_correlation(c1, c2)
    elif test == "Chi-square test":
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist() or df.columns.tolist()
        c1 = st.selectbox("Categorical column A", cat_cols, key="chi_a")
        c2 = st.selectbox("Categorical column B", cat_cols, key="chi_b", index=min(1, len(cat_cols) - 1))
        if st.button("Run test"):
            result = infer.chi_square_test(c1, c2)
    elif test == "One-way ANOVA":
        group_col = st.selectbox("Group column", df.columns.tolist(), key="anova_g")
        value_col = st.selectbox("Value column", numeric_cols, key="anova_v")
        if st.button("Run test"):
            result = infer.one_way_anova(value_col, group_col)

    if result is not None:
        st.json(result if isinstance(result, dict) else str(result))
        if st.button("Save this test result to research report"):
            st.session_state.report.add_section(f"Inferential Test: {test}", result)
            st.success("Added to research report.")

    st.divider()
    st.subheader("Reliability (Cronbach's alpha)")
    st.caption("Select the items belonging to one psychometric scale/questionnaire.")
    scale_cols = st.multiselect("Scale items", numeric_cols, key="alpha_cols")
    if scale_cols and len(scale_cols) >= 2:
        alpha = PsyMetrics.cronbach_alpha(df[scale_cols])
        st.metric("Cronbach's alpha", f"{alpha:.3f}", PsyMetrics.interpret_alpha(alpha))


# =============================================================================
# Tab: Probability
# =============================================================================


def tab_probability():
    st.header("Probability Engine")
    dist = st.selectbox("Distribution", ["Normal", "Binomial", "Poisson"])

    if dist == "Normal":
        c1, c2, c3 = st.columns(3)
        mean = c1.number_input("Mean", value=0.0)
        std = c2.number_input("Std deviation", value=1.0, min_value=0.0001)
        x = c3.number_input("x", value=0.0)
        mode = st.radio("Compute", ["P(X <= x)", "P(X >= x)", "PDF at x"], horizontal=True)
        if st.button("Compute"):
            if mode == "P(X <= x)":
                st.success(ProbabilityEngine.normal_probability(x, mean, std, mode="cdf"))
            elif mode == "P(X >= x)":
                st.success(ProbabilityEngine.normal_probability(x, mean, std, mode="sf"))
            else:
                st.success(ProbabilityEngine.normal_probability(x, mean, std, mode="pdf"))

    elif dist == "Binomial":
        c1, c2, c3 = st.columns(3)
        n = c1.number_input("n (trials)", value=10, min_value=1, step=1)
        p = c2.number_input("p (success probability)", value=0.5, min_value=0.0, max_value=1.0)
        k = c3.number_input("k (successes)", value=5, min_value=0, step=1)
        if st.button("Compute"):
            st.success(ProbabilityEngine.binomial_probability(int(k), int(n), p))

    elif dist == "Poisson":
        c1, c2 = st.columns(2)
        lam = c1.number_input("Lambda (rate)", value=3.0, min_value=0.0001)
        k = c2.number_input("k (events)", value=2, min_value=0, step=1)
        if st.button("Compute"):
            st.success(ProbabilityEngine.poisson_probability(int(k), lam))


# =============================================================================
# Tab: Machine Learning
# =============================================================================


def tab_machine_learning():
    st.header("Machine Learning")
    df = st.session_state.df
    if df is None:
        st.info("Load a dataset first.")
        return

    task = st.radio("Task", ["Classification", "Regression", "Clustering"], horizontal=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if task in ("Classification", "Regression"):
        target = st.selectbox("Target column", df.columns.tolist())
        feature_options = [c for c in numeric_cols if c != target]
        features = st.multiselect("Feature columns", feature_options, default=feature_options)

        if not features:
            st.warning("Select at least one feature column.")
            return

        working = df[features + [target]].dropna()
        X, y = working[features], working[target]

        c1, c2 = st.columns(2)
        test_size = c1.slider("Test size", 0.1, 0.5, 0.2)
        run = c2.button("Auto-train & compare models")

        if run and rate_limit_ok("ml_train"):
            if len(working) < 5:
                st.error(
                    f"Only {len(working)} complete row(s) remain after dropping missing "
                    "values -- need at least a handful of rows to train and evaluate a model."
                )
            else:
                try:
                    engine = PsyClassifier() if task == "Classification" else PsyRegressor()
                    X_train, X_test, y_train, y_test = engine.split_data(X, y, test_size=test_size)
                    if len(X_test) == 0 or len(X_train) == 0:
                        raise ValueError(
                            "The train/test split left an empty split -- lower the test "
                            "size or add more rows."
                        )
                    with st.spinner("Training and comparing candidate models..."):
                        result = engine.auto_train(X_train, X_test, y_train, y_test)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Training failed: {exc}")
                else:
                    st.success(f"Best model: {result['best_model']}")
                    st.subheader("Leaderboard")
                    st.dataframe(result["leaderboard"], use_container_width=True)
                    st.subheader("Test-set report")
                    st.json(result["test_report"])

                    st.session_state.trained_model = engine
                    st.session_state.trained_model_kind = task.lower()
                    st.session_state.trained_features = features
                    st.session_state.trained_target = target
                    st.session_state["_X_test"] = X_test
                    st.session_state["_y_test"] = y_test

        if st.session_state.trained_model is not None and st.button("Save ML results to research report"):
            st.session_state.report.add_section(f"Machine Learning ({task})", {"model_target": target})
            st.success("Added to research report.")

    else:  # Clustering
        feature_options = numeric_cols
        features = st.multiselect("Feature columns", feature_options, default=feature_options)
        n_clusters = st.slider("Number of clusters (KMeans)", 2, 10, 3)

        if features and st.button("Run clustering") and rate_limit_ok("clustering"):
            working = df[features].dropna()

            if len(working) < 2:
                st.error(
                    f"Only {len(working)} complete row(s) remain after dropping missing "
                    "values in the selected features -- need at least 2 to cluster."
                )
            elif len(working) < n_clusters:
                st.error(
                    f"You asked for {n_clusters} clusters but only {len(working)} complete "
                    "row(s) are available. Choose fewer clusters or more/complete data."
                )
            else:
                try:
                    cluster_engine = PsyClustering()
                    cluster_engine.set_model("kmeans", n_clusters=n_clusters)
                    labels = cluster_engine.fit_predict(working)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Clustering failed: {exc}")
                else:
                    report = PsyMetrics.clustering_report(working.values, np.asarray(labels))
                    st.json(report)
                    result_df = working.copy()
                    result_df["cluster"] = np.asarray(labels)
                    st.dataframe(result_df.head(50), use_container_width=True)


# =============================================================================
# Tab: Feature Selection
# =============================================================================


def tab_feature_selection():
    st.header("Feature Selection")
    df = st.session_state.df
    if df is None:
        st.info("Load a dataset first.")
        return

    target = st.selectbox("Target column", df.columns.tolist(), key="fs_target")
    task = st.radio("Task", ["classification", "regression"], horizontal=True, key="fs_task")

    selector = PsyFeatureSelector(df)

    if st.button("Low-variance features"):
        st.write(selector.low_variance_features())
    if st.button("Highly correlated (redundant) features"):
        st.write(selector.highly_correlated_features())
    if st.button("Run model-based importance ranking"):
        st.dataframe(selector.model_based_importance(target, task=task), use_container_width=True)
    if st.button("Run full auto-select pipeline"):
        with st.spinner("Selecting features..."):
            result = selector.auto_select(target, task=task)
        st.write("Recommended features:", result["recommended_features"])
        st.dataframe(result["importance_ranking"], use_container_width=True)


# =============================================================================
# Tab: Explainable AI
# =============================================================================


def tab_xai():
    st.header("Explainable AI")
    model = st.session_state.trained_model
    if model is None or "_X_test" not in st.session_state:
        st.info("Train a classification or regression model in the Machine Learning tab first.")
        return

    X_test = st.session_state["_X_test"]
    y_test = st.session_state["_y_test"]
    explainer = PsyExplainer(model, feature_names=st.session_state.trained_features)

    st.subheader("Global feature importance")
    try:
        st.dataframe(explainer.global_feature_importance(), use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Native importance unavailable: {exc}")

    st.subheader("Permutation importance")
    if st.button("Compute permutation importance"):
        with st.spinner("Permuting features..."):
            st.dataframe(explainer.permutation_importance(X_test, y_test), use_container_width=True)

    st.subheader("Local explanation for one respondent")
    idx = st.number_input("Row index (within test set)", 0, max(len(X_test) - 1, 0), 0)
    if st.button("Explain this prediction"):
        st.dataframe(explainer.explain_instance(X_test, int(idx)), use_container_width=True)


# =============================================================================
# Tab: Text / NLP
# =============================================================================


def tab_text_analysis():
    st.header("Text Analysis (Transformers / NLP)")
    if not HF_AVAILABLE:
        st.warning(
            "transformers/torch are not installed -- running in lightweight "
            "lexicon/TF-IDF fallback mode. Install with `pip install transformers torch` "
            "for pretrained-model accuracy."
        )

    df = st.session_state.df
    text_col = None
    if df is not None:
        text_like = [c for c in df.columns if df[c].dtype == object]
        if text_like:
            text_col = st.selectbox("Use a text column from the dataset (optional)", ["<paste manually>"] + text_like)

    if text_col and text_col != "<paste manually>":
        texts = df[text_col].dropna().astype(str).tolist()
        st.caption(f"Using {len(texts)} responses from column '{text_col}'.")
    else:
        raw = st.text_area(
            "Paste one open-ended response per line",
            "I feel really anxious about my exams lately.\nI'm grateful for my friends and family support.\nToday was just an ordinary, uneventful day.",
            height=150,
        )
        texts = [line for line in raw.splitlines() if line.strip()]

    analyzer = TextAnalyzer()

    if st.button("Analyze sentiment"):
        with st.spinner("Analyzing..."):
            summary = analyzer.sentiment_summary(texts)
        st.write("Sentiment distribution:", summary["distribution"])
        st.dataframe(summary["detail"], use_container_width=True)

    if st.button("Extract keywords"):
        st.dataframe(analyzer.extract_keywords(texts, top_k=15), use_container_width=True)

    n_themes = st.slider("Number of themes to discover", 2, 6, 3)
    if st.button("Discover themes"):
        with st.spinner("Clustering responses..."):
            themes = analyzer.discover_themes(texts, n_themes=n_themes)
        for cluster_id, info in themes["themes"].items():
            st.write(f"Theme {cluster_id} ({info['n_responses']} responses): {', '.join(info['keywords'])}")


# =============================================================================
# Tab: Automata (response-pattern quality)
# =============================================================================


def tab_automata():
    st.header("Automata Theory -- Survey Response-Pattern Quality")
    df = st.session_state.df
    if df is None:
        st.info("Load a dataset first.")
        return

    analyzer = ResponsePatternAnalyzer(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.subheader("Likert-scale validity check")
    col = st.selectbox("Column", numeric_cols, key="dfa_col")
    c1, c2 = st.columns(2)
    min_v = c1.number_input("Scale minimum", value=1, step=1)
    max_v = c2.number_input("Scale maximum", value=5, step=1)
    if st.button("Validate column"):
        st.json(analyzer.validate_likert_column(col, int(min_v), int(max_v)))

    st.subheader("Straight-lining (careless responding) detection")
    scale_cols = st.multiselect("Likert-scale item columns (same scale)", numeric_cols, key="sl_cols")
    run_limit = st.slider("Flag if identical answers repeat more than", 2, 15, 5)
    if scale_cols and st.button("Detect straight-lining"):
        result = analyzer.detect_straight_lining(scale_cols, run_limit=run_limit)
        n_flagged = int(result["flagged_straight_lining"].sum())
        st.metric("Respondents flagged", n_flagged, f"out of {len(result)}")
        st.dataframe(result, use_container_width=True)


# =============================================================================
# Tab: Report
# =============================================================================


def tab_report():
    st.header("Research Report Generator")
    report = st.session_state.report

    st.write(f"Sections added so far: {len(report.sections)}")
    for section in report.sections:
        st.write(f"- {section['heading']}")

    if st.button("Clear report"):
        st.session_state.report = ResearchReportGenerator(title="PsyInsight AI Research Report")
        st.rerun()

    title = st.text_input("Report title", value=report.title)
    report.title = title

    fmt = st.radio("Export format", ["Markdown", "HTML", "JSON"], horizontal=True)
    if fmt == "Markdown":
        content = report.to_markdown()
        st.download_button("Download report.md", content, file_name="psyinsight_report.md")
        st.code(content, language="markdown")
    elif fmt == "HTML":
        content = report.to_html()
        st.download_button("Download report.html", content, file_name="psyinsight_report.html")
        st.components.v1.html(content, height=500, scrolling=True)
    else:
        content = report.to_json()
        st.download_button("Download report.json", content, file_name="psyinsight_report.json")
        st.code(content, language="json")


# =============================================================================
# Tab: Visualization
# =============================================================================


def tab_visualization():
    st.header("Visualization")
    df = st.session_state.df
    if df is None:
        st.info("Load a dataset first.")
        return

    viz = Visualizer(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    kind = st.selectbox("Chart type", ["Histogram", "Box plot", "Scatter plot", "Correlation heatmap"])

    if not numeric_cols and kind != "Correlation heatmap":
        st.info("No numeric columns available to plot.")
        return

    if kind == "Histogram":
        col = st.selectbox("Column", numeric_cols)
        bins = st.slider("Bins", 5, 50, 10)
        if st.button("Plot"):
            st.pyplot(viz.histogram(col, bins=bins))
    elif kind == "Box plot":
        col = st.selectbox("Column", numeric_cols)
        if st.button("Plot"):
            st.pyplot(viz.box_plot(col))
    elif kind == "Scatter plot":
        c1 = st.selectbox("X", numeric_cols, key="sx")
        c2 = st.selectbox("Y", numeric_cols, key="sy", index=min(1, len(numeric_cols) - 1))
        if st.button("Plot"):
            st.pyplot(viz.scatter_plot(c1, c2))
    else:
        if st.button("Plot"):
            st.pyplot(viz.correlation_heatmap())


# =============================================================================
# Main
# =============================================================================


def main():
    _init_state()

    if not check_access_gate():
        return  # gate not passed yet -- password prompt is already rendered

    sidebar_data_loader()

    st.title("PsyInsight AI")
    st.caption(
        "Data Science - Statistics - Probability - Machine Learning - Deep Learning - "
        "NLP - Automata Theory - Explainable AI, unified for psychology research."
    )

    tabs = st.tabs(
        [
            "Preprocessing",
            "Statistics",
            "Probability",
            "Machine Learning",
            "Feature Selection",
            "Explainable AI",
            "Text / NLP",
            "Automata",
            "Visualization",
            "Report",
        ]
    )

    tab_functions = [
        tab_preprocessing,
        tab_statistics,
        tab_probability,
        tab_machine_learning,
        tab_feature_selection,
        tab_xai,
        tab_text_analysis,
        tab_automata,
        tab_visualization,
        tab_report,
    ]

    for tab, tab_fn in zip(tabs, tab_functions):
        with tab:
            # Isolate each tab so an unexpected error in one panel (e.g. an
            # edge case in the uploaded data) can't take down the whole app
            # or leak an internal traceback -- it's reported inline instead.
            try:
                tab_fn()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong in this section: {exc}")
                st.caption("Try a different selection, or reload the dataset from the sidebar.")


if __name__ == "__main__":
    main()
