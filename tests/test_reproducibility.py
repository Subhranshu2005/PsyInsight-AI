"""
Reproducibility tests (robustness review item #5).

Checks that:
  * dataframe_fingerprint() is a deterministic function of the data (same
    data -> same hash; different data, including reordering -> different
    hash).
  * environment_fingerprint() reports the Python version and package
    versions actually installed, without crashing if an optional package
    (torch/transformers) is missing.
  * reproducibility_snapshot() bundles both, plus the seed and any extra
    metadata, into one dict.
  * ResearchReportGenerator.add_reproducibility_section() actually
    attaches that information to a report, in both Markdown and JSON
    export.
"""

import json

import numpy as np
import pandas as pd
import pytest

from psyinsight.reporting.report_builder import ResearchReportGenerator
from psyinsight.utils import (
    dataframe_fingerprint,
    environment_fingerprint,
    reproducibility_snapshot,
    set_seed,
)


class TestDataframeFingerprint:
    def test_identical_data_same_hash(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert dataframe_fingerprint(df1) == dataframe_fingerprint(df2)

    def test_copy_has_same_hash(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert dataframe_fingerprint(df) == dataframe_fingerprint(df.copy())

    def test_different_values_different_hash(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})
        assert dataframe_fingerprint(df1) != dataframe_fingerprint(df2)

    def test_reordered_rows_different_hash(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        reordered = df.iloc[::-1].reset_index(drop=True)
        assert dataframe_fingerprint(df) != dataframe_fingerprint(reordered)

    def test_different_column_names_different_hash(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [1, 2, 3]})
        assert dataframe_fingerprint(df1) != dataframe_fingerprint(df2)

    def test_different_dtype_different_hash(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]}).astype({"a": "int64"})
        df2 = pd.DataFrame({"a": [1, 2, 3]}).astype({"a": "float64"})
        assert dataframe_fingerprint(df1) != dataframe_fingerprint(df2)

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            dataframe_fingerprint([1, 2, 3])

    def test_handles_nan_without_crashing(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        # Should not raise, and should be stable across calls.
        h1 = dataframe_fingerprint(df)
        h2 = dataframe_fingerprint(df)
        assert h1 == h2

    def test_empty_dataframe_does_not_crash(self):
        h = dataframe_fingerprint(pd.DataFrame())
        assert isinstance(h, str) and len(h) == 64  # sha256 hex digest length


class TestEnvironmentFingerprint:
    def test_reports_python_version(self):
        env = environment_fingerprint()
        assert "python_version" in env
        assert env["python_version"][0].isdigit()

    def test_reports_installed_core_dependencies(self):
        env = environment_fingerprint()
        versions = env["package_versions"]
        # numpy/pandas/scipy are hard dependencies of PsyInsight; if they
        # are importable at all (which every other test relies on), their
        # version metadata must be discoverable too.
        for pkg in ("numpy", "pandas", "scipy"):
            assert pkg in versions, f"expected '{pkg}' in captured package versions"

    def test_does_not_crash_on_uninstalled_optional_packages(self):
        # torch/transformers may or may not be installed in this
        # environment -- either way this must not raise.
        env = environment_fingerprint()
        assert isinstance(env["package_versions"], dict)

    def test_includes_a_timestamp(self):
        env = environment_fingerprint()
        assert "captured_at_utc" in env
        # Should be ISO-8601 parseable.
        from datetime import datetime

        datetime.fromisoformat(env["captured_at_utc"])


class TestReproducibilitySnapshot:
    def test_bundles_dataset_hash_seed_and_environment(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        seed = set_seed(42)
        snapshot = reproducibility_snapshot(df, seed=seed, extra={"model": "random_forest"})

        assert snapshot["random_seed"] == 42
        assert snapshot["dataset_hash"] == dataframe_fingerprint(df)
        assert snapshot["dataset_shape"] == {"rows": 3, "columns": 1}
        assert snapshot["extra"] == {"model": "random_forest"}
        assert "environment" in snapshot

    def test_works_without_a_dataframe(self):
        snapshot = reproducibility_snapshot(seed=7)
        assert "dataset_hash" not in snapshot
        assert snapshot["random_seed"] == 7

    def test_is_json_serializable(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        snapshot = reproducibility_snapshot(df, seed=1)
        # default=str as a safety net for anything unexpected, matching
        # how ResearchReportGenerator.to_json() serializes report content.
        json.dumps(snapshot, default=str)


class TestReportReproducibilitySection:
    def test_add_reproducibility_section_appears_in_markdown(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = ResearchReportGenerator(title="Study")
        report.add_reproducibility_section(df, seed=42, extra={"model": "logistic_regression"})

        md = report.to_markdown()
        assert "Reproducibility" in md
        assert dataframe_fingerprint(df) in md

    def test_add_reproducibility_section_appears_in_json(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = ResearchReportGenerator(title="Study")
        report.add_reproducibility_section(df, seed=42)

        payload = json.loads(report.to_json())
        section = next(s for s in payload["sections"] if s["heading"] == "Reproducibility")
        assert section["content"]["random_seed"] == 42

    def test_add_reproducibility_section_without_dataframe(self):
        report = ResearchReportGenerator(title="Study")
        report.add_reproducibility_section(seed=1)  # no dataframe supplied
        md = report.to_markdown()
        assert "Reproducibility" in md  # should not raise
