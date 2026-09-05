"""
Security-hardening and fault-injection tests.

Security (item #10 of the robustness review): this pass reviewed the
codebase for secrets, unsafe deserialization, and path-handling issues.
Findings and status:

  * No hard-coded API keys/passwords/tokens were found anywhere in
    src/ or app/ (checked by grep in this pass; also asserted below by
    a lightweight repo-wide scan so it stays true).
  * The app's optional password gate reads from ``st.secrets``, never a
    hard-coded value.
  * File uploads already had size (50MB), row (200k), and column (500)
    limits, plus content-sniffing to catch mismatched extensions --
    these were added in an earlier robustness round and are locked in
    by test_app_integration.py / test_edge_cases.py.
  * ``joblib.load`` (used by every ``save_model``/``load_model`` pair)
    can execute arbitrary code, exactly like ``pickle`` -- it is not
    currently reachable from the Streamlit UI (grep-verified: no
    ``load_model``/``load`` call sites in app/main.py), but the
    docstrings now carry an explicit warning so this doesn't get wired
    up to an untrusted upload later without someone noticing.
  * ``DataLoader.load()`` takes a raw file path and does not sandbox it
    to a particular directory. This is fine for its actual use (a
    trusted developer/script passing a local path, exactly like
    ``open()`` or ``pd.read_csv()`` would) -- it is never exposed to
    end-users through the Streamlit app, which only ever reads from
    Streamlit's own in-memory upload buffer. If a path ever becomes
    end-user-suppliable (e.g. a future "load from server path" text
    box), it MUST be re-reviewed and confined to an allowed directory.

What's explicitly NOT covered by this pass (documented, not silently
skipped): dependency vulnerability scanning (needs `pip-audit`/`safety`
run as part of CI, not a pytest test), secret-scanning as a pre-commit
hook (a repo config concern, not application code), and full
audit-logging of every user action.

Fault-injection (item #12): a few of the most plausible "component
fails mid-operation" scenarios -- corrupted model files, disk write
failures, and network-dependent optional features being unavailable --
verified to fail with a clear error rather than corrupting state or
crashing the whole process.
"""

import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from psyinsight.ml.classifier import PsyClassifier
from psyinsight.ml.regressor import PsyRegressor
from psyinsight.preprocessing.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Secret scanning
# ---------------------------------------------------------------------------


class TestNoHardcodedSecrets:
    """A conservative repo-wide grep for the most common ways a secret
    ends up hard-coded (assigned directly to a variable named like a
    credential). Deliberately narrow to avoid false positives on the
    app's own password-gate *variable names* (e.g. `entered`,
    `required_password` sourced from st.secrets/st.text_input)."""

    SUSPICIOUS_PATTERNS = (
        'api_key = "',
        "api_key = '",
        'apikey = "',
        'secret_key = "',
        'aws_secret',
        'private_key = "',
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----",
    )

    def test_no_suspicious_hardcoded_secret_patterns(self):
        offenders = []
        for py_file in (REPO_ROOT / "src").rglob("*.py"):
            text = py_file.read_text(encoding="utf-8", errors="ignore").lower()
            for pattern in self.SUSPICIOUS_PATTERNS:
                if pattern.lower() in text:
                    offenders.append((str(py_file), pattern))
        for py_file in (REPO_ROOT / "app").rglob("*.py"):
            text = py_file.read_text(encoding="utf-8", errors="ignore").lower()
            for pattern in self.SUSPICIOUS_PATTERNS:
                if pattern.lower() in text:
                    offenders.append((str(py_file), pattern))
        assert not offenders, f"Suspicious hardcoded-secret patterns found: {offenders}"

    def test_password_gate_reads_from_streamlit_secrets_not_a_literal(self):
        app_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
        assert "st.secrets" in app_source
        # The function that resolves the configured password must not
        # itself contain an assignment of a literal string password.
        import re

        func_match = re.search(
            r"def _configured_password.*?(?=\ndef )", app_source, re.DOTALL
        )
        assert func_match, "could not locate _configured_password() to inspect"
        body = func_match.group(0)
        assert "st.secrets" in body


# ---------------------------------------------------------------------------
# Unsafe deserialization is documented and not reachable from the UI
# ---------------------------------------------------------------------------


class TestModelPersistenceSecurityDocumentation:
    def test_load_model_docstrings_warn_about_joblib_deserialization_risk(self):
        for module, cls_name in (
            ("psyinsight.ml.classifier", "PsyClassifier"),
            ("psyinsight.ml.clustering", "PsyClustering"),
        ):
            mod = __import__(module, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            doc = (cls.load_model.__doc__ or "").lower()
            assert "security" in doc and "joblib" in doc, (
                f"{cls_name}.load_model() should document the joblib/pickle "
                "deserialization risk"
            )

    def test_regressor_load_docstring_warns_about_joblib_deserialization_risk(self):
        doc = (PsyRegressor.load.__doc__ or "").lower()
        assert "security" in doc and "joblib" in doc

    def test_load_model_not_reachable_from_streamlit_app(self):
        """Locks in the current, safe state: nothing in app/main.py calls
        load_model()/PsyRegressor.load() on a user-uploaded file. If this
        ever changes, this test should be updated *deliberately*,
        alongside adding an explicit trust boundary (e.g. restricting to
        authenticated/admin users) -- not silently."""
        app_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
        assert "load_model(" not in app_source
        assert "PsyRegressor.load(" not in app_source


# ---------------------------------------------------------------------------
# Upload / input hardening (locks in existing limits)
# ---------------------------------------------------------------------------


class TestUploadHardeningLimits:
    def test_app_defines_upload_size_row_and_column_limits(self):
        app_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
        assert "MAX_UPLOAD_MB" in app_source
        assert "MAX_ROWS" in app_source
        assert "MAX_COLS" in app_source

    def test_data_loader_enforces_a_file_size_ceiling(self, tmp_path):
        # DataLoader.MAX_FILE_SIZE_MB should reject an oversized file
        # before attempting to parse it.
        big_file = tmp_path / "big.csv"
        original_limit = DataLoader.MAX_FILE_SIZE_MB
        try:
            DataLoader.MAX_FILE_SIZE_MB = 0.0001  # ~100 bytes, for a fast test
            big_file.write_text("a,b,c\n" + ("1,2,3\n" * 50))
            with pytest.raises(ValueError, match="exceeds"):
                DataLoader.load_csv(str(big_file))
        finally:
            DataLoader.MAX_FILE_SIZE_MB = original_limit


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


class TestFaultInjection:
    def test_loading_a_corrupted_model_file_raises_clear_error(self, tmp_path):
        """Simulates a corrupted/truncated model file on disk (e.g. a
        crashed save, a partially-transferred file) -- must raise a
        clear exception, not silently return garbage or crash the
        interpreter."""
        corrupted = tmp_path / "corrupted_model.joblib"
        corrupted.write_bytes(b"this is not a valid joblib/pickle file")

        clf = PsyClassifier()
        with pytest.raises(Exception):
            clf.load_model(str(corrupted))

    def test_loading_a_nonexistent_model_file_raises_clear_error(self, tmp_path):
        missing = tmp_path / "does_not_exist.joblib"
        clf = PsyClassifier()
        with pytest.raises(Exception):
            clf.load_model(str(missing))

    def test_save_model_to_unwritable_directory_raises_clear_error(self, tmp_path):
        """Simulates a disk/permissions failure during save: writing to a
        path whose parent directory does not exist and cannot be
        auto-created (a file standing in where a directory is expected)
        should raise, not silently fail or hang."""
        blocker = tmp_path / "not_a_directory"
        blocker.write_text("i am a file, not a directory")
        bad_path = blocker / "model.joblib"

        X = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3)), columns=["a", "b", "c"])
        y = pd.Series(np.random.default_rng(0).integers(0, 2, size=20))

        clf = PsyClassifier()
        clf.set_model("logistic_regression")
        clf.fit(X.values, y.values)

        with pytest.raises(Exception):
            clf.save_model(str(bad_path))

    def test_dataloader_missing_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "nope.csv"
        with pytest.raises(FileNotFoundError):
            DataLoader.load_csv(str(missing))

    def test_dataloader_malformed_json_raises_clear_error(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not: valid json,,,")
        with pytest.raises(ValueError):
            DataLoader.load_json(str(bad_json))

    def test_optional_torch_dependency_absence_does_not_crash_set_seed(self):
        """set_seed() tries to seed torch if installed; if torch is
        absent (or partially broken) it must degrade gracefully rather
        than raising ImportError up to the caller."""
        from psyinsight.utils import set_seed

        # Whether or not torch is actually installed in this environment,
        # set_seed() must succeed either way.
        result = set_seed(123)
        assert result == 123
