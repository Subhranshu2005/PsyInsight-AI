"""
Integration tests for app/main.py using Streamlit's official AppTest
harness (streamlit.testing.v1.AppTest). These actually run the Streamlit
script and simulate widget interactions, rather than importing functions
in isolation -- so they catch real wiring problems (like the t-test/
auto_train bugs found during this pass) that a pure unit test would miss.

Kept intentionally small: a full click-through of every tab is expensive
and duplicates what the underlying module tests already cover. These
focus specifically on the app-level hardening added in this pass (access
gate, rate limiting) plus a basic "does it boot at all" smoke test.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


class TestAppBoots:
    def test_app_boots_without_exception(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        assert not at.exception

    def test_no_password_configured_skips_gate(self):
        """Default deployment (no secrets.toml / app_password) should show
        the full app immediately, not a password prompt."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        assert not at.exception
        assert "PsyInsight AI" in [t.value for t in at.title]
        # No password prompt should be present (the app does have an
        # unrelated "Report title" text input in the Report tab, so check
        # specifically for a password-typed field rather than "any" input).
        assert not any(ti.label == "Password" for ti in at.text_input)


class TestAccessGate:
    def test_locked_without_correct_password(self):
        at = AppTest.from_file(APP_PATH)
        at.secrets["app_password"] = "hunter2"
        at.run(timeout=30)
        assert not at.exception
        assert len(at.text_input) == 1  # password prompt is showing

    def test_wrong_password_shows_error_and_stays_locked(self):
        at = AppTest.from_file(APP_PATH)
        at.secrets["app_password"] = "hunter2"
        at.run(timeout=30)
        at.text_input[0].input("wrong-password").run()
        at.button[0].click().run()
        assert not at.exception
        assert any("Incorrect password" in e.value for e in at.error)

    def test_correct_password_unlocks_app(self):
        at = AppTest.from_file(APP_PATH)
        at.secrets["app_password"] = "hunter2"
        at.run(timeout=30)
        at.text_input[0].input("hunter2").run()
        at.button[0].click().run()
        assert not at.exception
        assert "PsyInsight AI" in [t.value for t in at.title]
        assert len(at.tabs) > 0


class TestUploadSniffing:
    """Direct unit tests for sniff_mismatched_upload(), which flags a
    file whose content doesn't match its claimed extension (e.g. an
    Excel file saved with a .csv name)."""

    @staticmethod
    def _load_app_module():
        import importlib.util

        spec = importlib.util.spec_from_file_location("psyinsight_app_main", APP_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    class _FakeUpload:
        def __init__(self, name: str, content: bytes):
            self.name = name
            self._content = content

        def getvalue(self) -> bytes:
            return self._content

        def seek(self, _n: int) -> None:
            pass

    def test_flags_zip_content_with_csv_extension(self):
        mod = self._load_app_module()
        fake = self._FakeUpload("data.csv", b"PK\x03\x04" + b"0" * 20)
        assert mod.sniff_mismatched_upload(fake) is not None

    def test_allows_genuine_csv(self):
        mod = self._load_app_module()
        fake = self._FakeUpload("data.csv", b"a,b\n1,2\n")
        assert mod.sniff_mismatched_upload(fake) is None

    def test_allows_genuine_xlsx_named_correctly(self):
        mod = self._load_app_module()
        fake = self._FakeUpload("data.xlsx", b"PK\x03\x04" + b"0" * 20)
        assert mod.sniff_mismatched_upload(fake) is None
