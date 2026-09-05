import pandas as pd

from psyinsight.utils import (
    PsyConfig,
    ensure_dataframe,
    infer_column_types,
    memory_usage_report,
    safe_sample,
    set_seed,
)


def test_set_seed_is_deterministic():
    import random

    set_seed(123)
    a = random.random()
    set_seed(123)
    b = random.random()
    assert a == b


def test_infer_column_types():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "group": ["a", "b", "a", "b"],
            "notes": [
                "this is a fairly long free text response about mood",
                "another quite long piece of open ended text here",
                "short",
                "yet another moderately long open text response",
            ],
        }
    )
    buckets = infer_column_types(df)
    assert "age" in buckets["numeric"]
    assert "group" in buckets["categorical"]


def test_ensure_dataframe_from_dict():
    df = ensure_dataframe({"a": [1, 2, 3]})
    assert isinstance(df, pd.DataFrame)
    assert list(df["a"]) == [1, 2, 3]


def test_safe_sample_does_not_raise_when_n_too_large():
    df = pd.DataFrame({"a": [1, 2]})
    sample = safe_sample(df, n=10)
    assert len(sample) == 2


def test_memory_usage_report_keys():
    df = pd.DataFrame({"a": [1, 2, 3]})
    report = memory_usage_report(df)
    assert "total_mb" in report and "n_rows" in report


def test_psyconfig_roundtrip(tmp_path):
    cfg = PsyConfig({"seed": 42})
    path = tmp_path / "config.json"
    cfg.to_json(str(path))
    loaded = PsyConfig.from_json(str(path))
    assert loaded.get("seed") == 42
