"""
Tests for psyinsight.visualization.Visualizer.

This module previously had no test coverage at all. Beyond basic
correctness, these tests specifically lock in the object-oriented
matplotlib refactor: every plotting method must return an independent
Figure object rather than drawing onto matplotlib's shared global state,
which is what makes concurrent use across threads (e.g. multiple
Streamlit user sessions in one process) safe.
"""

import concurrent.futures

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from psyinsight.visualization.visualizer import Visualizer


@pytest.fixture
def sample_df():
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "a": rng.randn(50),
            "b": rng.randn(50),
            "group": ["x"] * 25 + ["y"] * 25,
        }
    )


class TestVisualizerBasics:
    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            Visualizer([1, 2, 3])

    def test_histogram_returns_figure(self, sample_df):
        fig = Visualizer(sample_df).histogram("a", bins=5)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_box_plot_returns_figure(self, sample_df):
        fig = Visualizer(sample_df).box_plot("a")
        assert isinstance(fig, Figure)

    def test_scatter_plot_returns_figure(self, sample_df):
        fig = Visualizer(sample_df).scatter_plot("a", "b")
        assert isinstance(fig, Figure)

    def test_line_plot_returns_figure(self, sample_df):
        fig = Visualizer(sample_df).line_plot("a", "b")
        assert isinstance(fig, Figure)

    def test_bar_plot_returns_figure(self, sample_df):
        fig = Visualizer(sample_df).bar_plot("group", "a")
        assert isinstance(fig, Figure)

    def test_correlation_heatmap_returns_figure(self, sample_df):
        fig = Visualizer(sample_df).correlation_heatmap()
        assert isinstance(fig, Figure)

    def test_distribution_curve_returns_figure(self, sample_df):
        fig = Visualizer(sample_df).distribution_curve("a")
        assert isinstance(fig, Figure)

    def test_missing_column_raises_clear_error(self, sample_df):
        with pytest.raises(KeyError, match="not found in the dataset"):
            Visualizer(sample_df).histogram("does_not_exist")

    def test_save_plot_writes_file(self, sample_df, tmp_path):
        fig = Visualizer(sample_df).histogram("a")
        out = tmp_path / "plot.png"
        Visualizer.save_plot(fig, str(out))
        assert out.exists()

    def test_save_plot_rejects_non_figure(self):
        with pytest.raises(TypeError):
            Visualizer.save_plot("not a figure", "/tmp/whatever.png")


class TestVisualizerEdgeCases:
    def test_histogram_on_all_nan_column_does_not_crash(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        fig = Visualizer(df).histogram("a")
        assert isinstance(fig, Figure)

    def test_correlation_heatmap_with_no_numeric_columns_does_not_crash(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        fig = Visualizer(df).correlation_heatmap()
        assert isinstance(fig, Figure)

    def test_empty_dataframe_does_not_crash_on_construction(self):
        Visualizer(pd.DataFrame())  # should not raise


class TestVisualizerConcurrency:
    def test_each_call_produces_an_independent_figure(self, sample_df):
        """Two plots made back-to-back must not share the same Figure
        object -- confirms this isn't drawing onto shared global state."""
        viz = Visualizer(sample_df)
        fig1 = viz.histogram("a")
        fig2 = viz.scatter_plot("a", "b")
        assert fig1 is not fig2

    def test_concurrent_plotting_across_threads_is_safe(self):
        """Regression test for the matplotlib-global-state concurrency
        bug: hammer the Visualizer from many threads at once (simulating
        multiple Streamlit sessions in one process) and confirm every
        call gets back its own valid, uncorrupted Figure."""

        def make_plot(seed):
            rng = np.random.RandomState(seed)
            df = pd.DataFrame({"x": rng.randn(30), "y": rng.randn(30)})
            fig = Visualizer(df).scatter_plot("x", "y")
            return isinstance(fig, Figure) and len(fig.axes) == 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(make_plot, range(100)))

        assert all(results)
