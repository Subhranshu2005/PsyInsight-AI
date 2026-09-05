"""
PsyInsight AI
Visualization Module

Author: Subhranshu Ranjan Sahoo

This module provides visualization utilities for datasets.

Thread-safety note
-------------------
Every method here builds its own ``matplotlib.figure.Figure`` via the
object-oriented API (``Figure()`` / ``fig.add_subplot()``) and returns it,
instead of drawing onto matplotlib's global, process-wide ``pyplot`` state
(``plt.figure()`` / ``plt.gcf()`` / ``plt.show()``). That global state is
shared across every thread in a process -- in a multi-user Streamlit
deployment, two sessions plotting at the same moment could otherwise steal
or corrupt each other's figure. Because each call here creates an
independent ``Figure`` object, concurrent calls from different sessions
cannot interfere with one another, with no external locking required.

Callers are responsible for displaying/saving the returned ``Figure``
(e.g. ``st.pyplot(fig)`` in Streamlit, or ``fig.savefig(...)``).
"""

from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


class Visualizer:
    """
    A class for creating statistical visualizations.

    Every plotting method returns a ``matplotlib.figure.Figure`` rather
    than drawing to a global, shared canvas -- see the module docstring.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the Visualizer.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataset to visualize.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Visualizer expects a pandas DataFrame, got {type(dataframe).__name__}."
            )
        self.dataframe = dataframe

    def _require_column(self, column: str) -> None:
        if column not in self.dataframe.columns:
            raise KeyError(
                f"Column '{column}' not found in the dataset. "
                f"Available columns: {list(self.dataframe.columns)}"
            )

    def histogram(self, column: str, bins: int = 10) -> Figure:
        """
        Plot a histogram.
        """
        self._require_column(column)
        data = self.dataframe[column].dropna()

        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        if data.empty:
            ax.text(0.5, 0.5, f"No non-missing values in '{column}'", ha="center", va="center")
        else:
            ax.hist(data, bins=bins)

        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.grid(True)

        return fig

    def box_plot(self, column: str) -> Figure:
        """
        Plot a box plot.
        """
        self._require_column(column)
        data = self.dataframe[column].dropna()

        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        if data.empty:
            ax.text(0.5, 0.5, f"No non-missing values in '{column}'", ha="center", va="center")
        else:
            ax.boxplot(data)

        ax.set_title(f"Box Plot of {column}")
        ax.set_ylabel(column)
        ax.grid(True)

        return fig

    def scatter_plot(self, x_column: str, y_column: str) -> Figure:
        """
        Plot a scatter plot.
        """
        self._require_column(x_column)
        self._require_column(y_column)

        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        ax.scatter(self.dataframe[x_column], self.dataframe[y_column])

        ax.set_title(f"{y_column} vs {x_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.grid(True)

        return fig

    def line_plot(self, x_column: str, y_column: str) -> Figure:
        """
        Plot a line plot.
        """
        self._require_column(x_column)
        self._require_column(y_column)

        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        ax.plot(self.dataframe[x_column], self.dataframe[y_column])

        ax.set_title(f"{y_column} vs {x_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.grid(True)

        return fig

    def bar_plot(self, x_column: str, y_column: str) -> Figure:
        """
        Plot a bar chart.
        """
        self._require_column(x_column)
        self._require_column(y_column)

        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        ax.bar(self.dataframe[x_column], self.dataframe[y_column])

        ax.set_title(f"{y_column} by {x_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.grid(True)

        return fig

    def correlation_heatmap(self) -> Figure:
        """
        Plot a correlation heatmap.
        """
        correlation = self.dataframe.corr(numeric_only=True)

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        if correlation.empty:
            ax.text(
                0.5, 0.5, "No numeric columns available for a correlation heatmap",
                ha="center", va="center",
            )
            return fig

        im = ax.imshow(correlation, interpolation="nearest", aspect="auto")
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(len(correlation.columns)))
        ax.set_xticklabels(correlation.columns, rotation=90)
        ax.set_yticks(range(len(correlation.columns)))
        ax.set_yticklabels(correlation.columns)

        ax.set_title("Correlation Heatmap")
        fig.tight_layout()

        return fig

    def distribution_curve(self, column: str, bins: int = 20) -> Figure:
        """
        Plot a distribution curve.
        """
        self._require_column(column)
        data = self.dataframe[column].dropna()

        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        if data.empty:
            ax.text(0.5, 0.5, f"No non-missing values in '{column}'", ha="center", va="center")
        else:
            ax.hist(data, bins=bins, density=True)

        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
        ax.grid(True)

        return fig

    @staticmethod
    def save_plot(fig: Figure, filename: str) -> str:
        """
        Save a Figure returned by one of this class's plotting methods.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            A figure returned by e.g. ``histogram()`` or ``scatter_plot()``.
        filename : str
            Destination path.
        """
        if not isinstance(fig, Figure):
            raise TypeError(
                "save_plot() expects a matplotlib Figure (e.g. the return value "
                "of histogram()/scatter_plot()/etc.), not "
                f"{type(fig).__name__}. "
            )
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        return filename
