"""
PsyInsight AI
Visualization Module

Author: Subhranshu Ranjan Sahoo

This module provides visualization utilities for datasets.
"""

import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    """
    A class for creating statistical visualizations.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the Visualizer.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataset to visualize.
        """
        self.dataframe = dataframe

    def histogram(
        self,
        column: str,
        bins: int = 10,
    ):
        """
        Plot a histogram.
        """

        plt.figure(figsize=(8, 5))

        plt.hist(
            self.dataframe[column].dropna(),
            bins=bins,
        )

        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.grid(True)

        plt.show()

    def box_plot(
        self,
        column: str,
    ):
        """
        Plot a box plot.
        """

        plt.figure(figsize=(8, 5))

        plt.boxplot(
            self.dataframe[column].dropna()
        )

        plt.title(f"Box Plot of {column}")
        plt.ylabel(column)

        plt.grid(True)

        plt.show()

    def scatter_plot(
        self,
        x_column: str,
        y_column: str,
    ):
        """
        Plot a scatter plot.
        """

        plt.figure(figsize=(8, 5))

        plt.scatter(
            self.dataframe[x_column],
            self.dataframe[y_column],
        )

        plt.title(
            f"{y_column} vs {x_column}"
        )

        plt.xlabel(x_column)
        plt.ylabel(y_column)

        plt.grid(True)

        plt.show()

    def line_plot(
        self,
        x_column: str,
        y_column: str,
    ):
        """
        Plot a line plot.
        """

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.dataframe[x_column],
            self.dataframe[y_column],
        )

        plt.title(
            f"{y_column} vs {x_column}"
        )

        plt.xlabel(x_column)
        plt.ylabel(y_column)

        plt.grid(True)

        plt.show()

    def bar_plot(
        self,
        x_column: str,
        y_column: str,
    ):
        """
        Plot a bar chart.
        """

        plt.figure(figsize=(8, 5))

        plt.bar(
            self.dataframe[x_column],
            self.dataframe[y_column],
        )

        plt.title(
            f"{y_column} by {x_column}"
        )

        plt.xlabel(x_column)
        plt.ylabel(y_column)

        plt.grid(True)

        plt.show()

    def correlation_heatmap(self):
        """
        Plot a correlation heatmap.
        """

        correlation = self.dataframe.corr(
            numeric_only=True
        )

        plt.figure(figsize=(8, 6))

        plt.imshow(
            correlation,
            interpolation="nearest",
            aspect="auto",
        )

        plt.colorbar()

        plt.xticks(
            range(len(correlation.columns)),
            correlation.columns,
            rotation=90,
        )

        plt.yticks(
            range(len(correlation.columns)),
            correlation.columns,
        )

        plt.title("Correlation Heatmap")

        plt.tight_layout()

        plt.show()

    def distribution_curve(
        self,
        column: str,
        bins: int = 20,
    ):
        """
        Plot a distribution curve.
        """

        plt.figure(figsize=(8, 5))

        plt.hist(
            self.dataframe[column].dropna(),
            bins=bins,
            density=True,
        )

        plt.title(
            f"Distribution of {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Density")

        plt.grid(True)

        plt.show()

    def save_plot(
        self,
        filename: str,
    ):
        """
        Save the current plot.
        """

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
        )
