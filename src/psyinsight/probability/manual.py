"""
PsyInsight AI
Manual Probability Module

Author: Subhranshu Ranjan Sahoo

This module implements probability concepts from scratch.
"""

import math


class ManualProbability:
    """
    Manual implementations of probability distributions.
    """

    @staticmethod
    def normal_pdf(x: float, mean: float, std: float) -> float:
        """
        Compute the Probability Density Function (PDF)
        of the Normal Distribution manually.

        Parameters:
            x (float): Value at which PDF is evaluated.
            mean (float): Mean of the distribution.
            std (float): Standard deviation.

        Returns:
            float: Probability density.
        """

        if std <= 0:
            raise ValueError("Standard deviation must be greater than zero.")

        # Normalization term
        coefficient = 1 / (std * math.sqrt(2 * math.pi))

        # Exponential term
        exponent = -0.5 * ((x - mean) / std) ** 2

        # Final PDF
        pdf = coefficient * math.exp(exponent)

        return pdf
