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

    @staticmethod
    def combination(n: int, k: int) -> int:
        """
        Compute the number of combinations (n choose k).

        Parameters:
            n (int): Total number of items.
            k (int): Number of selected items.

        Returns:
            int: Number of possible combinations.
        """

        if k < 0 or k > n:
            raise ValueError("k must satisfy 0 <= k <= n.")

        return math.factorial(n) // (
            math.factorial(k) * math.factorial(n - k)
        )

    @staticmethod
    def binomial_pmf(n: int, k: int, p: float) -> float:
        """
        Compute the Probability Mass Function (PMF)
        of the Binomial Distribution manually.

        Parameters:
            n (int): Total number of trials.
            k (int): Number of successful trials.
            p (float): Probability of success.

        Returns:
            float: Binomial probability.
        """

        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1.")

        if k < 0 or k > n:
            raise ValueError("k must satisfy 0 <= k <= n.")

        combinations = ManualProbability.combination(n, k)

        probability = (
            combinations
            * (p ** k)
            * ((1 - p) ** (n - k))
        )

        return probability
