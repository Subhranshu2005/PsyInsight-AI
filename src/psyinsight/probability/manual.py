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
        """

        if std <= 0:
            raise ValueError("Standard deviation must be greater than zero.")

        coefficient = 1 / (std * math.sqrt(2 * math.pi))
        exponent = -0.5 * ((x - mean) / std) ** 2

        return coefficient * math.exp(exponent)

    @staticmethod
    def factorial(n: int) -> int:
        """
        Compute the factorial of a non-negative integer.
        """

        if n < 0:
            raise ValueError(
                "Factorial is not defined for negative numbers."
            )

        return math.factorial(n)

    @staticmethod
    def combination(n: int, k: int) -> int:
        """
        Compute the number of combinations (n choose k).
        """

        if k < 0 or k > n:
            raise ValueError("k must satisfy 0 <= k <= n.")

        return (
            ManualProbability.factorial(n)
            // (
                ManualProbability.factorial(k)
                * ManualProbability.factorial(n - k)
            )
        )

    @staticmethod
    def binomial_pmf(
        n: int,
        k: int,
        p: float
    ) -> float:
        """
        Compute the Probability Mass Function (PMF)
        of the Binomial Distribution manually.
        """

        if not (0 <= p <= 1):
            raise ValueError(
                "Probability must be between 0 and 1."
            )

        combinations = ManualProbability.combination(n, k)

        return (
            combinations
            * (p ** k)
            * ((1 - p) ** (n - k))
        )

    @staticmethod
    def poisson_pmf(
        k: int,
        lam: float
    ) -> float:
        """
        Compute the Probability Mass Function (PMF)
        of the Poisson Distribution manually.
        """

        if k < 0:
            raise ValueError(
                "k must be non-negative."
            )

        if lam <= 0:
            raise ValueError(
                "Lambda must be greater than zero."
            )

        return (
            (lam ** k)
            * math.exp(-lam)
            / ManualProbability.factorial(k)
        )
