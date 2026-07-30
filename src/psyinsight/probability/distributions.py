"""
PsyInsight AI
Probability Distributions Module

Author: Subhranshu Ranjan Sahoo

This module provides probability distribution utilities
using SciPy.
"""

from scipy.stats import norm, binom, poisson


class ProbabilityEngine:
    """
    Provides probability distribution functions
    using SciPy.
    """

    @staticmethod
    def normal_probability(
        x: float,
        mean: float,
        std: float
    ) -> float:
        """
        Compute the Probability Density Function (PDF)
        of a Normal Distribution.

        Parameters
        ----------
        x : float
            Value at which the PDF is evaluated.
        mean : float
            Mean of the distribution.
        std : float
            Standard deviation.

        Returns
        -------
        float
            Probability density.
        """

        if std <= 0:
            raise ValueError(
                "Standard deviation must be greater than zero."
            )

        return norm.pdf(x, loc=mean, scale=std)

    @staticmethod
    def binomial_probability(
        k: int,
        n: int,
        p: float
    ) -> float:
        """
        Compute the Probability Mass Function (PMF)
        of a Binomial Distribution.

        Parameters
        ----------
        k : int
            Number of successful trials.
        n : int
            Total number of trials.
        p : float
            Probability of success.

        Returns
        -------
        float
            Binomial probability.
        """

        if not (0 <= p <= 1):
            raise ValueError(
                "Probability must be between 0 and 1."
            )

        return binom.pmf(k, n, p)

    @staticmethod
    def poisson_probability(
        k: int,
        lam: float
    ) -> float:
        """
        Compute the Probability Mass Function (PMF)
        of a Poisson Distribution.

        Parameters
        ----------
        k : int
            Number of observed events.
        lam : float
            Average rate of occurrence.

        Returns
        -------
        float
            Poisson probability.
        """

        if lam <= 0:
            raise ValueError(
                "Lambda must be greater than zero."
            )

        return poisson.pmf(k, lam)
