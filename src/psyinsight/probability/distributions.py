"""
PsyInsight AI
Probability Distributions Module

Author: Subhranshu Ranjan Sahoo

This module provides probability distribution utilities.
"""

import numpy as np
from scipy.stats import norm, binom, poisson


class ProbabilityEngine:
    """
    Probability distributions for data analysis.
    """

    @staticmethod
    def normal_probability(x, mean, std):
        """
        Compute the probability density of a normal distribution.
        """
        return norm.pdf(x, loc=mean, scale=std)

    @staticmethod
    def binomial_probability(k, n, p):
        """
        Compute the probability of exactly k successes.
        """
        return binom.pmf(k, n, p)

    @staticmethod
    def poisson_probability(k, lam):
        """
        Compute the probability of observing k events.
        """
        return poisson.pmf(k, lam)
