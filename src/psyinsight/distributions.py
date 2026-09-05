"""
Deprecated compatibility shim.

The probability engine now lives at ``psyinsight.probability.distributions``.
This module re-exports it so any old ``from psyinsight.distributions import
ProbabilityEngine`` style imports keep working.
"""

from .probability.distributions import ProbabilityEngine  # noqa: F401

__all__ = ["ProbabilityEngine"]
