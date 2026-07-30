"""
PsyInsight AI
Probability Analyzer

Author: Subhranshu Ranjan Sahoo

This module interprets probability values.
"""


class ProbabilityAnalyzer:
    """
    Interpret probability values in simple language.
    """

    @staticmethod
    def interpret(probability: float) -> str:
        """
        Interpret a probability value.

        Parameters
        ----------
        probability : float

        Returns
        -------
        str
        """

        if probability < 0 or probability > 1:
            raise ValueError(
                "Probability must be between 0 and 1."
            )

        if probability < 0.05:
            return "Very Rare Event"

        elif probability < 0.25:
            return "Rare Event"

        elif probability < 0.50:
            return "Moderately Likely"

        elif probability < 0.75:
            return "Likely"

        else:
            return "Highly Likely"
