"""
PsyInsight AI — Automata Theory module
========================================

Applies classical automata theory (DFAs, NFAs, pattern matching) to a very
practical problem in psychological research: validating survey / response
data quality — malformed Likert responses, careless "straight-lining",
badly-formatted participant IDs, and free-text field formats.

    from psyinsight.automata import DFA, NFA, PatternMatcher
    from psyinsight.automata import SurveyAutomataLibrary, ResponsePatternAnalyzer
"""

from .dfa import DFA, SurveyAutomataLibrary
from .nfa import NFA, PatternMatcher
from .response_analyzer import ResponsePatternAnalyzer

__all__ = [
    "DFA",
    "NFA",
    "PatternMatcher",
    "SurveyAutomataLibrary",
    "ResponsePatternAnalyzer",
]
