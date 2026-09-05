"""
PsyInsight AI — Response Pattern Analyzer
==========================================

Applies the automata built in :mod:`psyinsight.automata.dfa` /
:mod:`psyinsight.automata.nfa` to whole survey DataFrames, turning formal
automata theory into a practical psychology-data-quality tool:

    * Validate every response against a Likert-scale DFA.
    * Detect careless "straight-lining" response patterns.
    * Validate participant ID formatting.
    * Validate free-text fields against a lightweight pattern.

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd

from psyinsight.utils import ensure_dataframe, get_logger

from .dfa import SurveyAutomataLibrary
from .nfa import PatternMatcher

__all__ = ["ResponsePatternAnalyzer"]

_logger = get_logger(__name__)


class ResponsePatternAnalyzer:
    """Runs automata-based quality checks over a survey/response DataFrame."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = ensure_dataframe(dataframe)

    # ------------------------------------------------------------------
    # Likert-scale validity
    # ------------------------------------------------------------------

    def validate_likert_column(self, column: str, min_value: int = 1, max_value: int = 5) -> Dict[str, Any]:
        """Check every value in ``column`` against a Likert-scale DFA that
        accepts a single digit in ``[min_value, max_value]``."""
        dfa = SurveyAutomataLibrary.likert_scale_dfa(min_value, max_value)
        series = self.df[column].dropna().astype(str)
        valid_mask = series.apply(lambda v: dfa.accepts(v.strip()))
        invalid_values = series[~valid_mask]

        return {
            "column": column,
            "n_checked": int(len(series)),
            "n_valid": int(valid_mask.sum()),
            "n_invalid": int((~valid_mask).sum()),
            "invalid_rows": invalid_values.index.tolist(),
            "invalid_values": invalid_values.unique().tolist(),
        }

    def validate_likert_columns(self, columns: Iterable[str], min_value: int = 1, max_value: int = 5) -> pd.DataFrame:
        rows = [self.validate_likert_column(c, min_value, max_value) for c in columns]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Straight-lining / careless responding
    # ------------------------------------------------------------------

    def detect_straight_lining(
        self, columns: List[str], run_limit: int = 5
    ) -> pd.DataFrame:
        """Flag respondents (rows) whose answers across ``columns`` contain
        a run of more than ``run_limit`` identical consecutive responses —
        a classic red flag for careless/low-effort survey completion.
        """
        subset = self.df[columns].astype(str)
        symbols = set(pd.unique(subset.values.ravel())) if not subset.empty else set()
        dfa = SurveyAutomataLibrary.no_straight_lining_dfa(symbols, run_limit=run_limit)

        flags = []
        for idx, row in subset.iterrows():
            sequence = row.tolist()
            accepted, path = dfa.run(sequence)
            flags.append(
                {
                    "row": idx,
                    "flagged_straight_lining": not accepted,
                    "longest_run_state": path[-1] if path else None,
                }
            )

        if not flags:
            # No rows (or no columns) to check -- return an empty, but
            # correctly-shaped, result instead of crashing on set_index().
            return pd.DataFrame(columns=["flagged_straight_lining", "longest_run_state"]).rename_axis(
                "row"
            )

        return pd.DataFrame(flags).set_index("row")

    # ------------------------------------------------------------------
    # Participant ID validation
    # ------------------------------------------------------------------

    def validate_participant_ids(self, column: str, prefix: str = "P") -> Dict[str, Any]:
        dfa = SurveyAutomataLibrary.participant_id_dfa(prefix=prefix)
        series = self.df[column].dropna().astype(str)
        valid_mask = series.apply(lambda v: dfa.accepts(list(v.strip())))
        invalid_values = series[~valid_mask]

        return {
            "column": column,
            "n_checked": int(len(series)),
            "n_valid": int(valid_mask.sum()),
            "n_invalid": int((~valid_mask).sum()),
            "invalid_values": invalid_values.unique().tolist()[:50],
        }

    # ------------------------------------------------------------------
    # Free-text pattern validation
    # ------------------------------------------------------------------

    def validate_pattern(self, column: str, pattern: str) -> Dict[str, Any]:
        """Validate a text column against a :class:`PatternMatcher` pattern."""
        matcher = PatternMatcher(pattern)
        series = self.df[column].dropna().astype(str)
        valid_mask = series.apply(matcher.match)

        return {
            "column": column,
            "pattern": pattern,
            "n_checked": int(len(series)),
            "n_valid": int(valid_mask.sum()),
            "n_invalid": int((~valid_mask).sum()),
            "invalid_values": series[~valid_mask].unique().tolist()[:50],
        }

    # ------------------------------------------------------------------
    # Convenience: full data-quality automaton report
    # ------------------------------------------------------------------

    def full_quality_report(
        self, likert_columns: List[str], min_value: int = 1, max_value: int = 5, run_limit: int = 5
    ) -> Dict[str, Any]:
        """Run the standard battery of automata checks and summarize results."""
        likert_report = self.validate_likert_columns(likert_columns, min_value, max_value)
        straight_lining = (
            self.detect_straight_lining(likert_columns, run_limit=run_limit)
            if len(likert_columns) >= run_limit
            else pd.DataFrame()
        )
        n_flagged = int(straight_lining["flagged_straight_lining"].sum()) if len(straight_lining) else 0

        return {
            "likert_validity": likert_report,
            "n_respondents_flagged_straight_lining": n_flagged,
            "straight_lining_detail": straight_lining,
            "total_invalid_likert_values": int(likert_report["n_invalid"].sum()) if len(likert_report) else 0,
        }
