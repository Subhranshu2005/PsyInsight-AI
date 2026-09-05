import pandas as pd

from psyinsight.automata import (
    DFA,
    NFA,
    PatternMatcher,
    ResponsePatternAnalyzer,
    SurveyAutomataLibrary,
)
from psyinsight.automata.nfa import EPSILON


def test_likert_dfa_accepts_valid_and_rejects_invalid():
    dfa = SurveyAutomataLibrary.likert_scale_dfa(1, 5)
    assert dfa.accepts("3")
    assert not dfa.accepts("7")
    assert not dfa.accepts("35")  # more than one digit


def test_participant_id_dfa():
    dfa = SurveyAutomataLibrary.participant_id_dfa(prefix="P")
    assert dfa.accepts(list("P001"))
    assert not dfa.accepts(list("X001"))
    assert not dfa.accepts(list("P01"))


def test_no_straight_lining_dfa_rejects_long_runs():
    dfa = SurveyAutomataLibrary.no_straight_lining_dfa({"1", "2", "3", "4", "5"}, run_limit=3)
    assert dfa.accepts(["5", "5", "5"])
    assert not dfa.accepts(["5", "5", "5", "5"])


def test_nfa_to_dfa_equivalence():
    nfa = NFA(
        states={"q0", "q1"},
        alphabet={"a", EPSILON},
        transitions={("q0", "a"): {"q1"}},
        start_state="q0",
        accept_states={"q1"},
    )
    dfa = nfa.to_dfa()
    assert nfa.accepts("a") == dfa.accepts("a")
    assert nfa.accepts("") == dfa.accepts("")


def test_pattern_matcher_ranges_and_quantifiers():
    matcher = PatternMatcher(r"[A-Z]+\d+")
    assert matcher.match("ABC123")
    assert not matcher.match("abc123")
    assert not matcher.match("AB")


def test_response_pattern_analyzer_flags_straight_lining():
    df = pd.DataFrame(
        {
            "q1": [5, 5, 5, 5, 5, 3],
            "q2": [5, 5, 5, 5, 5, 2],
            "q3": [5, 5, 5, 5, 5, 4],
            "q4": [5, 5, 5, 5, 5, 1],
            "q5": [5, 5, 5, 5, 5, 5],
            "q6": [5, 5, 5, 5, 5, 5],
        }
    )
    analyzer = ResponsePatternAnalyzer(df)
    result = analyzer.detect_straight_lining(["q1", "q2", "q3", "q4", "q5", "q6"], run_limit=5)
    assert result.loc[0, "flagged_straight_lining"]
    assert not result.loc[5, "flagged_straight_lining"]


def test_response_pattern_analyzer_participant_ids():
    df = pd.DataFrame({"pid": ["P001", "P002", "BAD"]})
    analyzer = ResponsePatternAnalyzer(df)
    result = analyzer.validate_participant_ids("pid")
    assert result["n_invalid"] == 1
