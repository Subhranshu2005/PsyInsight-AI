"""
PsyInsight AI — Deterministic Finite Automaton (DFA)
=====================================================

A general-purpose, well-tested DFA implementation, plus a small library of
DFAs that are actually useful when validating psychology survey data:
Likert-scale response strings, participant ID codes, and binary
response sequences.

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

__all__ = ["DFA", "SurveyAutomataLibrary"]


@dataclass
class DFA:
    """A textbook deterministic finite automaton.

    Parameters
    ----------
    states: set of state names.
    alphabet: set of valid input symbols.
    transitions: mapping ``(state, symbol) -> next_state``.
    start_state: the initial state.
    accept_states: set of accepting/final states.
    """

    states: Set[str]
    alphabet: Set[str]
    transitions: Dict[Tuple[str, str], str]
    start_state: str
    accept_states: Set[str]

    def __post_init__(self):
        if self.start_state not in self.states:
            raise ValueError(f"start_state '{self.start_state}' not in states")
        if not self.accept_states.issubset(self.states):
            raise ValueError("accept_states must be a subset of states")

    def step(self, state: str, symbol: str) -> Optional[str]:
        """Return the next state, or ``None`` if the transition is undefined
        (i.e. the automaton rejects immediately — no implicit dead state)."""
        return self.transitions.get((state, symbol))

    def run(self, sequence: Iterable[str]) -> Tuple[bool, List[str]]:
        """Run the DFA over ``sequence``.

        Returns ``(accepted, path)`` where ``path`` is the list of states
        visited (including the start state), useful for debugging /
        visualizing why a string was rejected.
        """
        state = self.start_state
        path = [state]
        for symbol in sequence:
            if symbol not in self.alphabet:
                return False, path
            next_state = self.step(state, symbol)
            if next_state is None:
                return False, path
            state = next_state
            path.append(state)
        return state in self.accept_states, path

    def accepts(self, sequence: Iterable[str]) -> bool:
        accepted, _ = self.run(sequence)
        return accepted

    def is_complete(self) -> bool:
        """Whether every (state, symbol) pair has a defined transition."""
        return all(
            (state, symbol) in self.transitions
            for state in self.states
            for symbol in self.alphabet
        )

    def to_dict(self) -> dict:
        return {
            "states": sorted(self.states),
            "alphabet": sorted(self.alphabet),
            "transitions": {f"{s}|{a}": t for (s, a), t in self.transitions.items()},
            "start_state": self.start_state,
            "accept_states": sorted(self.accept_states),
        }


class SurveyAutomataLibrary:
    """Factory of ready-made DFAs for common psychology-survey validation
    tasks, so users don't need to hand-build transition tables."""

    @staticmethod
    def likert_scale_dfa(min_value: int = 1, max_value: int = 5) -> DFA:
        """DFA that accepts a *single* Likert-scale digit in
        ``[min_value, max_value]`` — one symbol, one transition to ACCEPT."""
        alphabet = {str(i) for i in range(min_value, max_value + 1)}
        states = {"START", "ACCEPT", "REJECT"}
        transitions = {("START", sym): "ACCEPT" for sym in alphabet}
        return DFA(states, alphabet, transitions, "START", {"ACCEPT"})

    @staticmethod
    def binary_response_dfa() -> DFA:
        """Accepts any non-empty string over {0, 1} — i.e. a well-formed
        binary (yes/no) response sequence of arbitrary length."""
        states = {"START", "SEEN", "REJECT"}
        alphabet = {"0", "1"}
        transitions = {
            ("START", "0"): "SEEN",
            ("START", "1"): "SEEN",
            ("SEEN", "0"): "SEEN",
            ("SEEN", "1"): "SEEN",
        }
        return DFA(states, alphabet, transitions, "START", {"SEEN"})

    @staticmethod
    def participant_id_dfa(prefix: str = "P") -> DFA:
        """Accepts IDs of the shape ``<prefix><digit><digit><digit>``
        e.g. ``P001`` — a common participant coding scheme in psych studies."""
        digits = {str(i) for i in range(10)}
        alphabet = digits | {prefix}
        states = {"S0", "S1", "S2", "S3", "ACCEPT"}
        transitions = {("S0", prefix): "S1"}
        transitions.update({("S1", d): "S2" for d in digits})
        transitions.update({("S2", d): "S3" for d in digits})
        transitions.update({("S3", d): "ACCEPT" for d in digits})
        return DFA(states, alphabet, transitions, "S0", {"ACCEPT"})

    @staticmethod
    def no_straight_lining_dfa(scale_symbols: Iterable[str], run_limit: int = 5) -> DFA:
        """Accepts Likert response sequences that do **not** contain more
        than ``run_limit`` identical consecutive answers in a row — a
        classic automated check for "straight-lining" (careless/low-effort
        responding) in survey data.

        States encode "which symbol we are currently repeating" and "how
        long the current run is" (capped at ``run_limit``); reaching
        ``run_limit + 1`` repeats of the same symbol is unreachable
        (undefined transition => reject).
        """
        symbols = set(scale_symbols)
        states = {"START"}
        transitions: Dict[Tuple[str, str], str] = {}

        for sym in symbols:
            for run in range(1, run_limit + 1):
                states.add(f"{sym}:{run}")

        # From START, any symbol begins a run of length 1
        for sym in symbols:
            transitions[("START", sym)] = f"{sym}:1"

        for sym in symbols:
            for run in range(1, run_limit + 1):
                current = f"{sym}:{run}"
                # repeating the same symbol advances the run counter,
                # but only up to run_limit (beyond that => undefined => reject)
                if run < run_limit:
                    transitions[(current, sym)] = f"{sym}:{run + 1}"
                # switching to a different symbol resets the run to 1
                for other in symbols - {sym}:
                    transitions[(current, other)] = f"{other}:1"

        accept_states = set(states) - {"START"}
        return DFA(states, symbols, transitions, "START", accept_states)
