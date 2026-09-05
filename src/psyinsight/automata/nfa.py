"""
PsyInsight AI — Nondeterministic Finite Automaton (NFA) + regex-style matcher
==============================================================================

A general NFA implementation (with epsilon transitions) plus subset
construction to convert an NFA into an equivalent DFA. Also ships a tiny
regex-like `PatternMatcher` used elsewhere in PsyInsight AI to validate
free-text survey fields (e.g. email-like fields, numeric codes) without
pulling in a full regex engine dependency.

Author: Subhranshu Ranjan Sahoo
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

from .dfa import DFA

__all__ = ["NFA", "PatternMatcher"]

EPSILON = "ε"


class NFA:
    """A nondeterministic finite automaton with epsilon transitions.

    ``transitions`` maps ``(state, symbol)`` to a **set** of possible next
    states (symbol may be :data:`EPSILON`).
    """

    def __init__(
        self,
        states: Set[str],
        alphabet: Set[str],
        transitions: Dict[Tuple[str, str], Set[str]],
        start_state: str,
        accept_states: Set[str],
    ):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states

    def epsilon_closure(self, states: Iterable[str]) -> FrozenSet[str]:
        stack = list(states)
        closure = set(states)
        while stack:
            state = stack.pop()
            for nxt in self.transitions.get((state, EPSILON), set()):
                if nxt not in closure:
                    closure.add(nxt)
                    stack.append(nxt)
        return frozenset(closure)

    def move(self, states: Iterable[str], symbol: str) -> Set[str]:
        result: Set[str] = set()
        for state in states:
            result |= self.transitions.get((state, symbol), set())
        return result

    def accepts(self, sequence: Iterable[str]) -> bool:
        current = self.epsilon_closure({self.start_state})
        for symbol in sequence:
            current = self.epsilon_closure(self.move(current, symbol))
            if not current:
                return False
        return bool(current & self.accept_states)

    def to_dfa(self) -> DFA:
        """Subset construction: convert this NFA into an equivalent DFA."""
        start_closure = self.epsilon_closure({self.start_state})
        unmarked = [start_closure]
        dfa_states: Dict[FrozenSet[str], str] = {start_closure: "S0"}
        dfa_transitions: Dict[Tuple[str, str], str] = {}
        counter = 1

        while unmarked:
            current_set = unmarked.pop()
            current_name = dfa_states[current_set]
            for symbol in self.alphabet:
                if symbol == EPSILON:
                    continue
                target = self.epsilon_closure(self.move(current_set, symbol))
                if not target:
                    continue
                if target not in dfa_states:
                    dfa_states[target] = f"S{counter}"
                    counter += 1
                    unmarked.append(target)
                dfa_transitions[(current_name, symbol)] = dfa_states[target]

        accept_states = {
            name for subset, name in dfa_states.items() if subset & self.accept_states
        }

        return DFA(
            states=set(dfa_states.values()),
            alphabet=self.alphabet - {EPSILON},
            transitions=dfa_transitions,
            start_state="S0",
            accept_states=accept_states,
        )


class PatternMatcher:
    """A minimal, dependency-free matcher supporting a small, safe subset of
    regex-like syntax, purpose-built for validating survey/response fields:

        .   any single character
        *   zero or more of the previous token
        +   one or more of the previous token
        ?   zero or one of the previous token
        [abc]  character class
        \\d   digit shorthand

    This is intentionally not a full regex engine — it exists so PsyInsight
    AI's validation layer does not need to trust arbitrary regex from data
    files, while still giving researchers a familiar, restricted syntax for
    describing well-formed responses.
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self._tokens = self._tokenize(pattern)

    @staticmethod
    def _tokenize(pattern: str) -> List[str]:
        tokens: List[str] = []
        i = 0
        while i < len(pattern):
            ch = pattern[i]
            if ch == "\\" and i + 1 < len(pattern):
                tokens.append(pattern[i : i + 2])
                i += 2
            elif ch == "[":
                j = pattern.index("]", i)
                tokens.append(pattern[i : j + 1])
                i = j + 1
            else:
                tokens.append(ch)
                i += 1
        return tokens

    @staticmethod
    def _char_matches(token: str, ch: str) -> bool:
        if token == ".":
            return True
        if token == r"\d":
            return ch.isdigit()
        if token == r"\w":
            return ch.isalnum() or ch == "_"
        if token.startswith("[") and token.endswith("]"):
            body = token[1:-1]
            expanded = PatternMatcher._expand_char_class(body)
            return ch in expanded
        return token == ch

    @staticmethod
    def _expand_char_class(body: str) -> str:
        """Expand ``a-z`` style ranges inside a ``[...]`` character class into
        the full set of individual characters."""
        expanded = []
        i = 0
        while i < len(body):
            if i + 2 < len(body) and body[i + 1] == "-":
                expanded.extend(chr(c) for c in range(ord(body[i]), ord(body[i + 2]) + 1))
                i += 3
            else:
                expanded.append(body[i])
                i += 1
        return "".join(expanded)

    def match(self, text: str) -> bool:
        """Return ``True`` if ``text`` fully matches the pattern."""
        return self._match_here(self._tokens, text)

    def _match_here(self, tokens: List[str], text: str) -> bool:
        if not tokens:
            return text == ""

        token = tokens[0]
        quantifier = tokens[1] if len(tokens) > 1 and tokens[1] in "*+?" else None
        rest = tokens[2:] if quantifier else tokens[1:]

        if quantifier == "*":
            return self._match_star(token, rest, text)
        if quantifier == "+":
            return bool(text) and self._char_matches(token, text[0]) and self._match_star(
                token, rest, text[1:]
            )
        if quantifier == "?":
            if text and self._char_matches(token, text[0]) and self._match_here(rest, text[1:]):
                return True
            return self._match_here(rest, text)

        if text and self._char_matches(token, text[0]):
            return self._match_here(rest, text[1:])
        return False

    def _match_star(self, token: str, rest: List[str], text: str) -> bool:
        # Greedy match, then backtrack.
        i = 0
        while i < len(text) and self._char_matches(token, text[i]):
            i += 1
        while i >= 0:
            if self._match_here(rest, text[i:]):
                return True
            i -= 1
        return False
