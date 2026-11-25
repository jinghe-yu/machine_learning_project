"""Deterministic offline LLM substitute for exercising the benchmark pipeline."""

from __future__ import annotations

import json
import re
from typing import Sequence

from boolean_genetics import (
    Observation,
    expression_semantic_key,
    validate_expression,
)


def _candidate_formulas(variables: Sequence[str]) -> tuple[str, ...]:
    if len(variables) != 2:
        raise ValueError("The Boolean benchmark mock currently expects two variables")
    left, right = variables
    return (
        left,
        right,
        f"NOT {left}",
        f"NOT {right}",
        f"{left} AND {right}",
        f"{left} OR {right}",
    )


def _prompt_context(
    prompt: str,
) -> tuple[tuple[str, ...], tuple[Observation, ...], tuple[str, ...], frozenset[str], int | None]:
    variables_line = next(
        (line for line in prompt.splitlines() if line.startswith("Variables: ")),
        None,
    )
    observations_line = next(
        (line for line in prompt.splitlines() if line.startswith("Observations: ")),
        None,
    )
    if variables_line is None or observations_line is None:
        raise ValueError("Mock LLM received a prompt without task context")

    variables = tuple(
        name.strip() for name in variables_line.removeprefix("Variables: ").split(",")
    )
    raw_observations = json.loads(observations_line.removeprefix("Observations: "))
    observations = tuple(
        Observation(
            {name: bool(value) for name, value in row["inputs"].items()},
            bool(row["output"]),
        )
        for row in raw_observations
    )

    operators_line = next(
        (line for line in prompt.splitlines() if line.startswith("Use only these operators: ")),
        None,
    )
    operators = frozenset(
        item.strip().rstrip(".")
        for item in (
            operators_line.removeprefix("Use only these operators: ").split(",")
            if operators_line
            else ("AND", "OR", "NOT")
        )
    )
    depth_match = re.search(r"(?m)^Maximum expression depth: (\d+)\.$", prompt)
    max_depth = int(depth_match.group(1)) if depth_match else None

    accepted_header = "Previously accepted hypotheses:"
    accepted: list[str] = []
    if accepted_header in prompt:
        memory = prompt.split(accepted_header, 1)[1]
        for line in memory.splitlines():
            if line.startswith("Previous rejection:"):
                break
            if line.startswith("- "):
                accepted.append(line[2:].strip())
    return variables, observations, tuple(accepted), operators, max_depth


def _valid_candidates(
    variables: Sequence[str],
    observations: Sequence[Observation],
    operators: frozenset[str],
    max_depth: int | None,
) -> list[str]:
    valid = []
    for candidate in _candidate_formulas(variables):
        try:
            is_valid, _reason = validate_expression(
                candidate,
                observations,
                allowed_operators=operators,
                max_depth=max_depth,
                allow_constants=False,
            )
        except (SyntaxError, ValueError, TypeError):
            continue
        if is_valid:
            valid.append(candidate)
    return valid


class MockLLM:
    """Return intentional failures, then valid candidates outside prompt history."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str, temperature: float) -> str:
        del temperature
        variables, observations, accepted, operators, max_depth = _prompt_context(prompt)
        valid = _valid_candidates(variables, observations, operators, max_depth)
        call_index = self.calls
        self.calls += 1

        if call_index == 0:
            return "I do not know the answer."
        if call_index == 1:
            invalid = next(
                (
                    candidate
                    for candidate in _candidate_formulas(variables)
                    if candidate not in valid
                ),
                None,
            )
            if invalid is not None:
                return f"```python\n{invalid}\n```"
        if call_index == 2 and valid:
            return f"Hypothesis: {valid[0]}"
        if call_index == 3 and accepted:
            return f"```python\n{accepted[0]}\n```"

        accepted_keys = set()
        for candidate in accepted:
            try:
                accepted_keys.add(expression_semantic_key(candidate, variables))
            except (SyntaxError, ValueError, TypeError):
                continue
        for candidate in valid:
            if expression_semantic_key(candidate, variables) not in accepted_keys:
                return f"```python\n{candidate}\n```"
        return "I do not know the answer."


def mock_call_llm(prompt: str, temperature: float) -> str:
    """Convenience callable; use a fresh MockLLM instance per benchmark task."""
    return _default_mock(prompt, temperature)


def mock_baseline_response(
    observations: Sequence[Observation],
    variables: Sequence[str],
    target_count: int,
    allowed_operators: Sequence[str],
    max_depth: int,
) -> str:
    """Return a single-shot, numbered mock response for baseline pipeline tests."""
    candidates = _valid_candidates(
        variables,
        observations,
        frozenset(operator.upper() for operator in allowed_operators),
        max_depth,
    )
    return "\n".join(
        f"{index}. Expression: {candidate}"
        for index, candidate in enumerate(candidates[:target_count], start=1)
    )


_default_mock = MockLLM()
