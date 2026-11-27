"""Memory-based rejection sampling for Boolean-genetics hypotheses."""

from __future__ import annotations

import ast
import itertools
import json
import re
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence


CompletionFn = Callable[[str, float], str]
DeduplicationMode = Literal["none", "semantic", "mechanistic"]
_MAX_TRUTH_TABLE_VARIABLES = 16


@dataclass(frozen=True)
class Observation:
    inputs: Mapping[str, bool]
    output: bool


@dataclass(frozen=True)
class SamplingResult:
    hypotheses: tuple[str, ...]
    proposals: tuple[str | None, ...]
    raw_responses: tuple[str, ...]
    attempts: int
    rejection_counts: Mapping[str, int]
    reached_target: bool


def _normalize_operators(expression: str) -> str:
    return re.sub(
        r"\b(AND|OR|NOT|XOR)\b",
        lambda match: {"AND": "and", "OR": "or", "NOT": "not", "XOR": "^"}[
            match.group(1).upper()
        ],
        expression,
        flags=re.IGNORECASE,
    )


def _parse_expression(expression: str) -> ast.Expression:
    tree = ast.parse(_normalize_operators(expression), mode="eval")
    allowed = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.BinOp,
        ast.BitXor,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Call,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
            raise ValueError("only True and False constants are allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id.upper() != "NOR":
                raise ValueError("only NOR(...) function calls are allowed")
            if node.keywords or len(node.args) < 2:
                raise ValueError("NOR requires at least two positional arguments")
    return tree


def _evaluate(node: ast.AST, values: Mapping[str, bool]) -> bool:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, values)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ValueError(f"unknown variable: {node.id}")
        return values[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _evaluate(node.operand, values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
        return _evaluate(node.left, values) != _evaluate(node.right, values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return not any(_evaluate(argument, values) for argument in node.args)
    if isinstance(node, ast.BoolOp):
        parts = (_evaluate(value, values) for value in node.values)
        if isinstance(node.op, ast.And):
            return all(parts)
        if isinstance(node.op, ast.Or):
            return any(parts)
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def _extract_candidate(response: str) -> str:
    if not isinstance(response, str):
        raise ValueError("model response must be text")

    fenced = re.search(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)```", response, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        labeled = re.search(
            r"(?im)^\s*(?:hypothesis|formula|answer|expression)\s*[:=]\s*(.+?)\s*$",
            response,
        )
        candidate = labeled.group(1).strip() if labeled else response.strip()

    if not candidate:
        raise ValueError("response did not contain a Boolean expression")
    return candidate


def extract_candidate(response: str) -> str:
    """Extract one Boolean expression from a model completion."""
    return _extract_candidate(response)


def _validate_observations(
    observations: Sequence[Observation],
) -> tuple[tuple[str, ...], tuple[Observation, ...]]:
    if not observations:
        raise ValueError("at least one observation is required")

    normalized = tuple(observations)
    variables = tuple(sorted(normalized[0].inputs))
    if not variables:
        raise ValueError("observations must include at least one Boolean variable")
    if len(variables) > _MAX_TRUTH_TABLE_VARIABLES:
        raise ValueError(
            f"semantic deduplication supports at most {_MAX_TRUTH_TABLE_VARIABLES} variables"
        )

    seen_rows: dict[tuple[bool, ...], bool] = {}
    for index, observation in enumerate(normalized):
        if tuple(sorted(observation.inputs)) != variables:
            raise ValueError(f"observation {index} has a different variable set")
        if any(type(value) is not bool for value in observation.inputs.values()):
            raise ValueError(f"observation {index} inputs must be booleans")
        if type(observation.output) is not bool:
            raise ValueError(f"observation {index} output must be a boolean")
        row = tuple(observation.inputs[name] for name in variables)
        if row in seen_rows and seen_rows[row] != observation.output:
            raise ValueError(f"conflicting outputs for observation {row}")
        seen_rows[row] = observation.output
    return variables, normalized


def _semantic_key(expression: ast.Expression, variables: tuple[str, ...]) -> tuple[bool, ...]:
    return tuple(
        _evaluate(expression, dict(zip(variables, assignment)))
        for assignment in itertools.product((False, True), repeat=len(variables))
    )


def _build_prompt(
    variables: tuple[str, ...],
    observations: tuple[Observation, ...],
    accepted: Sequence[str],
    last_feedback: str | None,
    allowed_operators: frozenset[str],
    max_depth: int | None,
    allow_constants: bool,
    deduplication: DeduplicationMode,
    include_history: bool,
    mechanistic_options: Mapping[str, bool],
    proposals: Sequence[str | None] | None = None,
) -> str:
    data = [
        {"inputs": dict(observation.inputs), "output": observation.output}
        for observation in observations
    ]
    if deduplication == "semantic":
        uniqueness = (
            "Do not repeat an accepted hypothesis or a logically equivalent form."
        )
    elif deduplication == "mechanistic":
        uniqueness = (
            "Do not repeat an accepted hypothesis under the structural "
            "equivalence rules. "
            + "; ".join(
                [
                    rule
                    for enabled, rule in (
                        (
                            mechanistic_options["apply_commutativity"],
                            "operand order is ignored",
                        ),
                        (
                            mechanistic_options["apply_idempotence_and_or"],
                            "repeated AND/OR operands collapse",
                        ),
                        (
                            mechanistic_options["flatten_associativity"],
                            "nested AND/OR/XOR operations are flattened",
                        ),
                    )
                    if enabled
                ]
                + [
                    "De Morgan, distribution, and absorption are not applied."
                ]
            )
            + "."
        )
    else:
        uniqueness = "Repeating a previous hypothesis is allowed."
    history = ""
    if include_history:
        if proposals is None:
            label, entries = "Previously accepted hypotheses", accepted
        else:
            label, entries = "Previous proposals", proposals
        memory = "\n".join(f"- {entry or '(unparseable)'}" for entry in entries) or "(none)"
        history = f"{label}:\n{memory}\n"
    feedback = (
        f"\nPrevious rejection: {last_feedback}"
        if include_history and last_feedback
        else ""
    )
    depth_constraint = (
        f"Maximum expression depth: {max_depth}.\n" if max_depth is not None else ""
    )
    constant_constraint = (
        "Boolean constants are allowed."
        if allow_constants
        else "Do not use Boolean constants."
    )
    operator_syntax = {
        "AND": "`and`",
        "OR": "`or`",
        "NOT": "`not`",
        "XOR": "`^`",
        "NOR": "`NOR(a, b)`",
    }
    syntax = ", ".join(
        f"{operator} as {operator_syntax[operator]}"
        for operator in sorted(allowed_operators)
    )
    return (
        "Propose exactly one Boolean-genetics hypothesis consistent with every observation.\n"
        f"Use only these operators: {', '.join(sorted(allowed_operators))}. "
        "\n"
        f"Use these operator forms: {syntax}; use parentheses. "
        f"{constant_constraint} Return only the expression.\n"
        f"{uniqueness}\n"
        f"{depth_constraint}"
        f"Variables: {', '.join(variables)}\n"
        f"Observations: {json.dumps(data, sort_keys=True)}\n"
        f"{history}"
        f"{feedback}"
    )


def sample_hypotheses(
    observations: Sequence[Observation],
    complete: CompletionFn,
    target_count: int,
    *,
    max_attempts: int = 50,
    start_temperature: float = 1.0,
    end_temperature: float = 0.2,
    allowed_operators: Sequence[str] = ("AND", "OR", "NOT"),
    max_depth: int | None = None,
    allow_constants: bool = True,
    deduplication: DeduplicationMode = "semantic",
    include_history: bool = True,
    show_all_proposals: bool = False,
    feedback_mode: Literal["generic", "none", "counterexample"] = "generic",
    stop_on_target: bool = True,
    mechanistic_options: Mapping[str, bool] | None = None,
) -> SamplingResult:
    """Generate hypotheses until the target is met or the proposal budget runs out.

    `complete(prompt, temperature)` is the only model-specific integration point.
    Proposal history retains one entry per model call; failed extraction is `None`.
    """
    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    if max_attempts < 0:
        raise ValueError("max_attempts must be non-negative")
    if not (0.0 <= end_temperature <= start_temperature):
        raise ValueError("temperatures must satisfy 0 <= end_temperature <= start_temperature")

    variables, validated_observations = _validate_observations(observations)
    allowed = frozenset(operator.upper() for operator in allowed_operators)
    unsupported_operators = allowed.difference({"AND", "OR", "NOT", "XOR", "NOR"})
    if unsupported_operators:
        raise ValueError(f"unsupported operators: {', '.join(sorted(unsupported_operators))}")
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if deduplication not in {"none", "semantic", "mechanistic"}:
        raise ValueError(f"unsupported deduplication mode: {deduplication}")
    if feedback_mode not in {"generic", "none", "counterexample"}:
        raise ValueError(f"unsupported feedback mode: {feedback_mode}")
    expected_options = {
        "apply_commutativity",
        "apply_idempotence_and_or",
        "flatten_associativity",
    }
    canonicalization_options = {
        "apply_commutativity": True,
        "apply_idempotence_and_or": True,
        "flatten_associativity": True,
    }
    if mechanistic_options is not None:
        unsupported_options = set(mechanistic_options).difference(expected_options)
        if unsupported_options:
            raise ValueError(
                f"unsupported mechanistic option(s): {', '.join(sorted(unsupported_options))}"
            )
        canonicalization_options.update(mechanistic_options)
    if any(type(value) is not bool for value in canonicalization_options.values()):
        raise ValueError("mechanistic_options values must be booleans")
    accepted: list[str] = []
    seen_keys: set[tuple] = set()
    proposals: list[str | None] = []
    raw_responses: list[str] = []
    rejection_counts = {"invalid": 0, "inconsistent": 0, "duplicate": 0}
    last_feedback: str | None = None
    attempts = 0

    while (
        attempts < max_attempts
        and (not stop_on_target or len(accepted) < target_count)
    ):
        progress = min(len(accepted) / target_count, 1.0) if target_count else 1.0
        temperature = start_temperature + (end_temperature - start_temperature) * progress
        prompt = _build_prompt(
            variables,
            validated_observations,
            accepted,
            last_feedback if feedback_mode != "none" else None,
            allowed,
            max_depth,
            allow_constants,
            deduplication,
            include_history,
            canonicalization_options,
            proposals if show_all_proposals else None,
        )
        attempts += 1
        candidate: str | None = None
        response = complete(prompt, temperature)
        raw_responses.append(response if isinstance(response, str) else repr(response))

        try:
            candidate = _extract_candidate(response)
            parsed = _validate_candidate_syntax(
                candidate, variables, allowed, max_depth, allow_constants
            )
            if deduplication == "semantic":
                key = _semantic_key(parsed, variables)
            elif deduplication == "mechanistic":
                key = expression_mechanistic_key(
                    candidate, **canonicalization_options
                )
            else:
                key = None
        except (SyntaxError, ValueError, TypeError) as error:
            rejection_counts["invalid"] += 1
            proposals.append(candidate)
            last_feedback = f"{candidate or 'The response'} is invalid: {error}"
            continue

        proposals.append(candidate)
        mismatch = next(
            (observation for observation in validated_observations
             if _evaluate(parsed, observation.inputs) != observation.output),
            None,
        )
        if mismatch is not None:
            rejection_counts["inconsistent"] += 1
            last_feedback = f"{candidate} does not match all observed outputs."
            if feedback_mode == "counterexample":
                inputs = json.dumps(dict(mismatch.inputs), sort_keys=True)
                last_feedback += f" For inputs {inputs}, the observed output is {json.dumps(mismatch.output)}."
            continue
        if key is not None and key in seen_keys:
            rejection_counts["duplicate"] += 1
            last_feedback = (
                f"{candidate} duplicates an accepted hypothesis "
                f"under {deduplication} deduplication."
            )
            continue

        accepted.append(candidate)
        if key is not None:
            seen_keys.add(key)
        last_feedback = None

    return SamplingResult(
        hypotheses=tuple(accepted),
        proposals=tuple(proposals),
        raw_responses=tuple(raw_responses),
        attempts=attempts,
        rejection_counts=rejection_counts,
        reached_target=len(accepted) >= target_count,
    )


def _expression_depth(node: ast.AST) -> int:
    if isinstance(node, ast.Expression):
        return _expression_depth(node.body)
    if isinstance(node, ast.BoolOp):
        return 1 + max(_expression_depth(value) for value in node.values)
    if isinstance(node, ast.UnaryOp):
        return 1 + _expression_depth(node.operand)
    if isinstance(node, ast.BinOp):
        return 1 + max(_expression_depth(node.left), _expression_depth(node.right))
    if isinstance(node, ast.Call):
        return 1 + max(_expression_depth(argument) for argument in node.args)
    return 0


def _validate_candidate_syntax(
    candidate: str,
    variables: Sequence[str],
    allowed_operators: frozenset[str],
    max_depth: int | None,
    allow_constants: bool,
) -> ast.Expression:
    parsed = _parse_expression(candidate)
    referenced = {
        node.id
        for node in ast.walk(parsed)
        if isinstance(node, ast.Name) and node.id.upper() != "NOR"
    }
    unknown = referenced.difference(variables)
    if unknown:
        raise ValueError(f"unknown variable(s): {', '.join(sorted(unknown))}")
    if not allow_constants and any(isinstance(node, ast.Constant) for node in ast.walk(parsed)):
        raise ValueError("Boolean constants are not allowed")

    used_operators = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.BoolOp):
            used_operators.add("AND" if isinstance(node.op, ast.And) else "OR")
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            used_operators.add("NOT")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
            used_operators.add("XOR")
        elif isinstance(node, ast.Call):
            used_operators.add("NOR")
    disallowed = used_operators.difference(allowed_operators)
    if disallowed:
        raise ValueError(f"operator(s) not allowed: {', '.join(sorted(disallowed))}")
    if max_depth is not None and _expression_depth(parsed) > max_depth:
        raise ValueError(f"expression depth exceeds maximum of {max_depth}")
    return parsed


def validate_expression(
    expression: str,
    observations: Sequence[Observation],
    *,
    allowed_operators: Sequence[str] = ("AND", "OR", "NOT"),
    max_depth: int | None = None,
    allow_constants: bool = False,
) -> tuple[bool, str | None]:
    """Check syntax, task constraints, and consistency with observed data."""
    variables, validated_observations = _validate_observations(observations)
    allowed = frozenset(operator.upper() for operator in allowed_operators)
    parsed = validate_expression_constraints(
        expression,
        variables,
        allowed_operators=tuple(allowed),
        max_depth=max_depth,
        allow_constants=allow_constants,
    )
    if any(
        _evaluate(parsed, observation.inputs) != observation.output
        for observation in validated_observations
    ):
        return False, "expression does not match all observed outputs"
    return True, None


def validate_expression_constraints(
    expression: str,
    variables: Sequence[str],
    *,
    allowed_operators: Sequence[str] = ("AND", "OR", "NOT"),
    max_depth: int | None = None,
    allow_constants: bool = False,
) -> ast.Expression:
    """Parse an expression and enforce its syntax and hypothesis-space constraints."""
    allowed = frozenset(operator.upper() for operator in allowed_operators)
    unsupported_operators = allowed.difference({"AND", "OR", "NOT", "XOR", "NOR"})
    if unsupported_operators:
        raise ValueError(f"unsupported operators: {', '.join(sorted(unsupported_operators))}")
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    return _validate_candidate_syntax(
        expression, variables, allowed, max_depth, allow_constants
    )


def expression_operator_count(expression: str) -> int:
    """Count operator applications, using arity minus one for AND/OR chains."""
    parsed = _parse_expression(expression)
    count = 0
    for node in ast.walk(parsed):
        if isinstance(node, ast.BoolOp):
            count += len(node.values) - 1
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            count += 1
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
            count += 1
        elif isinstance(node, ast.Call):
            count += 1
    return count


def dataset_formula_to_candidate(formula: str, *, use_nor: bool) -> str:
    """Render a stored truth formula using the candidate depth convention."""
    if not use_nor:
        return formula

    class RenderNor(ast.NodeTransformer):
        def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
            self.generic_visit(node)
            operand = node.operand
            if (
                isinstance(node.op, ast.Not)
                and isinstance(operand, ast.BoolOp)
                and isinstance(operand.op, ast.Or)
            ):
                return ast.Call(
                    func=ast.Name(id="NOR", ctx=ast.Load()), args=operand.values, keywords=[]
                )
            return node

    return ast.unparse(RenderNor().visit(_parse_expression(formula)))


def evaluate_expression(expression: str, values: Mapping[str, bool]) -> bool:
    """Evaluate an allowed Boolean expression against one variable assignment."""
    parsed = _parse_expression(expression)
    return _evaluate(parsed, values)


def expression_semantic_key(
    expression: str,
    variables: Sequence[str],
) -> tuple[bool, ...]:
    """Return a truth-table key in deterministic variable/assignment order."""
    parsed = _parse_expression(expression)
    return _semantic_key(parsed, tuple(sorted(variables)))


def expression_mechanistic_key(
    expression: str,
    *,
    apply_commutativity: bool = True,
    apply_idempotence_and_or: bool = True,
    flatten_associativity: bool = True,
) -> tuple:
    """Canonicalize an expression using the benchmark's structural rules."""
    parsed = _parse_expression(expression)

    def canonical(node: ast.AST) -> tuple:
        if isinstance(node, ast.Expression):
            return canonical(node.body)
        if isinstance(node, ast.Name):
            return ("VAR", node.id)
        if isinstance(node, ast.Constant):
            return ("CONST", node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            operand = node.operand
            if isinstance(operand, ast.BoolOp) and isinstance(operand.op, ast.Or):
                children = [canonical(value) for value in operand.values]
                if apply_commutativity:
                    children.sort()
                return ("NOR", tuple(children))
            return ("NOT", canonical(operand))
        if isinstance(node, ast.Call):
            children = [canonical(argument) for argument in node.args]
            if apply_commutativity:
                children.sort()
            return ("NOR", tuple(children))
        if isinstance(node, ast.BoolOp):
            name = "AND" if isinstance(node.op, ast.And) else "OR"
            children = [canonical(value) for value in node.values]
            if flatten_associativity:
                flattened = []
                for child in children:
                    if child[0] == name:
                        flattened.extend(child[1])
                    else:
                        flattened.append(child)
                children = flattened
            if apply_commutativity:
                children.sort()
            if apply_idempotence_and_or and name in {"AND", "OR"}:
                children = [
                    child
                    for index, child in enumerate(children)
                    if index == 0 or child != children[index - 1]
                ]
            return (name, tuple(children))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
            children = [canonical(node.left), canonical(node.right)]
            if flatten_associativity:
                flattened = []
                for child in children:
                    if child[0] == "XOR":
                        flattened.extend(child[1])
                    else:
                        flattened.append(child)
                children = flattened
            if apply_commutativity:
                children.sort()
            return ("XOR", tuple(children))
        raise ValueError(f"unsupported syntax: {type(node).__name__}")

    return canonical(parsed)
