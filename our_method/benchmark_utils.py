"""Shared dataset loading and recovery metrics for Boolean benchmark runs."""

from __future__ import annotations

import ast
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from boolean_genetics import (
    Observation,
    extract_candidate,
    expression_mechanistic_key,
    expression_semantic_key,
    validate_expression,
)


def load_tasks(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as file:
        dataset = json.load(file)
    metadata = dataset["metadata"]
    if "datasets_by_n_observations" in dataset:
        tasks = [
            task
            for group in dataset["datasets_by_n_observations"].values()
            for task in group
        ]
    else:
        tasks = dataset["datasets"]
    if not tasks:
        raise ValueError(f"No observation sets found in {path}")
    return metadata, tasks


def select_tasks(
    tasks: Sequence[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("sample count must be at least one")
    if count > len(tasks):
        raise ValueError(f"Requested {count} tasks, but the dataset contains {len(tasks)}")
    return random.Random(seed).sample(list(tasks), count)


def load_completed_task_ids(path: Path, method: str) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as file:
        return {
            row["observation_set_id"]
            for row in csv.DictReader(file)
            if row.get("method") == method and row.get("observation_set_id")
        }


def to_observations(task: Mapping[str, Any]) -> tuple[Observation, ...]:
    observations = []
    for row in task["observations"]:
        inputs = {}
        for name, value in row["inputs"].items():
            if type(value) not in (int, bool) or value not in (0, 1):
                raise ValueError(f"Non-Boolean input value in {task['observation_set_id']}")
            inputs[name] = bool(value)
        output = row["output"]
        if type(output) not in (int, bool) or output not in (0, 1):
            raise ValueError(f"Non-Boolean output value in {task['observation_set_id']}")
        observations.append(Observation(inputs, bool(output)))
    return tuple(observations)


def task_target_count(task: Mapping[str, Any], variables: Sequence[str]) -> int:
    semantic_truths = {
        expression_semantic_key(item["formula"], variables)
        for item in task["ground_truth_expressions"]
    }
    return len(semantic_truths)


def parse_baseline_candidates(response: str) -> list[str]:
    candidates = []
    for line in response.splitlines():
        line = line.strip()
        line = re.sub(r"^(?:[-*]\s*|\d+[.)]\s*)", "", line)
        if not line:
            continue
        try:
            candidates.append(extract_candidate(line))
        except ValueError:
            continue
    return candidates


def score_hypotheses(
    task: Mapping[str, Any],
    candidates: Iterable[str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = list(candidates)
    observations = to_observations(task)
    variables = metadata["variables"]
    operators = metadata["operators"]
    max_depth = metadata["max_depth"]
    mech_options = metadata.get("mechanistic_opts", {})

    ground_truth_mechanistic = {
        ast.literal_eval(item["mechanistic_key"])
        if isinstance(item["mechanistic_key"], str)
        else tuple(item["mechanistic_key"])
        for item in task["ground_truth_expressions"]
    }
    ground_truth_semantic = {
        expression_semantic_key(item["formula"], variables)
        for item in task["ground_truth_expressions"]
    }
    recovered_mechanistic = set()
    recovered_semantic = set()
    valid_candidates = []
    for candidate in candidates:
        try:
            is_valid, _reason = validate_expression(
                candidate,
                observations,
                allowed_operators=operators,
                max_depth=max_depth,
                allow_constants=False,
            )
            if not is_valid:
                continue
            semantic_key = expression_semantic_key(candidate, variables)
            mechanistic_key = expression_mechanistic_key(
                candidate,
                apply_commutativity=mech_options.get("apply_commutativity", True),
                apply_idempotence_and_or=mech_options.get(
                    "apply_idempotence_and_or", True
                ),
                flatten_associativity=mech_options.get(
                    "flatten_associativity", True
                ),
            )
        except (SyntaxError, ValueError, TypeError):
            continue
        valid_candidates.append(candidate)
        if mechanistic_key in ground_truth_mechanistic:
            recovered_mechanistic.add(mechanistic_key)
        if semantic_key in ground_truth_semantic:
            recovered_semantic.add(semantic_key)

    return {
        "observation_set_id": task["observation_set_id"],
        "n_observations": task["n_observations"],
        "ground_truth_size": len(ground_truth_mechanistic),
        "semantic_ground_truth_size": len(ground_truth_semantic),
        "n_candidates": len(candidates),
        "n_valid": len(valid_candidates),
        "n_recovered_structural": len(recovered_mechanistic),
        "recovery_rate": (
            len(recovered_mechanistic) / len(ground_truth_mechanistic)
            if ground_truth_mechanistic
            else 0.0
        ),
        "n_recovered_semantic": len(recovered_semantic),
        "semantic_recovery_rate": (
            len(recovered_semantic) / len(ground_truth_semantic)
            if ground_truth_semantic
            else 0.0
        ),
        "valid_hypotheses": valid_candidates,
    }


def write_results_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("No benchmark results to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _result_csv_columns()
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_result_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _result_csv_columns()
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


def _result_csv_columns() -> list[str]:
    return [
        "method",
        "observation_set_id",
        "n_observations",
        "attempts",
        "ground_truth_size",
        "semantic_ground_truth_size",
        "n_candidates",
        "n_valid",
        "rejection_counts",
        "n_recovered_structural",
        "recovery_rate",
        "n_recovered_semantic",
        "semantic_recovery_rate",
    ]
