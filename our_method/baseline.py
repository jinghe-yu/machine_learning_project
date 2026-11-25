"""Run a one-shot Boolean-expression generation baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_utils import (
    append_result_csv,
    load_completed_task_ids,
    load_tasks,
    parse_baseline_candidates,
    score_hypotheses,
    select_tasks,
    task_target_count,
    to_observations,
)
from llm_caller import AINativeError, call_llm
from mock_llm import mock_baseline_response


def create_baseline_prompt(task: dict, target_count: int, variables: list[str]) -> str:
    observations = [
        {"inputs": row["inputs"], "output": row["output"]}
        for row in task["observations"]
    ]
    return (
        "Generate the requested number of distinct Boolean expressions consistent "
        "with every observation. Use only variables "
        f"{', '.join(variables)} and operators "
        f"{', '.join(task['operators'])}. No constants. "
        f"Maximum expression depth: {task['max_depth']}. "
        "Return one expression per line, with no explanation.\n"
        f"Number requested: {target_count}\n"
        f"Observations: {json.dumps(observations, sort_keys=True)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/boolean_dataset.json"),
        help="Official Boolean dataset JSON",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--mock", action="store_true", help="Run without network access")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip selected tasks already present in the output CSV",
    )
    args = parser.parse_args()

    metadata, all_tasks = load_tasks(args.dataset)
    selected_tasks = select_tasks(all_tasks, args.samples, args.seed)
    output = args.output or Path(
        "results/baseline_mock.csv" if args.mock else "results/baseline.csv"
    )
    method = "baseline_mock" if args.mock else "baseline"
    completed_ids = (
        load_completed_task_ids(output, method) if args.resume else set()
    )
    tasks = [
        task
        for task in selected_tasks
        if task["observation_set_id"] not in completed_ids
    ]
    completed_selected_count = len(selected_tasks) - len(tasks)
    for index, task in enumerate(tasks, start=1):
        target_count = task_target_count(task, metadata["variables"])
        if args.mock:
            response = mock_baseline_response(
                to_observations(task),
                metadata["variables"],
                target_count,
                metadata["operators"],
                metadata["max_depth"],
            )
        else:
            response = call_llm(
                create_baseline_prompt(task, target_count, metadata["variables"]),
                args.temperature,
            )
        candidates = parse_baseline_candidates(response)
        result = score_hypotheses(task, candidates, metadata)
        result["method"] = method
        result["attempts"] = 1
        append_result_csv(output, result)
        print(
            f"[{completed_selected_count + index}/{len(selected_tasks)}] "
            f"{task['observation_set_id']}: "
            f"structural recovery={result['recovery_rate']:.1%}, "
            f"semantic recovery={result['semantic_recovery_rate']:.1%}"
        )

    print(f"Results appended to {output}")


if __name__ == "__main__":
    try:
        main()
    except AINativeError as error:
        raise SystemExit(f"LLM request failed: {error}") from error
