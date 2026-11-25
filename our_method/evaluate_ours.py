"""Evaluate memory-based rejection sampling on official Boolean dataset JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_utils import (
    append_result_csv,
    load_completed_task_ids,
    load_tasks,
    score_hypotheses,
    select_tasks,
    task_target_count,
    to_observations,
)
from boolean_genetics import sample_hypotheses
from llm_caller import AINativeError, call_llm
from mock_llm import MockLLM


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
    parser.add_argument("--max-attempts", type=int, default=50)
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
        "results/ours_mock.csv" if args.mock else "results/ours.csv"
    )
    method = "ours_mock" if args.mock else "ours"
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
        completion = MockLLM() if args.mock else call_llm
        sampled = sample_hypotheses(
            to_observations(task),
            completion,
            target_count,
            max_attempts=args.max_attempts,
            allowed_operators=metadata["operators"],
            max_depth=metadata["max_depth"],
            allow_constants=False,
        )
        result = score_hypotheses(task, sampled.hypotheses, metadata)
        result["method"] = method
        result["attempts"] = sampled.attempts
        result["rejection_counts"] = dict(sampled.rejection_counts)
        append_result_csv(output, result)
        print(
            f"[{completed_selected_count + index}/{len(selected_tasks)}] "
            f"{task['observation_set_id']}: "
            f"structural recovery={result['recovery_rate']:.1%}, "
            f"semantic recovery={result['semantic_recovery_rate']:.1%}, "
            f"attempts={sampled.attempts}, "
            f"rejections={sampled.rejection_counts}"
        )

    print(f"Results appended to {output}")


if __name__ == "__main__":
    try:
        main()
    except AINativeError as error:
        raise SystemExit(f"LLM request failed: {error}") from error
