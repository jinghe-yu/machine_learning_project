"""Evaluate memory-based rejection sampling on official Boolean dataset JSON."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="Cap proposals per task (default: use the mechanistic ground-truth size)",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--dedup",
        choices=("none", "semantic", "mechanistic"),
        default="mechanistic",
        help="Duplicate criterion applied during sampling",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Do not include accepted hypotheses or rejection feedback in later prompts",
    )
    parser.add_argument("--mock", action="store_true", help="Run without network access")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip selected tasks already present in the output CSV",
    )
    args = parser.parse_args()
    if args.max_attempts is not None and args.max_attempts < 1:
        parser.error("--max-attempts must be at least one")

    metadata, all_tasks = load_tasks(args.dataset)
    selected_tasks = select_tasks(all_tasks, args.samples, args.seed)
    method = f"ours_{args.dedup}"
    if args.no_feedback:
        method += "_no_feedback"
    if args.mock:
        method += "_mock"
    output = args.output or Path(
        f"results/{method}.csv"
    )
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
        target_count = task_target_count(
            task, metadata["variables"], metadata.get("mechanistic_opts")
        )
        proposal_budget = (
            target_count
            if args.max_attempts is None
            else min(target_count, args.max_attempts)
        )
        completion = MockLLM() if args.mock else call_llm
        sampled = sample_hypotheses(
            to_observations(task),
            completion,
            target_count,
            max_attempts=proposal_budget,
            start_temperature=args.temperature,
            end_temperature=args.temperature,
            allowed_operators=metadata["operators"],
            max_depth=metadata["max_depth"],
            allow_constants=False,
            deduplication=args.dedup,
            include_history=not args.no_feedback,
            stop_on_target=False,
            mechanistic_options=metadata.get("mechanistic_opts"),
        )
        result = score_hypotheses(
            task, sampled.proposals, metadata, proposal_count=sampled.attempts
        )
        result["method"] = method
        result["attempts"] = sampled.attempts
        result["rejection_counts"] = json.dumps(
            dict(sampled.rejection_counts), sort_keys=True
        )
        result["proposal_candidates_json"] = json.dumps(sampled.proposals)
        result["raw_responses_json"] = json.dumps(sampled.raw_responses)
        append_result_csv(output, result)
        print(
            f"[{completed_selected_count + index}/{len(selected_tasks)}] "
            f"{task['observation_set_id']}: "
            f"VR={result['validity_rate']:.1%}, NR={result['novelty_rate']:.1%}, "
            f"RR={result['recovery_rate']:.1%}, "
            f"attempts={sampled.attempts}, "
            f"rejections={sampled.rejection_counts}"
        )

    print(f"Results appended to {output}")


if __name__ == "__main__":
    try:
        main()
    except AINativeError as error:
        raise SystemExit(f"LLM request failed: {error}") from error
