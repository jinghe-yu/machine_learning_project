"""Compare the two fixed-budget methods on a pilot or the full easy dataset."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import sys
from pathlib import Path
from statistics import mean

from benchmark_utils import load_tasks, score_hypotheses, task_target_count, to_observations
from boolean_genetics import sample_hypotheses
from mock_llm import MockLLM


ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek-flash"
MAX_CALLS = 42
FULL_MAX_CALLS = 180
MAX_OUTPUT_TOKENS = 64


def read_api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    env_file = ROOT / ".env"
    if not key and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "DEEPSEEK_API_KEY":
                key = value.strip().strip("\"'")
                break
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY in your environment or in our_method/.env.")
    if not key.isascii() or any(not 33 <= ord(character) <= 126 for character in key):
        raise SystemExit("DEEPSEEK_API_KEY must be a single printable ASCII value.")
    return key


class DeepSeekPilot:
    def __init__(self, api_key: str, request_limit: int, *, timeout: int = 30):
        self.api_key = api_key
        self.request_limit = request_limit
        self.timeout = timeout
        self.calls = 0
        self.missed_input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0

    def complete(self, prompt: str, temperature: float) -> str:
        if self.calls >= self.request_limit:
            raise RuntimeError("Request budget exhausted.")
        self.calls += 1
        body = json.dumps(
            {
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You solve Boolean-genetics hypothesis discovery tasks.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "thinking": {"type": "disabled"},
                "max_tokens": MAX_OUTPUT_TOKENS,
            }
        )
        connection = http.client.HTTPSConnection("api.deepseek.com", timeout=self.timeout)
        try:
            connection.request(
                "POST",
                "/chat/completions",
                body=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"DeepSeek returned HTTP {response.status}.")
            result = json.loads(response.read())
        finally:
            connection.close()

        usage = result["usage"]
        cached = usage.get("prompt_cache_hit_tokens", 0)
        self.cached_input_tokens += cached
        self.missed_input_tokens += usage["prompt_tokens"] - cached
        self.output_tokens += usage["completion_tokens"]
        return result["choices"][0]["message"]["content"] or ""

    def usage(self) -> dict:
        peak_cost = (
            2 * self.missed_input_tokens
            + 0.04 * self.cached_input_tokens
            + 8 * self.output_tokens
        ) / 1_000_000
        return {
            "requests": self.calls,
            "missed_input_tokens": self.missed_input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "off_peak_rmb": peak_cost / 2,
            "peak_rmb": peak_cost,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use the paid API instead of offline mock.")
    parser.add_argument("--all-tasks", action="store_true", help="Use all 56 verified easy tasks.")
    parser.add_argument("--resume-from", type=Path, help="Reuse completed rows from a prior report.")
    args = parser.parse_args()

    metadata, tasks = load_tasks(ROOT / "data/boolean_dataset.json")
    weighted_tasks = [
        (task_target_count(task, metadata["variables"], metadata["mechanistic_opts"]), task)
        for task in tasks
    ]
    if args.all_tasks:
        selected = weighted_tasks
    else:
        rng = random.Random(42)
        selected = [
            pair
            for count in (3, 4)
            for pair in rng.sample([pair for pair in weighted_tasks if pair[0] == count], 3)
        ]
    mode = "live" if args.live else "offline_mock"
    model = MODEL if args.live else None
    methods = {"baseline": ("none", False), "ours": ("mechanistic", True)}
    budgets = {}
    for budget, task in selected:
        answers = [item["formula"] for item in task["ground_truth_expressions"]]
        if score_hypotheses(task, answers, metadata)["recovery_rate"] != 1.0:
            raise SystemExit(f"Standard answers failed scoring for {task['observation_set_id']}.")
        for method in methods:
            budgets[task["observation_set_id"], method] = budget
    total_budget = sum(budgets.values())
    if total_budget > (FULL_MAX_CALLS if args.all_tasks else MAX_CALLS):
        raise SystemExit("Selected tasks exceed the request limit.")

    rows = []
    completed = set()
    if args.resume_from:
        prior = json.loads(args.resume_from.read_text(encoding="utf-8"))
        if (prior["mode"], prior["model"], prior["max_output_tokens"]) != (
            mode, model, MAX_OUTPUT_TOKENS,
        ):
            raise SystemExit("Prior report does not match this run's model or output limit.")
        for row in prior["rows"]:
            key = (row["task"], row["method"])
            if key in completed or budgets.get(key) != row["attempts"]:
                raise SystemExit(f"Prior report has a duplicate or mismatched row: {key}")
            rows.append(row)
            completed.add(key)
    reused = sum(budgets[key] for key in completed)
    remaining = total_budget - reused
    client = DeepSeekPilot(read_api_key(), remaining) if args.live and remaining else None
    print(f"{mode}: {len(selected)} tasks, {reused} reused proposals, {remaining} new.", file=sys.stderr)
    failure = None
    try:
        for budget, task in selected:
            for method, (deduplication, include_history) in methods.items():
                if (task["observation_set_id"], method) in completed:
                    continue
                sampled = sample_hypotheses(
                    to_observations(task),
                    client.complete if client else MockLLM(),
                    budget,
                    max_attempts=budget,
                    start_temperature=0.7,
                    end_temperature=0.7,
                    allowed_operators=metadata["operators"],
                    max_depth=metadata["max_depth"],
                    allow_constants=False,
                    deduplication=deduplication,
                    include_history=include_history,
                    stop_on_target=False,
                    mechanistic_options=metadata["mechanistic_opts"],
                )
                result = score_hypotheses(
                    task, sampled.proposals, metadata, proposal_count=sampled.attempts
                )
                rows.append(
                    {
                        "task": task["observation_set_id"],
                        "method": method,
                        "attempts": sampled.attempts,
                        "raw_responses": sampled.raw_responses,
                        "VR": result["validity_rate"],
                        "NR": result["novelty_rate"],
                        "RR": result["recovery_rate"],
                    }
                )
    except (OSError, http.client.HTTPException, RuntimeError, ValueError, KeyError) as error:
        failure = str(error)

    report = {
        "mode": mode,
        "model": model,
        "request_budget": total_budget,
        "reused_proposals": reused,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "rows": rows,
    }
    if client:
        report["usage"] = client.usage()
    if failure:
        report["error"] = failure
    else:
        report["means"] = {
            method: {
                metric: mean(row[metric] for row in rows if row["method"] == method)
                for metric in ("VR", "NR", "RR")
            }
            for method in methods
        }
    print(json.dumps(report, indent=2))
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
