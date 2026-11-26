"""Run the comparable Qwen settings on DeepSeek using the team's sampler."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from threading import Lock

from openai import APIConnectionError, APIStatusError, OpenAI

from benchmark_utils import load_tasks, score_hypotheses, select_tasks, task_target_count, to_observations
from boolean_genetics import sample_hypotheses
from depth_constraint_experiment import depth_examples, diagnosis
from mock_llm import MockLLM
from pilot_deepseek import MODEL, ROOT, read_api_key


DATASETS = {"easy": "boolean_dataset.json", "hard": "boolean_full_depth2.json"}
METHODS = {
    "baseline": ("none", False),
    "ours_mechanistic": ("mechanistic", True),
    "ours_mechanistic_no_feedback": ("mechanistic", False),
    "ours_semantic": ("semantic", True),
    "ours_semantic_no_feedback": ("semantic", False),
    "ours_none": ("none", True),
    "ours_none_no_feedback": ("none", False),
}
PRIMARY = ("baseline", "ours_mechanistic")
MAX_TOKENS = 128
MAX_NEW_RMB = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Journal:
    """Save completed tasks and conservatively reserve spending before each request."""

    def __init__(self, path: Path, manifest: dict):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = Lock()
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS config (manifest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, peak REAL NOT NULL, usage TEXT);
            CREATE TABLE IF NOT EXISTS tasks (job TEXT PRIMARY KEY, data TEXT NOT NULL);
        """)
        saved = self.db.execute("SELECT manifest FROM config").fetchone()
        serialized = json.dumps(manifest, sort_keys=True)
        if saved is None:
            with self.db:
                self.db.execute("INSERT INTO config VALUES (?)", (serialized,))
        elif saved[0] != serialized:
            raise ValueError("Checkpoint uses a different dataset or configuration.")
        self.exposure = self.db.execute("SELECT COALESCE(SUM(peak),0) FROM requests").fetchone()[0]

    def reserve(self, prompt: str) -> tuple[int, float]:
        input_allowance = len(prompt.encode("utf-8")) + 256
        estimate = (2 * input_allowance + 8 * MAX_TOKENS) / 1_000_000
        with self.lock, self.db:
            if self.exposure + estimate > MAX_NEW_RMB:
                raise RuntimeError("Stopped at the API spending guard.")
            request_id = self.db.execute("INSERT INTO requests(peak) VALUES (?)", (estimate,)).lastrowid
            self.exposure += estimate
            return request_id, estimate

    def settle(self, reservation: tuple[int, float], usage: dict) -> None:
        hit = usage["cached_input_tokens"]
        peak = (2 * (usage["prompt_tokens"] - hit) + 0.04 * hit +
                8 * usage["output_tokens"]) / 1_000_000
        request_id, estimate = reservation
        with self.lock, self.db:
            self.db.execute("UPDATE requests SET peak=?,usage=? WHERE id=?",
                            (peak, json.dumps(usage), request_id))
            self.exposure += peak - estimate

    def save(self, job: str, row: dict) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT INTO tasks VALUES (?, ?)", (job, json.dumps(row)))

    def records(self) -> dict:
        return {job: json.loads(data) for job, data in self.db.execute("SELECT job,data FROM tasks")}

    def usage(self) -> dict:
        totals = Counter()
        unknown = 0
        peak = held = 0.0
        for cost, usage in self.db.execute("SELECT peak,usage FROM requests"):
            if usage is None:
                unknown += 1
                held += cost
            else:
                totals.update(json.loads(usage))
                totals["requests"] += 1
                peak += cost
        return {**totals, "peak_rmb": peak, "off_peak_rmb": peak / 2,
                "unconfirmed_requests": unknown, "unconfirmed_peak_reserve_rmb": held}


def evaluate(client, journal, dataset, condition, method, metadata, task, budget):
    deduplication, include_history = METHODS[method]
    calls = []
    example = depth_examples(metadata["operators"], metadata["max_depth"]) if condition == "examples" else ""
    mock = MockLLM()

    def complete(prompt, temperature):
        if client is None:
            calls.append({"finish_reason": "stop"})
            return mock(prompt + example, temperature)
        prompt += example
        for attempt in range(2):
            reservation = journal.reserve(prompt)
            started = time.monotonic()
            try:
                response = client.chat.completions.create(
                    model=MODEL, temperature=temperature, max_tokens=MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": "You solve Boolean-genetics hypothesis discovery tasks."},
                        {"role": "user", "content": prompt},
                    ],
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except APIConnectionError:
                if attempt:
                    raise
                continue
            except APIStatusError as error:
                if attempt or error.status_code not in (429, 502, 503, 504):
                    raise
                time.sleep(0.5)
                continue
            if response.usage is None:
                raise RuntimeError("DeepSeek response lacks usage data.")
            tokens = response.usage
            usage = {
                "prompt_tokens": tokens.prompt_tokens,
                "cached_input_tokens": getattr(tokens, "prompt_cache_hit_tokens", 0) or 0,
                "output_tokens": tokens.completion_tokens,
            }
            journal.settle(reservation, usage)
            choice = response.choices[0]
            calls.append({**usage, "seconds": time.monotonic() - started,
                          "finish_reason": choice.finish_reason})
            return choice.message.content or ""

    result = sample_hypotheses(
        to_observations(task), complete, budget, max_attempts=budget,
        start_temperature=0.7, end_temperature=0.7,
        allowed_operators=metadata["operators"], max_depth=metadata["max_depth"],
        allow_constants=False, deduplication=deduplication, include_history=include_history,
        stop_on_target=False, mechanistic_options=metadata["mechanistic_opts"],
    )
    assert result.attempts == len(calls) == budget
    score = score_hypotheses(task, result.proposals, metadata, proposal_count=budget)
    del score["valid_hypotheses"]
    diagnostics = [
        diagnosis(candidate, metadata["variables"], metadata["operators"], metadata["max_depth"])
        for candidate in result.proposals
    ]
    return {
        "dataset": dataset, "condition": condition, "method": method, **score,
        "attempts": result.attempts, "n_over_depth": diagnostics.count("over_depth"),
        "n_parse_failure": diagnostics.count("parse_failure"),
        "n_other_constraint": diagnostics.count("other_constraint"),
        "n_truncated": sum(call["finish_reason"] == "length" for call in calls),
        "proposal_candidates_json": json.dumps(result.proposals),
        "raw_responses_json": json.dumps(result.raw_responses),
        "rejection_counts": json.dumps(result.rejection_counts),
        "diagnostics_json": json.dumps(diagnostics), "calls_json": json.dumps(calls),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--remaining-hard", action="store_true",
        help="Run the five hard configurations beyond baseline and structural feedback.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 49), default=24)
    args = parser.parse_args()
    manifest = {
        "mode": "live" if args.live else "mock",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sampler_sha256": sha256(ROOT / "boolean_genetics.py"),
        "scorer_sha256": sha256(ROOT / "benchmark_utils.py"),
        "depth_runner_sha256": sha256(ROOT / "depth_constraint_experiment.py"),
        "runner_sha256": sha256(Path(__file__)),
        "model_id": MODEL, "thinking": "disabled", "temperature": 0.7, "max_tokens": MAX_TOKENS,
        "task_seed": 42, "generation_seed": None,
        "generation_seed_note": "DeepSeek Chat Completions does not expose llama.cpp's generation seed.",
        "grammar": "Unavailable on the official DeepSeek Chat Completions API.",
        "datasets": {}, "cells": [],
    }
    jobs = {}
    for dataset, filename in DATASETS.items():
        if args.remaining_hard and dataset != "hard":
            continue
        source = ROOT / "data" / filename
        metadata, all_tasks = load_tasks(source)
        tasks = select_tasks(all_tasks, len(all_tasks), 42)
        budgets = [task_target_count(task, metadata["variables"], metadata["mechanistic_opts"]) for task in tasks]
        expected = (56, 90) if dataset == "easy" else (80, 975)
        assert (len(tasks), sum(budgets)) == expected
        manifest["datasets"][dataset] = {"file": filename, "sha256": sha256(source),
                                          "tasks": len(tasks), "proposals_per_cell": sum(budgets)}
        if args.remaining_hard:
            methods = [key for key in METHODS if key not in PRIMARY]
        else:
            methods = METHODS if dataset == "easy" else PRIMARY
        cells = [("prompt", key) for key in methods]
        if dataset == "easy":
            cells.extend(("examples", key) for key in PRIMARY)
            manifest["depth_examples"] = depth_examples(metadata["operators"], metadata["max_depth"]).strip()
        for condition, method in cells:
            manifest["cells"].append({"dataset": dataset, "condition": condition, "method": method})
            for task, budget in zip(tasks, budgets):
                job = (dataset, condition, method, task["observation_set_id"])
                jobs["/".join(job)] = (dataset, condition, method, metadata, task, budget)

    journal = Journal(args.checkpoint, manifest)
    client = OpenAI(base_url="https://api.deepseek.com", api_key=read_api_key(),
                    max_retries=0, timeout=60.0) if args.live else None
    try:
        rows = journal.records()

        def run(job):
            journal.save(job, evaluate(client, journal, *jobs[job]))

        pending = [job for job in jobs if job not in rows]
        print(f"{len(jobs)} task rows; {len(rows)} reused; {len(pending)} pending.", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, job) for job in pending]
            try:
                for index, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if index % 100 == 0 or index == len(futures):
                        print(f"{index}/{len(futures)} new rows saved.", flush=True)
            finally:
                for future in futures:
                    future.cancel()

        rows = journal.records()
        assert rows.keys() == jobs.keys()
        args.output.mkdir(parents=True, exist_ok=True)
        summary = []
        for cell in manifest["cells"]:
            dataset, condition, method = (cell[k] for k in ("dataset", "condition", "method"))
            prefix = "/".join((dataset, condition, method)) + "/"
            group = [rows[job] for job in jobs if job.startswith(prefix)]
            write_csv(args.output / dataset / "raw" / f"{method}_{condition}.csv", group)
            multi = [row for row in group if row["ground_truth_size"] > 1]
            complex_rates = [row["complex_recovery_rate"] for row in group if row["complex_recovery_rate"] is not None]
            proposals = sum(row["n_proposals"] for row in group)
            summary.append({
                **cell, "tasks": len(group), "proposals": proposals,
                "mean_VR": mean(row["validity_rate"] for row in group),
                "mean_NR": mean(row["novelty_rate"] for row in group),
                "mean_RR": mean(row["recovery_rate"] for row in group),
                "multiple_truths_mean_RR": mean(row["recovery_rate"] for row in multi),
                "mean_complex_RR": mean(complex_rates) if complex_rates else None,
                "over_depth_rate": sum(row["n_over_depth"] for row in group) / proposals,
                "n_truncated": sum(row["n_truncated"] for row in group),
            })
        write_csv(args.output / "summary.csv", summary)
        manifest["usage"] = journal.usage()
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        print(f"Saved {len(rows)} rows with usage {manifest['usage']}.", flush=True)
    finally:
        if client is not None:
            client.close()
        journal.db.close()


if __name__ == "__main__":
    main()
