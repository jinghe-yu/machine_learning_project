"""Run five paired DeepSeek Flash comparisons across the three Boolean datasets."""

from __future__ import annotations

import argparse
import http.client
import json
import random
import socket
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from threading import Lock
from zoneinfo import ZoneInfo

from benchmark_utils import load_tasks, score_hypotheses, task_target_count, to_observations
from boolean_genetics import dataset_formula_to_candidate, extract_candidate, sample_hypotheses
from mock_llm import MockLLM

from .pilot_deepseek import DeepSeekPilot, MAX_OUTPUT_TOKENS, MODEL, ROOT, read_api_key


DATASETS = {
    "basic": "boolean_basic_depth1.json",
    "extended": "boolean_dataset.json",
    "full": "boolean_full_depth2.json",
}
METHODS = {
    "independent": dict(deduplication="none", include_history=False),
    "history_only": dict(deduplication="none", show_all_proposals=True, feedback_mode="none"),
    "current": dict(deduplication="mechanistic"),
    "no_rejection_feedback": dict(deduplication="mechanistic", feedback_mode="none"),
    "counterexample": dict(deduplication="mechanistic", feedback_mode="counterexample"),
}
METRICS = ("validity_rate", "novelty_rate", "recovery_rate", "complex_recovery_rate")
TOKEN_FIELDS = ("missed_input_tokens", "cached_input_tokens", "output_tokens")
SHANGHAI = ZoneInfo("Asia/Shanghai")


def peak_time(now: datetime) -> bool:
    return now.weekday() < 5 and (9 <= now.hour < 12 or 14 <= now.hour < 18)


class Ledger:
    """Checkpoints completed rows and reserves peak-rate spending before each call."""

    def __init__(self, path: Path, mode: str, prior_cost: float):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = Lock()
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS config (mode TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, row TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY, cost REAL NOT NULL, usage TEXT
            );
        """)
        saved = self.db.execute("SELECT mode FROM config").fetchone()
        if saved is None:
            with self.db:
                self.db.execute("INSERT INTO config VALUES (?)", (mode,))
        elif saved[0] != mode:
            raise SystemExit("Use separate checkpoints for mock and live runs.")
        (_, outstanding, settled) = self.totals()
        self.exposure = outstanding + settled + prior_cost

    def close(self) -> None:
        self.db.close()

    def rows(self) -> dict:
        return {key: json.loads(row) for key, row in self.db.execute("SELECT id, row FROM results")}

    def save(self, key: str, row: dict) -> None:
        with self.db:
            self.db.execute("INSERT INTO results VALUES (?, ?)", (key, json.dumps(row)))

    def reserve(self, prompt: str) -> tuple[int, float]:
        # UTF-8 bytes plus an allowance for the fixed system message, roles, and framing.
        input_allowance = len(prompt.encode("utf-8")) + 256
        estimated_peak = (2 * input_allowance + 8 * MAX_OUTPUT_TOKENS) / 1_000_000
        with self.lock, self.db:
            if self.exposure + estimated_peak > 9.75:
                raise RuntimeError("Stopped before the 10 RMB project budget.")
            cursor = self.db.execute("INSERT INTO requests (cost) VALUES (?)", (estimated_peak,))
            self.exposure += estimated_peak
            return cursor.lastrowid, estimated_peak

    def settle(self, reservation: tuple[int, float], usage: dict, peak: bool) -> None:
        request_id, reserved_cost = reservation
        actual_peak = (
            2 * usage["missed_input_tokens"]
            + 0.04 * usage["cached_input_tokens"]
            + 8 * usage["output_tokens"]
        ) / 1_000_000
        cost = actual_peak if peak else actual_peak / 2
        with self.lock, self.db:
            self.db.execute("UPDATE requests SET cost = ?, usage = ? WHERE id = ?",
                            (cost, json.dumps(usage), request_id))
            self.exposure += cost - reserved_cost

    def totals(self) -> tuple[list[dict], float, float]:
        confirmed = []
        held = 0.0
        settled = 0.0
        for cost, usage in self.db.execute("SELECT cost, usage FROM requests"):
            if usage is None:
                held += cost
            else:
                settled += cost
                confirmed.append(json.loads(usage))
        return confirmed, held, settled


def score_row(round_id, dataset, method, task, metadata, proposals, raw, reused=False):
    scores = score_hypotheses(task, proposals, metadata)
    return {
        "round": round_id,
        "dataset": dataset,
        "method": method,
        "task": task["observation_set_id"],
        "attempts": len(raw),
        "reused": reused,
        "ground_truth_size": scores["ground_truth_size"],
        "n_complex_ground_truth": scores["n_complex_ground_truth"],
        "raw_responses": raw,
        **{key: scores[key] for key in METRICS},
    }


def row_key(round_id, dataset, method, task_id) -> str:
    return f"{round_id}/{dataset}/{method}/{task_id}"


def reuse_extended(tasks, metadata, mode: str) -> tuple[dict, float]:
    if mode != "live":
        return {}, 0.0
    expanded = json.loads((ROOT / "results/deepseek_flash_56.json").read_text())
    pilot = json.loads((ROOT / "results/deepseek_flash_pilot.json").read_text())
    if any((report["mode"], report["model"], report["max_output_tokens"]) !=
           (mode, MODEL, MAX_OUTPUT_TOKENS) for report in (expanded, pilot)):
        raise SystemExit("The prior 56-task experiment uses a different model or token limit.")
    by_id = {task["observation_set_id"]: (budget, task) for budget, task in tasks}
    rows = {}
    for prior in expanded["rows"]:
        method = {"baseline": "independent", "ours": "current"}[prior["method"]]
        budget, task = by_id[prior["task"]]
        raw = prior["raw_responses"]
        assert len(raw) == prior["attempts"] == budget
        proposals = [extract_candidate(response) if response.strip() else None for response in raw]
        row = score_row(1, "extended", method, task, metadata, proposals, raw, reused=True)
        assert tuple(row[k] for k in METRICS[:3]) == tuple(prior[k] for k in ("VR", "NR", "RR"))
        rows[row_key(1, "extended", method, prior["task"])] = row
    assert len(rows) == 112
    return rows, sum(report["usage"]["off_peak_rmb"] for report in (expanded, pilot))


def run_job(round_id, dataset, method, budget, task, metadata, ledger, api_key):
    if api_key:
        client = DeepSeekPilot(api_key, budget * 2, timeout=60)

        def complete(prompt, temperature):
            for attempt in range(2):
                reservation = ledger.reserve(prompt)
                before = client.usage()
                start = datetime.now(SHANGHAI)
                try:
                    response = client.complete(prompt, temperature)
                except (socket.timeout, ConnectionError, http.client.IncompleteRead):
                    if attempt:
                        raise
                    continue
                after = client.usage()
                ledger.settle(
                    reservation, {key: after[key] - before[key] for key in TOKEN_FIELDS},
                    peak_time(start) or peak_time(datetime.now(SHANGHAI)),
                )
                return response
    else:
        complete = MockLLM()
    result = sample_hypotheses(
        to_observations(task), complete, budget, max_attempts=budget,
        start_temperature=0.7, end_temperature=0.7,
        allowed_operators=metadata["operators"], max_depth=metadata["max_depth"],
        allow_constants=False, stop_on_target=False,
        mechanistic_options=metadata["mechanistic_opts"], **METHODS[method],
    )
    assert result.attempts == budget
    return score_row(round_id, dataset, method, task, metadata,
                     result.proposals, result.raw_responses)


def run_stage(pool, pending, rows, ledger, api_key, round_id, dataset, metadata):
    jobs = {
        pool.submit(run_job, round_id, dataset, method, budget, task, metadata, ledger, api_key): key
        for key, method, budget, task in pending
    }
    for index, future in enumerate(as_completed(jobs), 1):
        key = jobs[future]
        try:
            row = future.result()
        except Exception:
            for other in jobs:
                other.cancel()
            # Preserve successful in-flight jobs before propagating the failure.
            for other in as_completed(jobs):
                if not other.cancelled() and other.exception() is None and jobs[other] not in rows:
                    with ledger.lock:
                        ledger.save(jobs[other], other.result())
                    rows[jobs[other]] = other.result()
            raise
        with ledger.lock:
            ledger.save(key, row)
        rows[key] = row
        if index % 100 == 0:
            print(f"Round {round_id}, {dataset}: {index}/{len(jobs)} new rows", flush=True)


def summarize(rows, rounds):
    summary = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for method in METHODS:
            group = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            summary[dataset][method] = {}
            for name in (*METRICS, "multiple_solutions_recovery_rate"):
                eligible = [row for row in group if row["ground_truth_size"] > 1] if name.startswith("multiple") else group
                metric = "recovery_rate" if name.startswith("multiple") else name
                if not any(row[metric] is not None for row in eligible):
                    continue
                values = [
                    mean(row[metric] for row in eligible if row["round"] == round_id and row[metric] is not None)
                    for round_id in range(1, rounds + 1)
                ]
                summary[dataset][method][name] = {
                    "mean": mean(values), "stdev": stdev(values) if rounds > 1 else None,
                    "by_round": values,
                }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--rounds", type=int, choices=(1, 3), default=3)
    parser.add_argument("--workers", type=int, choices=range(1, 65), default=32)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The output already exists. Choose an unused filename.")
    mode = "live" if args.live else "mock"
    datasets = {}
    for label, filename in DATASETS.items():
        metadata, tasks = load_tasks(ROOT / "data" / filename)
        use_nor = "NOR" in metadata["operators"]
        weighted = []
        for task in tasks:
            answers = [
                dataset_formula_to_candidate(answer["formula"], use_nor=use_nor)
                for answer in task["ground_truth_expressions"]
            ]
            if score_hypotheses(task, answers, metadata)["recovery_rate"] != 1.0:
                raise SystemExit(f"Standard answers failed scoring for {label}/{task['observation_set_id']}.")
            budget = task_target_count(task, metadata["variables"], metadata["mechanistic_opts"])
            weighted.append((budget, task))
        datasets[label] = (metadata, weighted)
    api_key = read_api_key() if args.live else None
    metadata, weighted = datasets["extended"]
    reused, prior_cost = reuse_extended(weighted, metadata, mode)
    ledger = Ledger(args.checkpoint, mode, prior_cost)
    try:
        rows = ledger.rows()
        rows.update(reused)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for round_id in range(1, args.rounds + 1):
                for label, (metadata, weighted) in datasets.items():
                    jobs = []
                    for budget, task in weighted:
                        for method in METHODS:
                            key = row_key(round_id, label, method, task["observation_set_id"])
                            if key not in rows:
                                jobs.append((key, method, budget, task))
                    random.Random(f"{round_id}/{label}").shuffle(jobs)
                    run_stage(pool, jobs, rows, ledger, api_key, round_id, label, metadata)
                    paid, held, settled = ledger.totals()
                    print(f"Round {round_id}, {label} done. {len(paid)} new API responses; "
                          f"{settled:.4f} RMB so far; {held:.4f} RMB held for unknown usage.", flush=True)

        selected_rows = [row for key, row in sorted(rows.items()) if row["round"] <= args.rounds]
        paid, held, settled = ledger.totals()
        report = {
            "mode": mode,
            "model": MODEL if args.live else None,
            "temperature": 0.7,
            "thinking": "disabled",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "rounds": args.rounds,
            "new_api_responses": len(paid),
            "new_token_usage": {key: sum(call[key] for call in paid) for key in TOKEN_FIELDS},
            "prior_cost_rmb": prior_cost,
            "new_cost_rmb": settled,
            "unconfirmed_cost_reserve_rmb": held,
            "summary": summarize(selected_rows, args.rounds),
            "rows": selected_rows,
        }
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
        print(f"Saved {len(selected_rows)} rows to {args.output}", flush=True)
    finally:
        ledger.close()


if __name__ == "__main__":
    main()
