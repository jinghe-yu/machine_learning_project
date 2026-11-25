"""Summarize benchmark CSV files and draw a Matplotlib comparison chart."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return {row["observation_set_id"]: row for row in csv.DictReader(file)}


def _mean(rows: dict[str, dict[str, str]], column: str) -> float:
    if not rows:
        raise ValueError("Cannot summarize an empty result file")
    return sum(float(row[column]) for row in rows.values()) / len(rows)


def _plot_results(path: Path, summary: dict[str, dict[str, float]], is_mock: bool) -> None:
    labels = ("Baseline", "Ours")
    positions = np.arange(len(labels))
    width = 0.34
    figure, (recovery_axis, attempts_axis) = plt.subplots(
        1, 2, figsize=(11, 4.8), layout="constrained"
    )
    recovery_metrics = (
        ("recovery_rate", "Structural", "#4472C4"),
        ("semantic_recovery_rate", "Semantic", "#ED7D31"),
    )
    for offset, (metric, label, color) in enumerate(recovery_metrics):
        values = [summary[method][metric] for method in ("baseline", "ours")]
        bars = recovery_axis.bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            label=label,
            color=color,
        )
        recovery_axis.bar_label(bars, fmt="%.1f%%", padding=3, labels=[f"{v:.1%}" for v in values])
    recovery_axis.set_title("Recovery rate")
    recovery_axis.set_ylabel("Recovery rate")
    recovery_axis.set_xticks(positions, labels)
    recovery_axis.set_ylim(0, 1.15)
    recovery_axis.legend()
    recovery_axis.grid(axis="y", alpha=0.25)

    attempt_values = [summary[method]["mean_attempts"] for method in ("baseline", "ours")]
    attempt_bars = attempts_axis.bar(
        positions,
        attempt_values,
        width=0.55,
        color=("#7895B2", "#2463A6"),
    )
    attempts_axis.bar_label(attempt_bars, fmt="%.1f", padding=3)
    attempts_axis.set_title("Mean model calls per task")
    attempts_axis.set_xticks(positions, labels)
    attempts_axis.set_ylim(0, max(1.0, max(attempt_values) * 1.2))
    attempts_axis.grid(axis="y", alpha=0.25)

    title = "Boolean hypothesis recovery"
    if is_mock:
        title += " — MOCK DEMO, NOT EXPERIMENTAL RESULTS"
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/baseline.csv"))
    parser.add_argument("--ours", type=Path, default=Path("results/ours.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--chart", type=Path, default=Path("results/recovery.png"))
    args = parser.parse_args()

    missing = [path for path in (args.baseline, args.ours) if not path.is_file()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        parser.error(
            f"Missing results: {missing_paths}. Run baseline.py and evaluate_ours.py "
            "successfully before comparing."
        )

    baseline = _read_rows(args.baseline)
    ours = _read_rows(args.ours)
    if baseline.keys() != ours.keys():
        raise ValueError("Baseline and ours results must cover the same observation sets")

    summary = {}
    for method, rows in (("baseline", baseline), ("ours", ours)):
        summary[method] = {
            "recovery_rate": _mean(rows, "recovery_rate"),
            "semantic_recovery_rate": _mean(rows, "semantic_recovery_rate"),
            "mean_attempts": _mean(rows, "attempts"),
        }
    is_mock = any(
        "_mock" in row.get("method", "")
        for rows in (baseline, ours)
        for row in rows.values()
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ("method", "mean_structural_recovery", "mean_semantic_recovery", "mean_attempts")
        )
        for method, metrics in summary.items():
            writer.writerow(
                (
                    method,
                    metrics["recovery_rate"],
                    metrics["semantic_recovery_rate"],
                    metrics["mean_attempts"],
                )
            )
    _plot_results(args.chart, summary, is_mock)
    for method, metrics in summary.items():
        print(
            f"{method}: structural={metrics['recovery_rate']:.1%}, "
            f"semantic={metrics['semantic_recovery_rate']:.1%}"
        )
    print(f"Summary: {args.summary}\nChart: {args.chart}")


if __name__ == "__main__":
    main()
