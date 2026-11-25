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


def _mean(rows: dict[str, dict[str, str]], column: str) -> float | None:
    if not rows:
        raise ValueError("Cannot summarize an empty result file")
    values = []
    for row in rows.values():
        value = row.get(column)
        if value is not None and value.strip():
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _format_rate(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def _plot_results(
    path: Path,
    summary: dict[str, dict[str, float | None]],
    is_mock: bool,
) -> None:
    labels = ("Baseline", "Ours")
    positions = np.arange(len(labels))
    width = 0.24
    figure, (metrics_axis, strata_axis, attempts_axis) = plt.subplots(
        1, 3, figsize=(15, 4.8), layout="constrained"
    )
    metrics = (
        ("validity_rate", "VR", "#4472C4"),
        ("novelty_rate", "NR", "#ED7D31"),
        ("recovery_rate", "RR", "#70AD47"),
    )
    for offset, (metric, label, color) in enumerate(metrics):
        values = [
            summary[method][metric] or 0.0 for method in ("baseline", "ours")
        ]
        bars = metrics_axis.bar(
            positions + (offset - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
        bar_labels = [
            "n/a" if summary[method][metric] is None
            else f"{summary[method][metric]:.1%}"
            for method in ("baseline", "ours")
        ]
        metrics_axis.bar_label(bars, padding=3, labels=bar_labels)
    metrics_axis.set_title("Hypothesis metrics")
    metrics_axis.set_ylabel("Rate")
    metrics_axis.set_xticks(positions, labels)
    metrics_axis.set_ylim(0, 1.15)
    metrics_axis.legend()
    metrics_axis.grid(axis="y", alpha=0.25)

    for offset, (metric, label, color) in enumerate(
        (
            ("simple_recovery_rate", "Simple RR", "#5B9BD5"),
            ("complex_recovery_rate", "Complex RR", "#ED7D31"),
        )
    ):
        values = [
            summary[method][metric] or 0.0 for method in ("baseline", "ours")
        ]
        bars = strata_axis.bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            label=label,
            color=color,
        )
        bar_labels = [
            "n/a" if summary[method][metric] is None
            else f"{summary[method][metric]:.1%}"
            for method in ("baseline", "ours")
        ]
        strata_axis.bar_label(bars, padding=3, labels=bar_labels)
    strata_axis.set_title("Recovery by complexity")
    strata_axis.set_xticks(positions, labels)
    strata_axis.set_ylim(0, 1.15)
    strata_axis.legend()
    strata_axis.grid(axis="y", alpha=0.25)

    attempt_values = [
        summary[method]["mean_attempts"] or 0.0
        for method in ("baseline", "ours")
    ]
    attempt_bars = attempts_axis.bar(
        positions,
        attempt_values,
        width=0.55,
        color=("#7895B2", "#2463A6"),
    )
    attempts_axis.bar_label(attempt_bars, fmt="%.1f", padding=3)
    attempts_axis.set_title("Mean proposals per task")
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
    parser.add_argument(
        "--ours", type=Path, default=Path("results/ours_mechanistic.csv")
    )
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
    required_columns = {
        "n_proposals",
        "validity_rate",
        "novelty_rate",
        "recovery_rate",
        "simple_recovery_rate",
        "complex_recovery_rate",
        "attempts",
    }
    for label, rows in (("baseline", baseline), ("ours", ours)):
        if rows:
            missing = required_columns.difference(next(iter(rows.values())))
            if missing:
                raise ValueError(
                    f"{label} results lack required metrics: {', '.join(sorted(missing))}"
                )
    budget_mismatches = [
        task_id
        for task_id in baseline
        if baseline[task_id]["n_proposals"] != ours[task_id]["n_proposals"]
    ]
    if budget_mismatches:
        raise ValueError(
            "Baseline and ours must use the same proposal count for every task"
        )

    summary = {}
    for method, rows in (("baseline", baseline), ("ours", ours)):
        summary[method] = {
            "validity_rate": _mean(rows, "validity_rate"),
            "novelty_rate": _mean(rows, "novelty_rate"),
            "recovery_rate": _mean(rows, "recovery_rate"),
            "simple_recovery_rate": _mean(rows, "simple_recovery_rate"),
            "complex_recovery_rate": _mean(rows, "complex_recovery_rate"),
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
            (
                "method",
                "mean_VR",
                "mean_NR",
                "mean_RR",
                "mean_simple_RR",
                "mean_complex_RR",
                "mean_proposals",
            )
        )
        for method, metrics in summary.items():
            writer.writerow(
                (
                    method,
                    metrics["validity_rate"],
                    metrics["novelty_rate"],
                    metrics["recovery_rate"],
                    metrics["simple_recovery_rate"],
                    metrics["complex_recovery_rate"],
                    metrics["mean_attempts"],
                )
            )
    _plot_results(args.chart, summary, is_mock)
    for method, metrics in summary.items():
        print(
            f"{method}: VR={_format_rate(metrics['validity_rate'])}, "
            f"NR={_format_rate(metrics['novelty_rate'])}, "
            f"RR={_format_rate(metrics['recovery_rate'])}, "
            f"Simple RR={_format_rate(metrics['simple_recovery_rate'])}, "
            f"Complex RR={_format_rate(metrics['complex_recovery_rate'])}"
        )
    print(f"Summary: {args.summary}\nChart: {args.chart}")


if __name__ == "__main__":
    main()
