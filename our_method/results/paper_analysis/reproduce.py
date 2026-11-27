"""Recompute the Boolean benchmark tables and figures from saved experiment records."""

from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


OUTPUT = Path(__file__).resolve().parent
RESULTS = OUTPUT.parent
DATA = RESULTS.parent / "data"
MODELS = ("Qwen3.5-4B", "Qwen3.5-9B", "Ministral 3 8B", "Qwen3.8-27B", "DeepSeek Flash")
Q4, Q9, MINISTRAL, Q27, DEEPSEEK = MODELS
DIRS = dict(zip(MODELS, (
    "qwen35_4b_q4_k_m_20261126", "qwen35_9b_q4_k_m_20261126",
    "ministral3_8b_instruct_2512_q4_k_m_20261126", "qwen38_27b_nvfp4_20261126",
    "deepseek_flash_new_prompt_20261126",
)))
Q27_RAW = {
    "easy": (f"{DIRS[Q27]}/raw", f"{DIRS[Q27]}_repeats/run_02/raw", f"{DIRS[Q27]}_repeats/run_03/raw"),
    "hard": ("qwen38_27b_nvfp4_hard_depth2_20261126/raw",),
}
B, M, M0, S, S0, N, N0 = METHODS = (
    "baseline", "ours_mechanistic", "ours_mechanistic_no_feedback", "ours_semantic",
    "ours_semantic_no_feedback", "ours_none", "ours_none_no_feedback",
)
ABLATIONS = ((M, M0), (S, S0), (N, N0))
DATASETS = {"easy": "boolean_dataset.json", "hard": "boolean_full_depth2.json"}
RATE_FIELDS = {
    "rr": ("recovery_rate", "n_recovered_structural", "ground_truth_size"),
    "vr": ("validity_rate", "n_valid", "n_proposals"),
    "nr": ("novelty_rate", "n_unique_in_space", "n_proposals"),
    "semantic_rr": ("semantic_recovery_rate", "n_recovered_semantic", "semantic_ground_truth_size"),
    "simple_rr": ("simple_recovery_rate", "n_recovered_simple", "n_simple_ground_truth"),
    "complex_rr": ("complex_recovery_rate", "n_recovered_complex", "n_complex_ground_truth"),
}
INK, GRID, TEAL, ORANGE, GRAY = "#182839", "#d9e1e8", "#007c83", "#ae4d1d", "#53657a"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def task_index(dataset):
    data = json.loads((DATA / DATASETS[dataset]).read_text())
    return {t["observation_set_id"]: (t["n_observations"], t["n_compatible_expressions"])
            for group in data["datasets_by_n_observations"].values() for t in group}


def load_group(path, dataset):
    rows = {}
    for raw in read_csv(path):
        task = raw["observation_set_id"]
        assert task not in rows, (path, task)
        row = {k: int(raw[k]) for k in ("n_observations", "ground_truth_size", "n_proposals")}
        for name, (field, num, den) in RATE_FIELDS.items():
            hits, total = int(raw[num]), int(raw[den])
            assert 0 <= hits <= total, (path, task, field)
            rate = hits / total if total else None
            assert (not raw[field] if rate is None else np.isclose(float(raw[field]), rate,
                                                                   rtol=0, atol=1e-12))
            row[name], row[num], row[den] = rate, hits, total
        row["n_over_depth"] = int(raw["n_over_depth"]) if "n_over_depth" in raw else None
        calls = json.loads(raw["calls_json"]) if "calls_json" in raw else []
        for token_field in ("prompt_tokens", "output_tokens"):
            row[token_field] = (sum(call[token_field] for call in calls)
                                if calls and all(call.get(token_field) is not None for call in calls) else None)
        assert row["n_proposals"] == row["ground_truth_size"], (path, task)
        assert row["n_simple_ground_truth"] + row["n_complex_ground_truth"] == row["ground_truth_size"]
        rows[task] = row
    assert {key: (r["n_observations"], r["ground_truth_size"]) for key, r in rows.items()} == task_index(dataset), path
    assert sum(r["n_complex_ground_truth"] for r in rows.values()) == (300 if dataset == "hard" else 0)
    return rows


def subset(rows, stratum):
    if stratum == "all":
        return rows
    if stratum == "multiple_truths":
        return {k: r for k, r in rows.items() if r["ground_truth_size"] > 1}
    if stratum == "single_truth":
        return {k: r for k, r in rows.items() if r["ground_truth_size"] == 1}
    return {k: r for k, r in rows.items() if r["n_observations"] == int(stratum[-1])}


def metrics(rows):
    values = list(rows.values())
    out = {"tasks": len(rows), "proposals": sum(r["n_proposals"] for r in values)}
    for metric, (_, hits, _) in RATE_FIELDS.items():
        rates = [r[metric] for r in values if r[metric] is not None]
        out[metric] = np.mean(rates) if rates else None
        out[hits] = sum(r[hits] for r in values)
    out["complex_tasks"] = sum(r["complex_rr"] is not None for r in values)
    out["complex_truths"] = sum(r["n_complex_ground_truth"] for r in values)
    has_depth = all(r["n_over_depth"] is not None for r in values)
    out["over_depth_rate"] = sum(r["n_over_depth"] for r in values) / out["proposals"] if has_depth else None
    for token_field in ("prompt_tokens", "output_tokens"):
        tokens = [r[token_field] for r in values]
        out[token_field] = sum(tokens) if None not in tokens else None
    return out


def paired(treatment, control):
    assert treatment.keys() == control.keys() and treatment
    keys = sorted(treatment)
    delta = np.array([100 * (treatment[k]["rr"] - control[k]["rr"]) for k in keys])
    observations = np.array([control[k]["n_observations"] for k in keys])
    rng = np.random.default_rng(20261126)
    boot = np.zeros(20_000)
    for count in np.unique(observations):
        stratum = delta[observations == count]
        indices = rng.integers(len(stratum), size=(len(boot), len(stratum)))
        boot += stratum[indices].sum(axis=1) / len(delta)
    low, high = np.quantile(boot, [0.025, 0.975])
    return dict(tasks=len(keys), delta_rr_pp=delta.mean(), lower_95_pp=low, upper_95_pp=high,
                wins=int((delta > 0).sum()), ties=int((delta == 0).sum()), losses=int((delta < 0).sum()))


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def load_runs():
    runs = {}
    for model in MODELS:
        easy = DIRS[model] + ("/easy/raw" if model == DEEPSEEK else "/raw")
        hard = {
            DEEPSEEK: "deepseek_flash_hard_seven_20261126/raw",
            Q27: Q27_RAW["hard"][0],
        }.get(model, f"{DIRS[model]}/hard/raw")
        for dataset, folder in (("easy", easy), ("hard", hard)):
            for method in METHODS:
                suffix = "_prompt" if model == DEEPSEEK else ""
                path = RESULTS / folder / f"{method}{suffix}.csv"
                runs["main", dataset, model, "1", method, "prompt"] = load_group(path, dataset)
    for method in METHODS:
        for run, folder in enumerate(Q27_RAW["easy"][1:], start=2):
            path = RESULTS / folder / f"{method}.csv"
            runs["main", "easy", Q27, str(run), method, "prompt"] = load_group(path, "easy")
        path = RESULTS / f"deepseek_flash_20261126/raw/{method}.csv"
        runs["earlier_prompt", "easy", DEEPSEEK, "1", method, "prompt"] = load_group(path, "easy")
    for method in (B, M):
        for model in (Q4, Q9, DEEPSEEK):
            for condition in ("prompt", "examples", "grammar"):
                if model == DEEPSEEK and condition == "grammar":
                    continue
                if model == DEEPSEEK and condition == "prompt":
                    continue  # The main easy table already contains these two cells.
                tag = f"qwen3.5-{model[-2:].lower()}-q4_k_m"
                path = (RESULTS / DIRS[model] / "easy/raw" / f"{method}_{condition}.csv" if model == DEEPSEEK
                        else RESULTS / "depth_constraints_20261126/raw" / f"{tag}_{method}_{condition}.csv")
                runs["depth_control", "easy", model, "1", method, condition] = load_group(path, "easy")
    return runs


def archived_deepseek():
    export = []
    for suffix in ("pilot", "56", "boolean_full"):
        source = RESULTS / f"deepseek_flash_{suffix}.json"
        data = json.loads(source.read_text())
        buckets = defaultdict(list)
        for row in data["rows"]:
            buckets[row.get("dataset", f"initial_{suffix}"), row["method"], row.get("round", 1)].append(row)
        for (dataset, method, run), rows in sorted(buckets.items()):
            assert len({r["task"] for r in rows}) == len(rows)
            values = {}
            for metric, legacy, field in (("rr", "RR", "recovery_rate"), ("nr", "NR", "novelty_rate"),
                                          ("vr", "VR", "validity_rate"), ("complex_rr", None, "complex_recovery_rate")):
                observed = [r.get(field, r.get(legacy)) for r in rows]
                observed = [v for v in observed if v is not None]
                values[metric] = np.mean(observed) if observed else None
                if values[metric] is not None:
                    expected = (data["summary"][dataset][method][field]["by_round"][run - 1]
                                if suffix == "boolean_full" else data["means"][method][legacy])
                    assert np.isclose(values[metric], expected, atol=1e-12, rtol=0)
            export.append(dict(source=source.name, dataset=dataset, method=method, run=run,
                               tasks=len(rows), proposals=sum(r["attempts"] for r in rows),
                               rows_reused=sum(bool(r.get("reused")) for r in rows), **values))
    write_csv("earlier_deepseek_runs.csv", export)


def export_tables(runs):
    summaries, effects = [], []
    for (study, dataset, model, run, method, condition), rows in runs.items():
        identifiers = dict(study=study, dataset=dataset, model=model, run=run, method=method,
                           condition=condition)
        if condition != "prompt":
            control = runs[("main" if model == DEEPSEEK else study), dataset, model, run, method, "prompt"]
            effects.append({**identifiers, "stratum": "all", "reference": "prompt", **paired(rows, control)})
        elif study != "depth_control" and method != B:
            references = [B]
            references += [off for on, off in ABLATIONS if method == on]
            for reference in references:
                control = runs[study, dataset, model, run, reference, "prompt"]
                for stratum in ("all", "single_truth", "multiple_truths", "observations_1",
                                "observations_2", "observations_3", "observations_4"):
                    active, reference_rows = subset(rows, stratum), subset(control, stratum)
                    if active:
                        effects.append({**identifiers, "stratum": stratum, "reference": reference,
                                        **paired(active, reference_rows)})
        strata = ("all", "single_truth", "multiple_truths", "observations_1", "observations_2",
                  "observations_3", "observations_4") if study == "main" else ("all",)
        for stratum in strata:
            selected = subset(rows, stratum)
            if selected:
                summaries.append({**identifiers, "stratum": stratum, **metrics(selected)})
    write_csv("summary_metrics.csv", summaries)
    write_csv("paired_effects.csv", effects)


def select(runs, dataset, model, method, run="1", condition="prompt", study="main"):
    return runs[study, dataset, model, run, method, condition]


def main_metrics(runs, dataset, model, method, stratum="all"):
    repeats = ("1", "2", "3") if (dataset, model) == ("easy", Q27) else ("1",)
    by_run = [metrics(subset(select(runs, dataset, model, method, run), stratum)) for run in repeats]
    return {key: np.mean([m[key] for m in by_run]) if by_run[0][key] is not None else None
            for key in by_run[0]}


def q27_summary(runs):
    summary = {}
    for dataset, folders in Q27_RAW.items():
        run_ids = tuple(str(i) for i in range(1, len(folders) + 1))
        usage = {method: dict(calls=0, prompt_tokens=0, completion_tokens=0, length_limited_calls=0)
                 for method in METHODS}
        for folder in folders:
            with (RESULTS / folder / "requests.jsonl").open(encoding="utf-8") as file:
                for line in file:
                    request = json.loads(line)
                    response = request["response"]
                    totals = usage[request["method"]]
                    assert response["model"] == Q27.lower(), folder
                    totals["calls"] += 1
                    for field in ("prompt_tokens", "completion_tokens"):
                        totals[field] += response["usage"][field]
                    totals["length_limited_calls"] += response["choices"][0]["finish_reason"] == "length"

        for method in METHODS:
            by_run = [select(runs, dataset, Q27, method, run) for run in run_ids]
            scores = [metrics(rows) for rows in by_run]
            totals = usage[method]
            assert totals["calls"] == sum(score["proposals"] for score in scores), (dataset, method)
            rr = np.array([100 * score["rr"] for score in scores])
            row = dict(model=Q27, dataset=dataset, method=method, runs=len(scores),
                       tasks_per_run=scores[0]["tasks"], **totals,
                       rr_run_1_pct=rr[0], rr_run_2_pct=rr[1] if len(rr) > 1 else None,
                       rr_run_3_pct=rr[2] if len(rr) > 2 else None,
                       rr_sample_sd_pp=np.std(rr, ddof=1) if len(rr) > 1 else None)
            for field in RATE_FIELDS:
                row[f"{field}_mean_pct"] = (100 * np.mean([score[field] for score in scores])
                                             if scores[0][field] is not None else None)
            for prefix in ("simple", "complex"):
                row[f"{prefix}_targets_recovered"] = sum(score[f"n_recovered_{prefix}"] for score in scores)
                row[f"{prefix}_targets_total"] = sum(r[f"n_{prefix}_ground_truth"] for rows in by_run
                                                     for r in rows.values())
            for reference, label in ((B, "baseline"), (N0, "no_dedup_no_feedback")):
                control = [select(runs, dataset, Q27, reference, run) for run in run_ids]
                deltas = rr - np.array([100 * metrics(rows)["rr"] for rows in control])
                row[f"delta_vs_{label}_pp"] = deltas.mean()
                if len(rr) > 1:
                    row[f"delta_vs_{label}_sample_sd_pp"] = np.std(deltas, ddof=1)
                else:
                    effect = paired(by_run[0], control[0])
                    for field in ("lower_95_pp", "upper_95_pp", "wins", "ties", "losses"):
                        row[f"vs_{label}_{field}"] = effect[field]
            summary[dataset, method] = row
    write_csv("qwen38_27b_summary.csv", list(summary.values()))
    return summary


def format_axes(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", pad=9, fontweight="bold")
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def matrix(ax, values, rows, cols, cmap, norm, style, outline=False):
    x_edges = np.arange(len(cols) + 1) - .5
    y_edges = np.arange(len(rows) + 1) - .5
    img = ax.pcolormesh(x_edges, y_edges, values, cmap=cmap, norm=norm, edgecolors="white", linewidth=.7)
    ax.set(xlim=(x_edges[0], x_edges[-1]), ylim=(y_edges[-1], y_edges[0]))
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    winners = values.argmax(axis=1) if outline else None
    for (i, j), value in np.ndenumerate(values):
        rgb = cmap(norm(value))[:3]
        luminance = np.dot([((c + .055) / 1.055) ** 2.4 if c > .04045 else c / 12.92 for c in rgb],
                           [.2126, .7152, .0722])
        color = "white" if luminance < .23 else INK
        ax.text(j, i, format(value, style), ha="center", va="center", color=color)
        if winners is not None and j == winners[i]:
            ax.add_patch(Rectangle((j - .45, i - .40), .90, .80, fill=False, lw=1.05, ec=color))
    return img


def save(fig, name):
    (OUTPUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUTPUT / "figures" / f"{name}.pdf")
    fig.savefig(OUTPUT / "figures" / f"{name}.png", dpi=600)
    plt.close(fig)


def figure_recovery(runs):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.25))
    fig.subplots_adjust(left=.16, right=.985, bottom=.175, top=.94, hspace=.51)
    cmap = LinearSegmentedColormap.from_list("recovery", ["#f6f8fb", "#9cc7d4", "#3685a1", "#124468"])
    columns = ("Baseline", "Mechanistic\n+ FB", "Mechanistic\n− FB", "Semantic\n+ FB", "Semantic\n− FB",
               "No dedup\n+ FB", "No dedup\n− FB")
    norm = matplotlib.colors.Normalize(0, 65)
    for ax, dataset, letter, title in zip(axes, ("easy", "hard"), ("a", "b"),
                                           ("Depth 1 · 56 tasks", "Depth 2 · 80 tasks")):
        vals = np.array([[100 * main_metrics(runs, dataset, model, method)["rr"] for method in METHODS]
                         for model in MODELS])
        img = matrix(ax, vals, MODELS, columns, cmap, norm, ".1f", outline=True)
        format_axes(ax, letter, title)
    bar = fig.colorbar(img, cax=fig.add_axes([.36, .082, .38, .020]), orientation="horizontal",
                       ticks=[0, 15, 30, 45, 60])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2, pad=2)
    bar.set_label("Mechanistic recovery rate (%)", labelpad=2)
    save(fig, "fig1_recovery_overview")


def figure_hard(runs):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.48, 1]})
    fig.subplots_adjust(left=.153, right=.985, bottom=.30, top=.83, wspace=.19)
    format_axes(ax, "a", "Paired change on depth-2 tasks")
    ax.axvline(0, c=GRAY, lw=.85)
    ys = np.arange(len(MODELS))
    for i, model in enumerate(MODELS):
        for method, offset, color, symbol in ((M, -.14, TEAL, "o"), (S, .14, ORANGE, "s")):
            p = paired(select(runs, "hard", model, method), select(runs, "hard", model, B))
            x, lo, hi = (p[k] for k in ("delta_rr_pp", "lower_95_pp", "upper_95_pp"))
            ax.errorbar(x, i + offset, xerr=[[x - lo], [hi - x]], fmt=symbol, color=color,
                        ms=4.8, elinewidth=1.2, capsize=2)
    ax.set(yticks=ys, yticklabels=MODELS, ylim=(len(MODELS) - .5, -.5), xlabel="RR difference vs baseline (pp)")
    ax.set_xlim(-7, 11)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=.55, color=GRID)
    labels = ("Mechanistic + FB", "Semantic + FB")
    ax.legend(handles=[Line2D([], [], marker=marker, color=color, lw=0, label=label)
                       for marker, color, label in zip(("o", "s"), (TEAL, ORANGE), labels)],
              loc="upper center", bbox_to_anchor=(.5, -.29), ncols=2, frameon=False, columnspacing=1)
    vals = np.array([[100 * main_metrics(runs, "hard", model, method)["complex_rr"] for method in (B, M, S)]
                     for model in MODELS])
    cmap = LinearSegmentedColormap.from_list("complex", ["#f7f8fb", "#e7a777", "#994020"])
    matrix(bx, vals, [""] * len(MODELS), ("Baseline", "Mech.\n+ FB", "Sem.\n+ FB"), cmap,
           matplotlib.colors.Normalize(0, 9), ".2f")
    format_axes(bx, "b", "Complex-mechanism RR (%)")
    bx.set_xlabel("76 tasks with complex targets", labelpad=5)
    save(fig, "fig2_hard_mechanisms")


def figure_feedback(runs):
    fig = plt.figure(figsize=(7.2, 4.9))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.06), left=.156, right=.985,
                            bottom=.15, top=.925, hspace=.61, wspace=.18)
    cmap = LinearSegmentedColormap.from_list("feedback", ["#9f461f", "#f8f8f7", "#006a80"])
    panels = (("easy", "multiple_truths", "a", "Depth 1 · 24 multi-target tasks"),
              ("hard", "all", "b", "Depth 2 · 80 tasks"))
    for col, (dataset, stratum, letter, title) in enumerate(panels):
        ax = fig.add_subplot(grid[0, col])
        vals = np.array([[100 * (main_metrics(runs, dataset, model, on, stratum)["rr"]
                                  - main_metrics(runs, dataset, model, off, stratum)["rr"])
                          for on, off in ABLATIONS] for model in MODELS])
        matrix(ax, vals, MODELS if col == 0 else [""] * 5, ("Mechanistic", "Semantic", "No dedup"),
               cmap, TwoSlopeNorm(vmin=-14, vcenter=0, vmax=14), "+.1f")
        format_axes(ax, letter, title)
        ax.set_xlabel("Feedback on − off (pp)", labelpad=4)
    ax = fig.add_subplot(grid[1, :])
    format_axes(ax, "c", "Qwen3.8-27B · Three depth-1 runs")
    for method, label, color, shape, line in ((B, "Baseline", GRAY, "o", "--"),
            (N0, "No dedup − FB", "#7b65a3", "D", ":"), (M, "Mechanistic + FB", TEAL, "^", "-"),
            (S, "Semantic + FB", ORANGE, "s", "-")):
        scores = [100 * metrics(select(runs, "easy", Q27, method, str(i)))["rr"] for i in (1, 2, 3)]
        ax.plot([1, 2, 3], scores, color=color, marker=shape, linestyle=line, lw=1.25, ms=4.7, label=label)
    ax.set(xlim=(.7, 3.3), xticks=[1, 2, 3], xticklabels=["Run 1", "Run 2", "Run 3"],
           ylim=(34, 56), yticks=[35, 40, 45, 50, 55], ylabel="RR (%)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=.55, color=GRID)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.235), ncols=4, frameon=False)
    save(fig, "fig3_feedback_and_repeats")


def figure_depth(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.24))
    fig.subplots_adjust(left=.257, right=.985, bottom=.238, top=.86, wspace=.22)
    conditions = ("prompt", "examples", "grammar")
    marks = ("o", "s", "D")
    colors = (GRAY, ORANGE, TEAL)
    pairs = [(model, method) for model in (Q4, Q9, DEEPSEEK) for method in (B, M)]
    ticks = [f"{model} · {'Baseline' if method == B else 'Mech. + FB'}" for model, method in pairs]
    for i, (ax, field, title, label) in enumerate(zip(axes, ("rr", "over_depth_rate"),
            ("Recovery rate", "Depth violations"), ("Macro RR (%)", "Over-depth calls (%)"))):
        format_axes(ax, "ab"[i], title)
        for j, (model, method) in enumerate(pairs):
            for offset, condition, marker, color in zip((-.22, 0, .22), conditions, marks, colors):
                study = "main" if (model, condition) == (DEEPSEEK, "prompt") else "depth_control"
                key = (study, "easy", model, "1", method, condition)
                if key in runs:
                    x = 100 * metrics(runs[key])[field]
                    ax.hlines(j + offset, 0, x, color=color, linewidth=.95, alpha=.66)
                    ax.plot(x, j + offset, marker=marker, color=color, ms=4.3, clip_on=False)
        ax.set(yticks=range(len(pairs)), yticklabels=ticks if i == 0 else [""] * len(pairs),
               ylim=(5.65, -.60), xlim=(0, 64 if i == 0 else 55), xlabel=label)
        ax.set_axisbelow(True)
        ax.grid(axis="x", linewidth=.55, color=GRID)
        for y in (1.5, 3.5):
            ax.axhline(y, color=GRID, lw=.55)
    fig.legend(handles=[Line2D([], [], marker=m, color=c, label=l, lw=.95)
                        for m, c, l in zip(marks, colors, ("Prompt", "Depth examples", "Grammar"))],
               loc="lower center", bbox_to_anchor=(.60, .046), frameon=False, ncols=3)
    save(fig, "fig4_depth_constraints")


def figure_q27(summary):
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(7.2, 3.4),
                                     gridspec_kw={"width_ratios": [1.35, 1.08, .76]})
    fig.subplots_adjust(left=.168, right=.978, bottom=.25, top=.84, wspace=.20)
    fig.suptitle(Q27, x=.168, y=.975, ha="left", fontsize=10, fontweight="bold")
    ys = np.arange(len(METHODS))
    labels = ("Baseline", "Mech. + FB", "Mech. − FB", "Semantic + FB", "Semantic − FB", "No dedup + FB", "No dedup − FB")
    easy = [summary["easy", m] for m in METHODS]
    hard = [summary["hard", m] for m in METHODS]
    for panel, letter, title in zip((ax, bx, cx), "abc",
                                    ("Depth 1 (56 tasks)", "Depth 2 (80 tasks)", "Complex RR")):
        format_axes(panel, letter, title)
        panel.set(yticks=ys, yticklabels=labels if panel == ax else [""] * len(ys), ylim=(6.5, -.5))
        for y in (0.5, 2.5, 4.5):
            panel.axhline(y, color=GRID, lw=.55)

    run_styles = ((1, -.17, TEAL, "o"), (2, 0, ORANGE, "s"), (3, .17, "#7a619b", "^"))
    ax.plot([row["rr_mean_pct"] for row in easy], ys, "|", color=INK, ms=14, mew=1.4, label="Mean")
    for run, offset, color, marker in run_styles:
        ax.plot([row[f"rr_run_{run}_pct"] for row in easy], ys + offset, marker, ms=4.8,
                color=color, markeredgecolor="white", markeredgewidth=.45, label=f"Run {run}")
    ax.set(xlim=(34, 56), xticks=[35, 40, 45, 50, 55], xlabel="Macro RR (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(.50, -.225), ncols=4, frameon=False,
              columnspacing=.58, handletextpad=.20, handlelength=1)

    overall = [row["rr_mean_pct"] for row in hard]
    simple = [row["simple_rr_mean_pct"] for row in hard]
    bx.hlines(ys, overall, simple, lw=.95, color="#a7b7c7")
    bx.plot(overall, ys, "o", color=GRAY, ms=4.5, label="Overall")
    bx.plot(simple, ys, "D", color=TEAL, ms=4.5, label="Simple")
    bx.set(xlim=(8, 31), xticks=[10, 15, 20, 25, 30], xlabel="Macro RR (%)")
    bx.legend(loc="upper center", bbox_to_anchor=(.50, -.225), ncols=2, frameon=False,
              columnspacing=.85, handletextpad=.3, handlelength=1)
    for panel in (ax, bx):
        panel.set_axisbelow(True)
        panel.grid(axis="x", linewidth=.55, color=GRID)

    cx.set(xlim=(0, 1), xticks=[], xlabel="Recovered · Macro RR")
    for y, row in zip(ys, hard):
        count, total = row["complex_targets_recovered"], row["complex_targets_total"]
        cx.text(.5, y, f"{count}/{total}  ·  {row['complex_rr_mean_pct']:.2f}%", ha="center", va="center")
    save(fig, "fig5_qwen38_27b")


def main():
    runs = load_runs()
    export_tables(runs)
    archived_deepseek()
    q27 = q27_summary(runs)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "legend.fontsize": 7.5, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42, "pdf.compression": 9, "savefig.facecolor": "white"})
    figure_recovery(runs)
    figure_hard(runs)
    figure_feedback(runs)
    figure_depth(runs)
    figure_q27(q27)
    print(f"Verified {sum(map(len, runs.values()))} task rows across {len(runs)} CSV groups.")


if __name__ == "__main__":
    main()
