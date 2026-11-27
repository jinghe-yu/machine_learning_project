"""Compare prompt examples and grammar-constrained decoding on the Boolean benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import keyword
import os
import re
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from benchmark_utils import load_tasks, score_hypotheses, select_tasks, task_target_count, to_observations
from boolean_genetics import (
    _expression_depth,
    _parse_expression,
    sample_hypotheses,
    validate_expression_constraints,
)


LOCAL_MODELS = (
    ("qwen3.5-4b-q4_k_m", "http://127.0.0.1:8081/v1"),
    ("qwen3.5-9b-q4_k_m", "http://127.0.0.1:8080/v1"),
)
CONDITIONS = ("prompt", "examples", "grammar")
METHODS = ("baseline", "ours_mechanistic")
OPERATORS = {
    "AND": '{arg} (" and " {arg})+',
    "OR": '{arg} (" or " {arg})+',
    "NOT": '"not " {arg}',
    "XOR": '{arg} " ^ " {arg}',
    "NOR": '"NOR(" {arg} (", " {arg})+ ")"',
}


def depth_grammar(variables: list[str], operators: list[str], max_depth: int) -> str:
    """Render depth-bounded Boolean syntax without task observations or answers."""
    if not variables or any(
        not name.isidentifier() or keyword.iskeyword(name) or name.upper() in OPERATORS
        for name in variables
    ):
        raise ValueError("Expected non-reserved Python identifiers as variables")
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("Depth ablations require a nonnegative max_depth")
    allowed = sorted({operator.upper() for operator in operators})
    if not allowed or set(allowed).difference(OPERATORS):
        raise ValueError("Expected a nonempty set of supported Boolean operators")
    root = f"expression-{max_depth}" if max_depth else "variable"
    names = " | ".join(json.dumps(name, ensure_ascii=False) for name in variables)
    rules = [f"root ::= {root}", f"variable ::= {names}"]
    for depth in range(1, max_depth + 1):
        argument = "variable" if depth == 1 else f"operand-{depth - 1}"
        options = ["variable", *(OPERATORS[op].format(arg=argument) for op in allowed)]
        rules.append(f"expression-{depth} ::= {' | '.join(options)}")
        if depth < max_depth:
            rules.append(f'operand-{depth} ::= variable | "(" expression-{depth} ")"')
    return "\n".join(rules) + "\n"


def depth_examples(operators: list[str], max_depth: int) -> str:
    templates = {
        "AND": "{left} and b",
        "OR": "{left} or b",
        "NOT": "not {left}",
        "XOR": "{left} ^ b",
        "NOR": "NOR({left}, b)",
    }
    formats = [templates[operator] for operator in templates if operator in operators]
    legal = "a"
    for _ in range(max_depth):
        legal = formats[0].format(left=legal if legal == "a" else f"({legal})")
    outer = templates["NOT"] if "NOT" in operators else formats[0]
    too_deep = outer.format(left=f"({legal})" if max_depth else legal)
    return (
        "\nDepth examples with illustrative variables a and b. "
        f"`{legal}` has depth {max_depth} and is allowed by the depth rule. "
        f"`{too_deep}` has depth {max_depth + 1} and exceeds the limit. "
        "Use only the listed task variables in your answer.\n"
    )


def diagnosis(candidate: str | None, variables: list[str], operators: list[str], depth: int) -> str:
    if candidate is None:
        return "parse_failure"
    try:
        expression = _parse_expression(candidate)
    except (SyntaxError, ValueError, TypeError):
        return "parse_failure"
    if _expression_depth(expression) > depth:
        return "over_depth"
    try:
        validate_expression_constraints(
            candidate, variables, allowed_operators=operators, max_depth=depth
        )
    except (SyntaxError, ValueError, TypeError):
        return "other_constraint"
    return "in_space"


def model_endpoint(value: str) -> tuple[str, str]:
    model, _, endpoint = value.partition("=")
    url = urlsplit(endpoint)
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", model)
        or url.scheme != "http"
        or url.hostname not in {"localhost", "127.0.0.1", "::1"}
        or url.username is not None
        or url.query
        or url.fragment
        or url.path.rstrip("/") != "/v1"
    ):
        raise argparse.ArgumentTypeError("Expected MODEL_ID=http://127.0.0.1:PORT/v1")
    return model, endpoint


def run_task(client, model, method, condition, task, metadata, budget, args, grammar, examples):
    task_id = task["observation_set_id"]
    calls = []

    def complete(prompt: str, temperature: float) -> str:
        seed_material = f"{args.generation_seed}:{task_id}:{len(calls)}".encode()
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big") & 0x7FFFFFFF
        if condition == "examples":
            prompt += examples
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You solve Boolean-genetics hypothesis discovery tasks."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            seed=seed,
            max_tokens=args.max_tokens,
            extra_body={"grammar": grammar} if condition == "grammar" else None,
        )
        choice = response.choices[0]
        calls.append({
            "seed": seed,
            "seconds": time.perf_counter() - started,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "finish_reason": choice.finish_reason,
        })
        return choice.message.content or ""

    variables = metadata["variables"]
    operators = metadata["operators"]
    depth = metadata["max_depth"]
    ours = method == "ours_mechanistic"
    sampled = sample_hypotheses(
        to_observations(task),
        complete,
        budget,
        max_attempts=budget,
        start_temperature=args.temperature,
        end_temperature=args.temperature,
        allowed_operators=operators,
        max_depth=depth,
        allow_constants=False,
        deduplication="mechanistic" if ours else "none",
        include_history=ours,
        stop_on_target=False,
        mechanistic_options=metadata.get("mechanistic_opts"),
    )
    score = score_hypotheses(task, sampled.proposals, metadata, proposal_count=sampled.attempts)
    diagnostics = [diagnosis(candidate, variables, operators, depth) for candidate in sampled.proposals]
    if condition == "grammar" and any(
        label != "in_space" and call["finish_reason"] != "length"
        for label, call in zip(diagnostics, calls)
    ):
        raise RuntimeError(f"Grammar returned a complete out-of-space answer for {model}/{task_id}")
    score.update(
        model=model,
        method=method,
        condition=condition,
        n_over_depth=diagnostics.count("over_depth"),
        n_parse_failure=diagnostics.count("parse_failure"),
        n_other_constraint=diagnostics.count("other_constraint"),
        n_truncated=sum(call["finish_reason"] == "length" for call in calls),
        request_seconds=sum(call["seconds"] for call in calls),
        proposal_candidates_json=json.dumps(sampled.proposals),
        raw_responses_json=json.dumps(sampled.raw_responses),
        diagnostics_json=json.dumps(diagnostics),
        rejection_counts=json.dumps(sampled.rejection_counts),
        calls_json=json.dumps(calls),
    )
    return score


def read_cell(path, model, method, condition, budgets):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    result = {}
    for row in rows:
        task_id = row["observation_set_id"]
        if (
            task_id in result
            or (row["model"], row["method"], row["condition"]) != (model, method, condition)
            or int(row["n_proposals"]) != budgets.get(task_id)
            or len(json.loads(row["calls_json"])) != budgets[task_id]
        ):
            raise ValueError(f"Invalid or unpaired task record in {path}: {task_id}")
        result[task_id] = row
    return result


def append_row(path, score):
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(score))
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(score)
        file.flush()
        os.fsync(file.fileno())


def write_summary(output, cells, manifest):
    rows = []
    for (model, method, condition), cell in cells.items():
        plain = cells[model, method, "prompt"]
        proposals = sum(int(item["n_proposals"]) for item in cell.values())
        over_depth = sum(int(item["n_over_depth"]) for item in cell.values())
        rows.append({
            "model": model,
            "method": method,
            "condition": condition,
            "tasks": len(cell),
            "proposals": proposals,
            "n_over_depth": over_depth,
            "over_depth_rate": over_depth / proposals,
            "n_parse_failure": sum(int(item["n_parse_failure"]) for item in cell.values()),
            "n_other_constraint": sum(int(item["n_other_constraint"]) for item in cell.values()),
            "n_truncated": sum(int(item["n_truncated"]) for item in cell.values()),
            "mean_VR": statistics.fmean(float(item["validity_rate"]) for item in cell.values()),
            "mean_NR": statistics.fmean(float(item["novelty_rate"]) for item in cell.values()),
            "mean_RR": statistics.fmean(float(item["recovery_rate"]) for item in cell.values()),
            "delta_RR_pp": 100 * statistics.fmean(
                float(item["recovery_rate"]) - float(plain[task_id]["recovery_rate"])
                for task_id, item in cell.items()
            ),
            "request_seconds": sum(float(item["request_seconds"]) for item in cell.values()),
        })
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 布尔表达式深度约束对照",
        "",
        f"本轮在同一批 {len(manifest['budgets'])} 道布尔题上比较三种解码配置。"
        f"数据集深度上限为 {manifest['max_depth']}，每题各组调用预算与机制真值数一致。"
        "普通提示和语法约束组使用相同的提示，示例组在此基础上增加一对深度示例。",
        "",
        "| 模型 | 方法 | 条件 | 超深次数 | 超深率 | VR | RR | ΔRR（百分点） | 请求耗时（秒） |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['condition']} | "
            f"{row['n_over_depth']}/{row['proposals']} | {row['over_depth_rate']:.1%} | "
            f"{row['mean_VR']:.1%} | {row['mean_RR']:.1%} | {row['delta_RR_pp']:+.2f} | "
            f"{row['request_seconds']:.1f} |"
        )
    lines.extend([
        "",
        "超深率按所有模型调用计算。可以解析且超过深度上限的候选计为超深。"
        "解析失败、其他空间约束违规和截断次数单独记录。VR、NR 和 RR 逐题计算后取平均。"
        "ΔRR 使用同模型、同方法下的普通提示作配对对照。",
        "",
        "语法只使用变量、允许的算子和深度上限，按规范的 Python 表达式形式限制生成。"
        "模型仍需根据观测选择候选式。各配置共享逐题逐次调用的生成种子，本轮为单次配对实验。",
        "",
        "[汇总 CSV](summary.csv) · [运行配置](manifest.json) · [解码语法](grammar.gbnf)",
        "",
        "`raw/` 包含逐题分数、原始回答、深度诊断、生成种子、token 数和请求耗时。",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/boolean_dataset.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", type=model_endpoint, metavar="MODEL=URL")
    parser.add_argument("--samples", type=int, help="Use all tasks by default")
    parser.add_argument("--task-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    models = args.model or list(LOCAL_MODELS)
    if len({model for model, _ in models}) != len(models):
        parser.error("Model IDs must be distinct")
    if args.max_tokens < 1 or not 0 <= args.temperature <= 2:
        parser.error("Use a positive token cap and temperature between 0 and 2")

    metadata, all_tasks = load_tasks(args.dataset)
    tasks = select_tasks(all_tasks, len(all_tasks) if args.samples is None else args.samples, args.task_seed)
    budgets = {
        task["observation_set_id"]: task_target_count(task, metadata["variables"], metadata.get("mechanistic_opts"))
        for task in tasks
    }
    if len(budgets) != len(tasks) or any(budget < 1 for budget in budgets.values()):
        raise ValueError("Expected distinct tasks with positive proposal budgets")
    depth = metadata["max_depth"]
    grammar = depth_grammar(metadata["variables"], metadata["operators"], depth)
    examples = depth_examples([op.upper() for op in metadata["operators"]], depth)
    manifest = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "experiment_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "models": dict(models),
        "max_depth": depth,
        "task_seed": args.task_seed,
        "generation_seed": args.generation_seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "conditions": list(CONDITIONS),
        "methods": list(METHODS),
        "budgets": budgets,
        "depth_examples": examples.strip(),
    }
    output = args.output
    manifest_file = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume or not manifest_file.exists() or json.loads(manifest_file.read_text()) != manifest:
            parser.error("Output already contains a different or non-resumable run")
    (output / "raw").mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "grammar.gbnf").write_text(grammar, encoding="utf-8")

    cells = {}
    for model, endpoint in models:
        with OpenAI(
            api_key="not-needed", base_url=endpoint, timeout=90, max_retries=2,
            http_client=httpx.Client(trust_env=False),
        ) as client:
            if not any(item.id == model for item in client.models.list().data):
                raise ValueError(f"{endpoint} does not expose {model}")
            for method in METHODS:
                for condition in CONDITIONS:
                    path = output / "raw" / f"{model}_{method}_{condition}.csv"
                    cell = read_cell(path, model, method, condition, budgets)
                    print(f"{model} / {method} / {condition}: {len(cell)}/{len(tasks)} already saved", flush=True)
                    for index, task in enumerate(tasks, start=1):
                        task_id = task["observation_set_id"]
                        if task_id not in cell:
                            score = run_task(
                                client, model, method, condition, task, metadata, budgets[task_id],
                                args, grammar, examples,
                            )
                            append_row(path, score)
                            cell[task_id] = score
                        if index % 14 == 0 or index == len(tasks):
                            print(f"  {index}/{len(tasks)} tasks saved", flush=True)
                    cells[model, method, condition] = cell
    write_summary(output, cells, manifest)
    print(f"Report saved to {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
