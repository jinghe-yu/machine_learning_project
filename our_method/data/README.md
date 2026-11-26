# Boolean 困难题集

[boolean_full_depth2.json](boolean_full_depth2.json) 是用仓库内的 HypoSpace 生成器生成的困难题集。它采用论文表 3 的 `full` 算子档位，深度和种子使用下面记录的配置。对照论文表 3 或组内已有结果时，请以双方实际使用的题目 ID、生成配置和评分口径为准。[HypoSpace 论文 §5.3](https://arxiv.org/html/2510.15614#S5.SS3)

- 生成配置为 `--operators full --max-depth 2 --seed 33550336`。变量为 `x`、`y`，算子为 `AND`、`OR`、`NOT`、`NOR`。
- 共 80 道题。按观测条数 1、2、3、4 分组，分别有 8、24、32、16 题。
- 76 道题含有至少一个复杂真值。按这份 JSON 的公式写法计数，简单解含 0–3 个运算符，复杂解含 4–5 个运算符。
- SHA-256 为 `441e1d4d96a0f6f987e0355a03d4f66869b99dcab5d298b6343cbee6b0356378`。

## 题目结构

`metadata` 记录算子、深度上限和机制去重规则。`datasets_by_n_observations` 按观测条数分组。每题的 `observation_set_id` 用于对齐组内结果，`observations` 是给模型的输入输出观测，`ground_truth_expressions` 保存用于评分的公式和 `mechanistic_key`。

## 运行对照实验

按[实验说明](../README.md#setup)安装依赖，从 `our_method` 目录运行。假设本地 OpenAI 兼容推理服务监听在 `127.0.0.1:8080`，把 `your-local-model-id` 换成服务返回的模型 ID。

```bash
export LLM_BASE_URL=http://127.0.0.1:8080/v1
export LLM_MODEL=your-local-model-id

python baseline.py --dataset data/boolean_full_depth2.json --samples 80 --seed 42 --temperature 0.7 --output results/hard_baseline.csv
python evaluate_ours.py --dataset data/boolean_full_depth2.json --samples 80 --seed 42 --temperature 0.7 --dedup mechanistic --output results/hard_ours_mechanistic.csv
python compare_results.py --baseline results/hard_baseline.csv --ours results/hard_ours_mechanistic.csv --summary results/hard_summary.csv --chart results/hard_metrics.png
```

这组命令按每题的机制真值数分配调用预算，每种方法共 975 次调用。`hard_summary.csv` 中的 `mean_RR` 是困难题整体 RR，`mean_complex_RR` 是按含复杂真值的题逐题计算后取平均的复杂解 RR。

候选式中的 `NOR` 请使用 `NOR(a, b)` 写法，例如 `not NOR(x, y)`。现有提示会要求这种写法，评分器也会把它作为一层算子校验。生成器保存的真值公式可能以展开的 `NOT (a OR b)` 表示同一个 `NOR`，评分通过 `mechanistic_key` 匹配，复杂解分组使用 JSON 中原始公式的运算符计数。

## 重新生成

安装 NumPy 和 SymPy 后，从 `our_method` 目录运行下列命令。对比实验请优先共用上面带 SHA-256 的 JSON。

```bash
python ../_HypoSpace-main/boolean/boolean_dataset.py --operators full --max-depth 2 --seed 33550336 --output data/boolean_full_depth2_regenerated.json
```
