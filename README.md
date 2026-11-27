# 大语言模型的布尔假设恢复实验

给定几条关于 `x` 和 `y` 的输入输出观测，多个布尔表达式都可能符合条件。本项目基于 HypoSpace 的布尔任务，在相同的模型调用预算下，比较独立采样与带去重和反馈的采样方法，衡量模型能找回多少种不同结构的表达式。

## 快速开始

推荐使用 Python 3.12 或更新版本。在仓库根目录创建虚拟环境并安装依赖。

```bash
cd our_method
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1` 激活环境。以下命令均在 `our_method/` 中执行。

```bash
python baseline.py --mock --samples 3 --seed 42 --output outputs/demo/baseline.csv --resume
python evaluate_ours.py --mock --samples 3 --seed 42 --output outputs/demo/ours.csv --resume
python compare_results.py --baseline outputs/demo/baseline.csv --ours outputs/demo/ours.csv --summary outputs/demo/summary.csv --chart outputs/demo/recovery.png
python -m unittest discover
```

这组命令使用模拟回复，不需要模型、API 密钥或网络请求。生成的逐题 CSV、指标汇总和图表均在 `outputs/demo/`，仅用于验证运行流程。

## 使用真实模型

启动一个 OpenAI 兼容的推理服务，将 `LLM_MODEL` 替换为服务提供的模型 ID。需要鉴权时，在本地设置 `LLM_API_KEY`。PowerShell 可用 `$env:LLM_MODEL = "your-model-id"` 的写法设置环境变量。

```bash
export LLM_BASE_URL=http://127.0.0.1:8080/v1
export LLM_MODEL=your-model-id
python baseline.py --samples 56 --seed 42 --output outputs/real/baseline.csv
python evaluate_ours.py --samples 56 --seed 42 --output outputs/real/ours.csv
python compare_results.py --baseline outputs/real/baseline.csv --ours outputs/real/ours.csv --summary outputs/real/summary.csv --chart outputs/real/recovery.png
```

基线独立生成候选。实验方法默认按表达式结构去重，并把已接受的候选和近期拒绝原因反馈给模型。`--dedup` 可选 `mechanistic`（结构）、`semantic`（真值表）或 `none`，`--no-feedback` 关闭反馈。每题的默认调用次数等于该题的机制真值数，无效和重复回复都消耗预算。比较时两组采用相同题集、种子和预算。难题使用 `--dataset data/boolean_full_depth2.json --samples 80`，另选输出路径。遇到中断可用原参数和输出路径加 `--resume` 继续。

| 指标 | 计算方式 |
| --- | --- |
| 有效率（VR） | 符合表达式限制且满足全部观测的候选数 ÷ 调用次数 |
| 新颖率（NR） | 假设空间内不同机制的候选数 ÷ 调用次数 |
| 恢复率（RR） | 找回的不同机制真值数 ÷ 该题的机制真值总数 |

## 查看与复现实验结果

综合报告汇总了五款模型在普通题和困难题上的表现，以及去重、反馈和解码深度约束的对照。已保存的逐题结果可直接离线复算，无需 Git 历史或推理服务。脚本会在报告目录重新生成统计表和图表。[综合报告](our_method/results/paper_analysis/REPORT.md)

```bash
python results/paper_analysis/reproduce.py
```

## 仓库结构

```text
our_method/
├── baseline.py、evaluate_ours.py、compare_results.py   实验与绘图入口
├── benchmark_cli.py、benchmark_utils.py               评测流程与评分
├── boolean_genetics.py                               表达式校验与采样
├── llm_caller.py、mock_llm.py                          模型接口与模拟模型
├── data/                                             题集
├── experiments/                                      扩展实验
├── results/                                          原始实验记录与综合分析
└── tests/                                            单元测试
references/
├── hypospace/                                        HypoSpace 参考代码
└── HypoSpace-ICML-2026.pdf                            参考论文
```

题集配置、额外模型后端和扩展实验入口见各自说明。[题集说明](our_method/data/README.md) · [扩展实验说明](our_method/README.md)

## 引用

题集和评估任务来自 Chen 等人的 HypoSpace。参考代码中的共享配置和模块在本仓库中合并维护。[ICML 2026 论文](https://proceedings.mlr.press/v306/chen26bz.html) · [HypoSpace 上游仓库](https://github.com/CTT-Pavilion/_HypoSpace)
