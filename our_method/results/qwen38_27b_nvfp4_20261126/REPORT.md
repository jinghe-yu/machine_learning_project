# Qwen3.8-27B NVFP4：56 题七组实验

已在 RTX 5090 上通过现有 vLLM 服务完成全部七组实验，每组 56 题、90 次调用，共 392 条逐题结果和 630 次调用。运行时间为 2026-11-26 06:06:00–06:13:46 UTC。

本轮机制恢复率 RR 最高的是语义去重＋有反馈（50.60%）；Baseline 为 36.76%，相差 +13.84 个百分点。默认机制去重＋有反馈为 40.62%，相差 +3.87 个百分点。以下均为逐题指标的算术平均值。

## 七组结果

- [Baseline](raw/baseline.csv)：VR 47.32%，NR 50.00%，RR 36.76%；90 次调用。
- [机制去重＋有反馈](raw/ours_mechanistic.csv)：VR 42.41%，NR 53.27%，RR 40.62%；90 次调用。
- [机制去重＋无反馈](raw/ours_mechanistic_no_feedback.csv)：VR 53.87%，NR 53.87%，RR 43.90%；90 次调用。
- [语义去重＋有反馈](raw/ours_semantic.csv)：VR 51.49%，NR 59.08%，RR 50.60%；90 次调用。
- [语义去重＋无反馈](raw/ours_semantic_no_feedback.csv)：VR 50.15%，NR 52.08%，RR 41.52%；90 次调用。
- [不去重＋有反馈](raw/ours_none.csv)：VR 58.63%，NR 52.98%，RR 45.54%；90 次调用。
- [不去重＋无反馈](raw/ours_none_no_feedback.csv)：VR 55.36%，NR 55.21%，RR 44.35%；90 次调用。

VR 衡量候选式是否满足语法、任务约束和观测；NR 统计假设空间内不同机制的比例，包括与观测不符的结构合法候选；RR 衡量机制真值的恢复比例。

## 结果解释与限制

- 这是六种配置加一次独立 Baseline，不是七种不同算法。Baseline 与“不去重＋无反馈”的所有请求载荷已核对完全相同；两次独立随机运行的 RR 却相差 +7.59 个百分点，说明随机波动明显，不能把所有组间差异归因于方法。
- 固定了任务选择种子 42；生成未固定解码种子，各组只有一次运行，尚无多次重复或显著性结论。
- 32 题只有一个机制真值，每组只调用一次，后续历史反馈没有生效机会。24 道多真值题中，Baseline、机制去重＋有反馈、语义去重＋有反馈的 RR 分别为 31.60%、44.79%、47.22%。
- 所有配置使用相同温度 0.7、最大输出 256 tokens、单并发；每题预算等于机制真值数，拒绝和重复也消耗预算。请求均设置 enable_thinking=false，但部分实际回复仍包含解释或分析文字。
- 630 次回复中 626 次以 stop 结束，4 次达到 token 上限（baseline、机制去重无反馈、语义去重无反馈、不去重无反馈各一次）。这些回复按原有提取和计分规则保留，没有选择性重跑或删除。
- 数据是仓库自带 boolean_dataset.json 的全部 56 个观测集，变量 x/y、NOT/AND/OR、深度 1、六个假设；这是该小型子集的结果，不能推广为论文整个 benchmark 的成绩。
- 27B 使用当前提示修正：操作符语法示例只列任务允许的操作符。旧 4B/9B 七组报告使用此前提示，且为 Q4_K_M / llama.cpp / Metal；本轮为 NVFP4 / vLLM / RTX 5090，因此跨模型差异不能仅归因于参数规模。
- 新增 depth_constraint_experiment.py 的普通提示／深度示例／语法约束实验未混入本轮。此前 10 题 smoke 和已中止的三方法实验均独立保存，不计入这 630 次调用。

## 可复核文件

- [总体汇总](summary.csv)、[真值数量分层](strata.csv)、[原始汇总](raw/summary.json)。
- [运行配置及数据／源码哈希](raw/run_config.json)、[实际硬件记录](hardware.json)、[完整请求与回复](raw/requests.jsonl)、[冻结源码](source.tar.gz)。
- [离线复核脚本](verify_results.py)、[复核结果](verification.json)、[文件校验和](checksums.json)。复核使用冻结源码重放全部 630 条回复，逐一检查请求提示、反馈／去重规则、候选、拒绝、逐题数值指标、汇总和 token 用量。

复核命令（仅标准库，不调用模型）：

```bash
python3 our_method/results/qwen38_27b_nvfp4_20261126/verify_results.py our_method/results/qwen38_27b_nvfp4_20261126
```
