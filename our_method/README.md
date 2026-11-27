# 扩展实验

安装、离线演示和常用评测命令见[仓库首页](../README.md)。以下命令在 `our_method/` 中执行。

| 命令 | 用途 |
| --- | --- |
| `python -m experiments.depth_constraint_experiment --help` | 用支持 GBNF 的 llama.cpp 服务比较普通提示、深度示例和语法约束 |
| `python -m experiments.run_deepseek_team_comparison --help` | 运行 DeepSeek 七组对照与深度示例实验 |
| `python -m experiments.pilot_deepseek --help` | 运行 DeepSeek 先导实验 |
| `python -m experiments.run_boolean_experiments --help` | 在三套题集上运行早期实验的五组对照配置 |

DeepSeek 脚本默认使用模拟回复。真实调用需显式传入 `--live` 并设置 `DEEPSEEK_API_KEY`，也可将此变量写入本地 `.env`。报告包含已完成实验的模型、题集、提示和预算配置。[DeepSeek 七组对照](results/deepseek_flash_new_prompt_20261126/REPORT.md) · [深度约束报告](results/depth_constraints_20261126/REPORT.md)

常规评测入口还支持 `OPENROUTER_API_KEY` 和 `AINATIVE_API_KEY`。`LLM_MODEL` 可指定模型。仅设置 OpenRouter 密钥时，默认使用 `openrouter/free`。没有显式配置兼容端点或 OpenRouter 时，程序走 AINative，模型默认是 `gpt-oss-120b`，也可用 `AINATIVE_MODEL` 指定。
