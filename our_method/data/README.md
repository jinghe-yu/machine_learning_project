# 布尔题集

| 题集 | 题数 | 最大深度 | 算子 | 跑完一组的默认调用数 |
| --- | ---: | ---: | --- | ---: |
| [Basic](boolean_basic_depth1.json) | 35 | 1 | AND、OR | 60 |
| [Extended，默认](boolean_dataset.json) | 56 | 1 | AND、OR、NOT | 90 |
| [Full，困难题](boolean_full_depth2.json) | 80 | 2 | AND、OR、NOT、NOR | 975 |

三套题都使用变量 `x`、`y`。`metadata` 记录算子、深度上限和机制去重规则。`datasets_by_n_observations` 按观测条数分组。每题的 `observation_set_id` 用于对齐结果，`observations` 是模型能看到的观测，`ground_truth_expressions` 保存评分所用的公式和机制键。

## 困难题

80 道题按观测条数 1、2、3、4 分组，分别有 8、24、32、16 题。76 题至少有一个复杂真值。按数据集原始公式中的运算符数划分，简单解有 0–3 个运算符，复杂解有 4–5 个。

候选式用 `NOR(a, b)` 表示 NOR，例如 `not NOR(x, y)`。题集可能将等价的真值公式写为 `NOT (a OR b)`，评分通过机制键匹配。

困难题集的 SHA-256 为 `441e1d4d96a0f6f987e0355a03d4f66869b99dcab5d298b6343cbee6b0356378`。运行方法见[仓库首页](../../README.md#使用真实模型)。

## 重新生成困难题

从 `our_method/` 运行以下命令。对比实验使用仓库内带校验值的题集。

```bash
python ../references/hypospace/boolean/boolean_dataset.py --operators full --max-depth 2 --seed 33550336 --output outputs/boolean_full_depth2_regenerated.json
```
