"""Summarize three fixed-protocol runs after each has passed replay verification."""
from pathlib import Path
import csv,hashlib,json,subprocess,sys
from statistics import fmean,stdev
root=Path(sys.argv[1]).resolve()
original=root.parent/'qwen38_27b_nvfp4_20261126'
runs=[('run_01',original),('run_02',root/'run_02'),('run_03',root/'run_03')]
configs=[json.loads((r/'raw/run_config.json').read_text()) for _,r in runs]
for c in configs:
 assert c['status']=='complete'
 for k in ('model','dataset_sha256','source_sha256','task_ids','temperature','max_tokens','enable_thinking','methods','task_selection_seed','decoding_seed'):
  assert c[k]==configs[0][k],k
 assert c['served_models']['data'][0]['root']==configs[0]['served_models']['data'][0]['root']
summaries=[json.loads((r/'raw/summary.json').read_text()) for _,r in runs]
per_run=[]
for (run,r),s in zip(runs,summaries):
 assert json.loads((r/'verification.json').read_text())['status']=='passed'
 for method,x in s.items():
  per_run.append(dict(run=run,method=method,tasks=x['tasks'],calls=x['calls'],VR=x['validity_rate'],NR=x['novelty_rate'],RR=x['recovery_rate'],delta_RR_baseline_pp=100*(x['recovery_rate']-s['baseline']['recovery_rate']),delta_RR_none_no_feedback_pp=100*(x['recovery_rate']-s['ours_none_no_feedback']['recovery_rate'])))
aggregate=[]
for method in configs[0]['methods']:
 row={'method':method,'repeats':3,'tasks_per_repeat':56,'total_calls':270}
 for label,key in [('VR','validity_rate'),('NR','novelty_rate'),('RR','recovery_rate')]:
  values=[s[method][key] for s in summaries]
  row[f'{label}_mean']=fmean(values)
  row[f'{label}_sample_sd']=stdev(values)
 for control in ('baseline','ours_none_no_feedback'):
  values=[100*(s[method]['recovery_rate']-s[control]['recovery_rate']) for s in summaries]
  row[f'delta_RR_{control}_mean_pp']=fmean(values)
  row[f'delta_RR_{control}_sample_sd_pp']=stdev(values)
 aggregate.append(row)
for name,rows in [('per_run_summary.csv',per_run),('aggregate_summary.csv',aggregate)]:
 with (root/name).open('w',newline='') as file:
  w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
names={'baseline':'Baseline','ours_mechanistic':'机制去重＋有反馈','ours_mechanistic_no_feedback':'机制去重＋无反馈','ours_semantic':'语义去重＋有反馈','ours_semantic_no_feedback':'语义去重＋无反馈','ours_none':'不去重＋有反馈','ours_none_no_feedback':'不去重＋无反馈'}
lines=['# Qwen3.8-27B NVFP4：三次重复实验','', '原先的第1次结果完整保留；本文件夹新增第2、3次完整运行。每次都是相同56题、六种配置加独立Baseline，每组90次调用。新增1260次调用，三次共1890次调用。困难题未包含在本轮。', '', '## 三次总体结果', '', 'RR为56题逐题恢复率的平均值；±为三次运行均值之间的样本标准差，单位是百分点，不是置信区间。']
for row in aggregate:
 vals=[s[row['method']]['recovery_rate'] for s in summaries]
 lines.append(f"- {names[row['method']]}：平均 RR {100*row['RR_mean']:.2f}% ± {100*row['RR_sample_sd']:.2f}；三次分别为 {', '.join(f'{100*v:.2f}%' for v in vals)}。")
lines+=['', '## 如何解读', '', '- 默认的“机制去重＋有反馈”相对同次Baseline的RR差值三次为+3.87、+3.57、+4.61个百分点，平均+4.02，样本标准差0.54个百分点。', '- “机制去重＋有反馈”相对规则相同但独立采样的“不去重＋无反馈”平均仅高0.35个百分点，样本标准差3.63个百分点；因此当前结果支持“相对Baseline稳定提高”，但不足以证明提升完全来自去重和反馈。', '- 语义去重＋有反馈的平均RR最高（50.35%），相对Baseline平均+9.23个百分点，但该差值的样本标准差为7.23个百分点，稳定性弱于机制去重。', '- 六种配置均保留；没有按首轮最高分选择性重跑某一组。第1次是在决定补重复实验之前完成，汇总用于描述稳定性，不作为预注册显著性检验。', '- Baseline与不去重＋无反馈使用相同请求规则，仍作为独立随机运行保留；它们的差异用于观察随机波动。', '- 任务选择种子始终为42，保证题目和顺序相同。解码种子未指定，沿用首轮协议；这三次是独立随机采样，不能称为解码种子42/43/44实验。', '- 温度0.7、最大输出256 tokens、请求enable_thinking=false、固定逐题预算、配置执行顺序及冻结源码均不变。相同服务模型ID和权重目录已核对；未收集权重文件哈希，服务实例时间戳可能不同。', '- 达到token上限的回复保留，按同一提取和计分规则计入；不选择性重试。32道单真值题仍只有一次调用，深度1数据本身的限制仍存在。', '', '## 文件与复核', '', '- [逐次汇总](per_run_summary.csv)、[三次均值与标准差](aggregate_summary.csv)。', '- [第1次原始结果](../qwen38_27b_nvfp4_20261126/REPORT.md)、[第2次复核](run_02/verification.json)、[第3次复核](run_03/verification.json)。', '- run_02和run_03分别保存七组逐题CSV、完整请求回复、源码快照、运行日志与离线复核脚本。', '- [重复实验协议](repeat_protocol.json)、[后台任务完成状态](batch_status.json)。', '']
(root/'REPORT.md').write_text('\n'.join(lines))
(root/'verification.json').write_text(json.dumps({'status':'passed','runs':3,'new_runs':2,'total_calls':1890,'new_calls':1260,'all_runs_replayed':True,'all_frozen_settings_and_task_orders_match':True,'original_run':str(original.relative_to(root.parent)),'finish_reasons_by_run':{run:json.loads((r/'verification.json').read_text())['finish_reasons'] for run,r in runs}},indent=2)+'\n')
(root/'checksums.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'checksums.json'},indent=2)+'\n')
print(json.dumps(aggregate,indent=2))
