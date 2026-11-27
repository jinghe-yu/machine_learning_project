"""Replay recorded responses against the frozen experiment source; no API calls."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import sys
import tarfile
import tempfile

root = Path(sys.argv[1]).resolve()
raw = root / 'raw'
config = json.loads((raw / 'run_config.json').read_text())
assert config['status'] == 'complete'
records = [json.loads(line) for line in (raw / 'requests.jsonl').read_text().splitlines()]
summary = json.loads((raw / 'summary.json').read_text())
assert len(records) == 630
by_task = collections.defaultdict(list)
for record in records:
    by_task[record['method'], record['task_id']].append(record)
assert set(r['response']['model'] for r in records) == {config['model']}
finish_reasons = collections.Counter(r['response']['choices'][0]['finish_reason'] for r in records)
rows_by_method = {}
snapshot = root / 'source.tar.gz'
if not snapshot.is_file():
    snapshot = Path(__file__).resolve().with_name('source.tar.gz')
with tempfile.TemporaryDirectory() as directory:
    with tarfile.open(snapshot) as archive:
        archive.extractall(directory, filter='data')
    source = next(Path(directory).rglob('run_qwen_smoke.py')).parent
    for name, digest in config['source_sha256'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    dataset = source / 'data/boolean_dataset.json'
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == config['dataset_sha256']
    sys.path.insert(0, str(source))
    from benchmark_utils import load_tasks, task_target_count, to_observations, score_hypotheses
    from boolean_genetics import sample_hypotheses
    metadata, tasks = load_tasks(dataset)
    tasks = {t['observation_set_id']: t for t in tasks}
    assert set(tasks) == set(config['task_ids']) and len(tasks) == 56
    for method in config['methods']:
        rows = list(csv.DictReader((raw / f'{method}.csv').open()))
        rows_by_method[method] = rows
        assert [r['observation_set_id'] for r in rows] == config['task_ids']
        dedup = 'none' if method == 'baseline' else method.split('_')[1]
        feedback = method != 'baseline' and not method.endswith('_no_feedback')
        for row in rows:
            task = tasks[row['observation_set_id']]
            trace = by_task[method, row['observation_set_id']]
            target = task_target_count(task, metadata['variables'], metadata.get('mechanistic_opts'))
            assert len(trace) == int(row['attempts']) == int(row['n_proposals']) == target
            iterator = iter(trace)
            def complete(prompt, temperature):
                record = next(iterator)
                assert record['request'] == {
                    'model': config['model'],
                    'messages': [
                        {'role': 'system', 'content': 'You solve Boolean-genetics hypothesis discovery tasks.'},
                        {'role': 'user', 'content': prompt}],
                    'temperature': temperature, 'max_tokens': config['max_tokens'],
                    'chat_template_kwargs': {'enable_thinking': False}}
                return record['response']['choices'][0]['message']['content'] or ''
            sampled = sample_hypotheses(
                to_observations(task), complete, target, max_attempts=target,
                start_temperature=.7, end_temperature=.7,
                allowed_operators=metadata['operators'], max_depth=metadata['max_depth'],
                allow_constants=False, deduplication=dedup, include_history=feedback,
                stop_on_target=False, mechanistic_options=metadata.get('mechanistic_opts'))
            assert list(sampled.proposals) == json.loads(row['proposal_candidates_json'])
            assert list(sampled.raw_responses) == json.loads(row['raw_responses_json'])
            assert dict(sampled.rejection_counts) == json.loads(row['rejection_counts'])
            score = score_hypotheses(task, sampled.proposals, metadata, proposal_count=sampled.attempts)
            for key, value in score.items():
                if isinstance(value, (int, float)):
                    assert math.isclose(float(row[key]), value, abs_tol=1e-12), (method, key)
        assert sum(int(r['attempts']) for r in rows) == summary[method]['calls'] == 90
        assert len(rows) == summary[method]['tasks'] == 56
        for metric in ('validity_rate', 'novelty_rate', 'recovery_rate'):
            assert math.isclose(fmean(float(r[metric]) for r in rows), summary[method][metric], abs_tol=1e-12)
        for token in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            assert sum(r['response']['usage'][token] for r in records if r['method'] == method) == summary[method]['usage'][token]
        assert math.isclose(sum(r['elapsed_seconds'] for r in records if r['method'] == method), summary[method]['api_elapsed_seconds'])
for task_id in config['task_ids']:
    assert [r['request'] for r in by_task['baseline', task_id]] == [r['request'] for r in by_task['ours_none_no_feedback', task_id]]
base = rows_by_method['baseline']
aggregate = []
strata = []
for method, rows in rows_by_method.items():
    differences = [float(r['recovery_rate']) - float(b['recovery_rate']) for r, b in zip(rows, base)]
    aggregate.append(dict(method=method, tasks=len(rows), proposals=summary[method]['calls'],
        mean_VR=summary[method]['validity_rate'], mean_NR=summary[method]['novelty_rate'],
        mean_RR=summary[method]['recovery_rate'],
        delta_RR_pp=100 * fmean(differences), wins=sum(d > 0 for d in differences),
        ties=sum(d == 0 for d in differences), losses=sum(d < 0 for d in differences),
        api_elapsed_seconds=summary[method]['api_elapsed_seconds']))
    for label, selected in [('single_truth', [r for r in rows if int(r['ground_truth_size']) == 1]),
                            ('multiple_truths', [r for r in rows if int(r['ground_truth_size']) > 1])]:
        strata.append(dict(method=method, stratum=label, tasks=len(selected),
            mean_RR=fmean(float(r['recovery_rate']) for r in selected)))
for filename, rows in [('summary.csv', aggregate), ('strata.csv', strata)]:
    with (root / filename).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
verification = dict(status='passed', tasks_per_group=56, groups=7, task_rows=392,
    recorded_calls=len(records), calls_per_group=90, finish_reasons=dict(finish_reasons),
    frozen_source_hashes_match=True, dataset_hash_matches=True,
    all_prompts_replayed=True, all_candidates_and_rejections_replayed=True,
    all_numeric_task_metrics_recomputed=True, summary_and_token_usage_reconciled=True,
    baseline_and_none_no_feedback_payloads_identical=True)
(root / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
checksum_file = root / 'checksums.json'
shared_paths = [name for name in json.loads(checksum_file.read_text()) if name.startswith('../')] if checksum_file.is_file() else []
checksums = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in shared_paths}
checksums.update({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in root.rglob('*') if p.is_file() and p != checksum_file})
checksum_file.write_text(json.dumps(checksums, indent=2, sort_keys=True) + '\n')
print(json.dumps(verification, indent=2))
print(json.dumps(aggregate, indent=2))
