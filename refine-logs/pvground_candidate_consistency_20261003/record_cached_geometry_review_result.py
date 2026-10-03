"""Preserve the actual reviewer response and refresh stale status wording."""
import datetime
import hashlib
import json
from pathlib import Path


local = Path(__file__).parent
trace = local/'.aris/traces/experiment-audit/2026-10-03_cached_geometry_run01'
report = json.loads((local/'CACHED_GEOMETRY_REVIEW.json').read_bytes())
assert report['verdict'] in ('PASS', 'WARN')
assert report['blocking_issues'] == []
meta = json.loads((trace/'run.meta.json').read_bytes())
assert meta['status'] == 'running'
now = datetime.datetime.now().astimezone().isoformat()
text = (local/'CACHED_GEOMETRY_REVIEW.md').read_text(encoding='utf-8')
final = (local/'CACHED_GEOMETRY_REVIEW_AGENT_FINAL.txt').read_text(encoding='utf-8')
response = dict(call_number=1, purpose='cached-geometry-integrity', timestamp=now,
                agent_id='/root/pv_cached_geometry_integrity', model='gpt-6-astra',
                reasoning_effort='max', reviewer_family='openai',
                review_independence='same-family', acceptance_status='provisional',
                status='ok', verdict=report['verdict'],
                raw_final_response=final, full_report_markdown=text,
                full_report_json=report,
                artifact_sha256={name: hashlib.sha256((local/name).read_bytes()).hexdigest()
                                 for name in ['CACHED_GEOMETRY_REVIEW.md', 'CACHED_GEOMETRY_REVIEW.json',
                                              'CACHED_GEOMETRY_REVIEW_AGENT_FINAL.txt']})
(trace/'001-cached-geometry-integrity.response.json').write_text(json.dumps(response, indent=2)+'\n', encoding='utf-8')
(trace/'001-cached-geometry-integrity.response.md').write_text(text, encoding='utf-8')
meta.update(status='completed', completed_at=now, review_verdict=report['verdict'],
            response_file='001-cached-geometry-integrity.response.json')
(trace/'run.meta.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
with (local/'.aris/meta/events.jsonl').open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(dict(event='review_completed', skill='experiment-audit',
                                purpose=response['purpose'], timestamp=now,
                                agent_id=response['agent_id'], status='ok',
                                verdict=report['verdict'], acceptance_status='provisional',
                                trace_path=str(trace)))+'\n')

# The launch snapshot is historical evidence, not a claim about a newer step.
observation = json.loads((local/'control_boundary_observation_000.json').read_bytes())
assert observation['controller_live'] and observation['child_live']
progress = [json.loads(line.split(' ', 1)[1]) for line in observation['stage_log_tail'].splitlines()
            if line.startswith('PVG_TRAIN_PROGRESS ')]
assert progress[-1]['step'] == 3648
plan = local/'EXPERIMENT_PLAN.md'
lines = plan.read_text(encoding='utf-8').splitlines(keepends=True)
assert '正式训练未启动' in lines[2]
lines[2] = ('日期：2026-10-03。状态：CPU/GPU真实预检已完成；正式配对已于08:34启动。'
            '截至11:15实际观察，G控制训练为3648/3723步；该观察不是服务器实时状态，'
            '尚无新正式精度。\n')
plan.write_text(''.join(lines), encoding='utf-8')
tracker = local/'EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines(keepends=True)
for i, line in enumerate(lines):
    if line.startswith('| C2 |'):
        assert '| NOT_STARTED |' in line
        fields = line.rstrip('\n').split('|')
        fields[4] = ' RUNNING_OBSERVED '
        fields[5] = ' 2026-10-03 11:15真实观察：g_control/train，3648/3723步；control_boundary_observation_000.json；无新正式精度 '
        lines[i] = '|'.join(fields)+'\n'
    elif line.startswith('| C3 |'):
        assert '| NOT_STARTED |' in line
        fields = line.rstrip('\n').split('|')
        fields[4] = ' QUEUED_IN_PAIR '
        fields[5] = ' 已启动串行配对中的第二训练阶段，11:15尚未开始；负对比目标及有效权重变化WARN保留 '
        lines[i] = '|'.join(fields)+'\n'
tracker.write_text(''.join(lines), encoding='utf-8')
(local/'cached_geometry_status_refresh.json').write_text(json.dumps(dict(
    time_cst=now, source='control_boundary_observation_000.json', observed_cst=observation['time_cst'],
    observed_control_updates=3648, new_accuracy_available=False,
    modified=['EXPERIMENT_PLAN.md', 'EXPERIMENT_TRACKER.md'],
    review_warning_retained=True, training_source_changed=False), indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(verdict=report['verdict'], same_family=True,
                     acceptance='provisional', status_wording_refreshed=True,
                     GPU_forward_executed=False, optimizer_updates=0)))
