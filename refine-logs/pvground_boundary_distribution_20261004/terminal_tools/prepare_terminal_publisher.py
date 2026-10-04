"""Prepare the result publisher without reading unfinished experiment results."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
basis = (local.parent / 'pvground_range_head_only_20261004/publish_terminal.py').read_text(encoding='utf-8')
output = local / 'publish_terminal.py'
assert not output.exists()
guards_start = "workspace = Path('C:/Users/gb')"
section_start = "section = f'''"
sync_start = "assert all(not (repo/relative).exists()"
record_start = 'record = dict(time_cst='
assert all(basis.count(marker) == 1 for marker in (guards_start, section_start, sync_start, record_start))
guards = basis[basis.index(guards_start):basis.index(section_start)]
sync = basis[basis.index(sync_start):basis.index(record_start)]
sync = sync.replace('pvground_range_head_only_20261004', 'pvground_boundary_distribution_20261004')
old_manifest = 'head-only pair complete9508 local{local_hits} whole{whole_hits}; fresh audit {verdict}, owned nonbest retired.'
assert sync.count(old_manifest) == 1
sync = sync.replace(old_manifest,
    'frozen-G boundary pair complete9508 residual{residual_hits} distribution{distribution_hits}; fresh audit {verdict}, verified owned retention.')
sync = sync.replace('Record complete stable G range comparison and fresh terminal audit',
    'Record complete frozen G boundary comparison and fresh terminal audit')
sync = sync.replace("'push','origin','HEAD:main'", "'-c','http.version=HTTP/1.1','push','origin','HEAD:main'")
header = '''"""Publish actual completed boundary results and the actual fresh audit."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
assert not (local/'terminal_publication.json').exists()
previous = json.loads((local/'residual_phase_publication.json').read_bytes())
intake = json.loads((local/'complete/INTAKE.json').read_bytes())
summary = json.loads((local/'analysis/SUMMARY.json').read_bytes())
audit = json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
trace = local/'.aris/traces/experiment-audit/2026-10-04_boundary_terminal_run01'
response = json.loads((trace/'002-terminal-integrity.response.json').read_bytes())
wait = json.loads((local/'wait.json').read_bytes())
status = intake['status']
assert wait['status'] == 'complete' and wait['controller_exit'] == '0'
assert status['status'] == 'complete' and intake['controller_exit'] == 0 and not intake['controller_alive']
assert len(status['completed']) == 4 and summary['head_only'] and summary['original_g_state_unchanged']
assert summary['trainable_parameters'] == dict(residual=400614, distribution=456102)
assert not audit['blocking_issues'] and audit['verdict'].upper() in ('PASS','WARN')
assert audit['review_independence'] == response['review_independence'] == 'same-family'
assert audit['acceptance_status'] == response['acceptance_status'] == 'provisional'
assert response['actual_task'] == '/root/pvg_boundary_terminal_integrity'
assert response['verdict'] == audit['verdict'] and not response['blocking_issues']
for name, item in response['reviewer_reports'].items():
    raw = (local/'analysis'/name).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
metrics = summary['phases']['formal']['metrics']
residual_hits = [metrics['residual']['bbs'][key] for key in ('rec_hits25','rec_hits50')]
distribution_hits = [metrics['distribution']['bbs'][key] for key in ('rec_hits25','rec_hits50')]
verdict = audit['verdict'].upper()
retentions = [json.loads((local/'complete'/arm/'weight_retention.json').read_bytes())
    for arm in ('residual','distribution')]
deleted_bytes = sum(item['bytes'] for retention in retentions for item in retention['deleted'])
report = (local/'analysis/REPORT.md').read_text(encoding='utf-8')
decision = (local/'NEXT_EXPERIMENT_DECISION.md').read_text(encoding='utf-8')
'''
section = """section = f'''

## 20.376.45 固定原G的普通残差／六面分布完整终态与独立上下文核对（{stamp}）

本轮controller471894实际于{status['finished_cst']}完成全部四阶段，exit0；现有观察器自然闭合后才收集结果。保留§20.376.43的原始启动记录；计划与预检不改写为性能证据。

{report.replace('# PV-Ground frozen-G residual / distribution boundary comparison','### 实际结果与范围',1)}

新上下文核对实际为{verdict}、0 blocking；调用gpt-6-astra/max/forknone，same-family/provisional，未独立验证服务端SKU。以原始审阅回复和analysis/EXPERIMENT_AUDIT.md、JSON中实际执行范围为准。CPU重算不等于原始Mask、全部候选框或优化器回放。两组共同使用1302维完整／局部支撑；400614参数残差与456102参数分布头同时改变表示、监督及输出参数量，不宣称纯DFL因果增益，也不等于已经实现方向条件化六面token解码。

实际收集{len(intake['files'])}份SHA绑定的文本、源码和逐行证据；收集与分析执行0次模型前向、0次优化器更新、0份权重下载。恢复、9508条正式开发验证、CPU阈值计数及SHA核对之后，controller实际清理自有非最佳终点共{deleted_bytes}字节。保留指标最佳{status['retained_best']['name']}；实验目录实际剩余权重{len(intake['weights_retained'])}份，不创建失败权重的本地归档。原G、官方PV及必要V99链继续保护。收集时系统盘{intake['system_disk_free_bytes']}字节、数据盘{intake['directory_free_bytes']}字节可用，这只是该次快照。

{decision}

证据：refine-logs/pvground_boundary_distribution_20261004/complete、analysis、terminal_audit_trace、terminal_tools。同一检查点ScanRefer开发线5615／4754的本轮检查结果为{summary['scanrefer_development_target_pass']}。完整跨基准目标仍未完成；尚无本方法Nr3D／Sr3D结果，也没有教师或最终质量回写结果。
'''
new = old + section.encode('utf-8')
prefix = 'refine-logs/pvground_boundary_distribution_20261004/'
payloads = {}
for directory in ('complete','analysis'):
    for path in sorted((local/directory).rglob('*')):
        if path.is_file():
            payloads[prefix+path.relative_to(local).as_posix()] = path.read_bytes()
for path in sorted(trace.iterdir()):
    if path.is_file():
        payloads[prefix+'terminal_audit_trace/'+path.name] = path.read_bytes()
for name in ('TERMINAL_AUDIT_REQUEST.txt','ACTUAL_TERMINAL_REVIEWER_RECEIPT.json',
    'ACTUAL_TERMINAL_REVIEWER_MESSAGE.txt','ACTUAL_TERMINAL_REVIEWER_RESPONSE.txt',
    'record_terminal_audit_request.py',
    'record_terminal_audit_result.py','prepare_terminal_publisher.py','publish_terminal.py',
    'terminal_publisher_preparation.json','terminal_audit_recorders_preparation.json',
    'terminal_publisher_trace_update.json','NEXT_EXPERIMENT_DECISION.md','volume_group_cpu_validation.json'):
    payloads[prefix+'terminal_tools/'+name] = (local/name).read_bytes()
latest = wait['observations'][-1]['file']
payloads[prefix+'terminal_tools/'+latest] = (local/latest).read_bytes()
"""
footer = '''record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.45',
    heads=heads,github_main=heads[0],handoff_sha256=digest,handoff_bytes=len(new),
    four_local_and_remote_equal=True,exact_committed_payloads=True,payload_count=len(payloads),
    residual_bbs_hits=residual_hits,distribution_bbs_hits=distribution_hits,full_pair_complete=True,
    retained_metric_best=status['retained_best']['name'],owned_deleted_bytes=deleted_bytes,
    weights_downloaded=0,fresh_pair_audit_complete=True,audit_verdict=verdict,
    review_independence='same-family',acceptance_status='provisional',goal_achieved=False)
(local/'terminal_publication.json').write_text(json.dumps(record,indent=2)+'\\n',encoding='utf-8')
with (workspace/'memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write(f'\\nPVGround {record["time_cst"]} section20.376.45 published {heads[0]} fourlocal+remote SHA{digest}; actual boundary pair residual{residual_hits} distribution{distribution_hits}, fresh audit{verdict}/0blocking samefamily/provisional, {deleted_bytes}B verified owned retired/no archives. GoalACTIVE_UNMET.\\n')
print(json.dumps(record),flush=True)
'''
text = header + guards + section + sync + footer
ast.parse(text)
output.write_text(text, encoding='utf-8')
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), status='PREPARED_NOT_EXECUTED',
    publisher_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    actual_pair_closure_required=True, actual_fresh_audit_required=True,
    actual_next_decision_required=True, remote_queries=0, model_forwards=0,
    weights_deleted=0, active_training_source_changed=False)
(local/'terminal_publisher_preparation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
