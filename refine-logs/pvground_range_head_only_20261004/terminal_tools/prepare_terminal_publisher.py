"""Prepare a gated publisher; do not consume results or publish before pair closure."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
basis = (local/'publish_local_phase.py').read_text(encoding='utf-8')
output = local/'publish_terminal.py'
assert not output.exists()
guard_start = "workspace = Path('C:/Users/gb')"
section_start = "section = f'''"
sync_start = "assert all(not (repo/relative).exists()"
record_start = 'record = dict(time_cst='
assert all(basis.count(marker) == 1 for marker in (guard_start,section_start,sync_start,record_start))
guards = basis[basis.index(guard_start):basis.index(section_start)]
sync = basis[basis.index(sync_start):basis.index(record_start)]
old_manifest = 'head-only local complete9508 bbs{bbs["rec_hits25"]}/{bbs["rec_hits50"]}; whole still active, controller CPU proof only, owned nonbest retired.'
new_manifest = 'head-only pair complete9508 local{local_hits} whole{whole_hits}; fresh audit {verdict}, owned nonbest retired.'
assert sync.count(old_manifest) == 1
sync = sync.replace(old_manifest,new_manifest).replace(
    'Record completed local range head control while whole arm runs',
    'Record complete stable G range comparison and fresh terminal audit')
header = r'''"""Publish actual completed pair only after native closure and fresh audit."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
assert not (local/'terminal_publication.json').exists()
previous = json.loads((local/'local_phase_publication.json').read_bytes())
intake = json.loads((local/'complete/INTAKE.json').read_bytes())
summary = json.loads((local/'analysis/SUMMARY.json').read_bytes())
audit = json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
trace = local/'.aris/traces/experiment-audit/2026-10-04_head_only_terminal_run01'
response = json.loads((trace/'002-terminal-integrity.response.json').read_bytes())
wait = json.loads((local/'wait.json').read_bytes())
status = intake['status']
assert wait['status'] == 'complete' and wait['controller_exit'] == '0'
assert status['status'] == 'complete' and intake['controller_exit'] == 0 and not intake['controller_alive']
assert len(status['completed']) == 4 and summary['head_only'] and summary['original_g_state_unchanged']
assert not audit['blocking_issues'] and audit['verdict'].upper() in ('PASS','WARN')
assert audit['review_independence'] == response['review_independence'] == 'same-family'
assert audit['acceptance_status'] == response['acceptance_status'] == 'provisional'
assert response['actual_task'] == '/root/pvg_head_only_terminal_integrity'
assert response['verdict'] == audit['verdict'] and not response['blocking_issues']
for name,item in response['reviewer_reports'].items():
    raw = (local/'analysis'/name).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
metrics = summary['phases']['formal']['metrics']
local_hits = [metrics['local_range']['bbs'][key] for key in ('rec_hits25','rec_hits50')]
whole_hits = [metrics['whole_range']['bbs'][key] for key in ('rec_hits25','rec_hits50')]
verdict = audit['verdict'].upper()
retentions = [json.loads((local/'complete'/arm/'weight_retention.json').read_bytes())
    for arm in ('local_range','whole_range')]
deleted_bytes = sum(item['bytes'] for retention in retentions for item in retention['deleted'])
report = (local/'analysis/REPORT.md').read_text(encoding='utf-8')
decision = (local/'NEXT_EXPERIMENT_DECISION.md').read_text(encoding='utf-8')
'''
section = r"""section = f'''

## 20.376.42 固定原G范围头来源对照完整终态及新鲜审查（{stamp}）

唯一controller450028实际于{status['finished_cst']}完成四阶段，exit0；观察器74041自然结束，完整收取后不重启训练或重复预检。§20.376.40的结构和预算保持，§20.376.41只是当时第一组终态，本节更新完整配对。原EXPERIMENT_PLAN中的planned状态保留为启动前计划，不当作当前运行状态。

{report.replace('# PV-Ground frozen-G local / whole complete range-source comparison','### 实际结果与分析限度',1)}

新鲜审查实际{verdict}、0阻断，gpt-6-astra／max／forknone，same-family／provisional，未独立验证后端SKU。审查文件和实际原始请求／响应保留，不将执行者检查冒称独立审查。原始Mask、完整候选框、优化器是否重放以及CPU重算范围，以analysis/EXPERIMENT_AUDIT.md和JSON中的实际execution_scope／limitations为准；不扩大未执行范围。

已收取{len(intake['files'])}份本轮原始文字／逐行／源码文件，完整INTAKE记录SHA。收取和本地分析模型前向0、优化更新0、权重下载0；控制器在恢复评估、所选框CPU阈值校验和父SHA核对后退役自有非最佳终点，共{deleted_bytes}字节，未创建失败本地权重归档。当前本轮指标最佳为{status['retained_best']['name']}，保留自有权重{len(intake['weights_retained'])}份；保护原G、官方PV和V99所需链。收取时系统盘{intake['system_disk_free_bytes']}字节、数据盘{intake['directory_free_bytes']}字节可用，不写成实时余量。

{decision}

证据：refine-logs/pvground_range_head_only_20261004/complete、analysis、terminal_audit_trace及terminal_tools。ScanRefer同检查点目标5615／4754的本轮检查为{summary['scanrefer_development_target_pass']}；完整三基准目标仍未完成，未新增Nr3D／Sr3D训练结果，不自动晋级完整目标。
'''
new = old + section.encode('utf-8')
prefix = 'refine-logs/pvground_range_head_only_20261004/'
payloads = {}
for directory in ('complete','analysis'):
    for path in sorted((local/directory).rglob('*')):
        if path.is_file():
            payloads[prefix+path.relative_to(local).as_posix()] = path.read_bytes()
for path in sorted(trace.iterdir()):
    if path.is_file():
        payloads[prefix+'terminal_audit_trace/'+path.name] = path.read_bytes()
for name in ('collect_terminal.py','analyze_terminal.py','prepare_terminal_audit_request.py',
    'TERMINAL_AUDIT_REQUEST.txt','record_terminal_audit_request.py','record_terminal_audit_result.py',
    'prepare_terminal_publisher.py','publish_terminal.py','NEXT_EXPERIMENT_DECISION.md',
    'terminal_collector_metadata_update.json','terminal_analyzer_text_update.json',
    'terminal_audit_preparation.json'):
    payloads[prefix+'terminal_tools/'+name] = (local/name).read_bytes()
latest = wait['observations'][-1]['file']
payloads[prefix+'terminal_tools/'+latest] = (local/latest).read_bytes()
"""
footer = r'''record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.42',
    heads=heads,github_main=heads[0],handoff_sha256=digest,handoff_bytes=len(new),
    four_local_and_remote_equal=True,exact_committed_payloads=True,payload_count=len(payloads),
    local_bbs_hits=local_hits,whole_bbs_hits=whole_hits,whole_formal_complete=True,
    retained_metric_best=status['retained_best']['name'],owned_deleted_bytes=deleted_bytes,
    weights_downloaded=0,fresh_pair_audit_complete=True,audit_verdict=verdict,
    review_independence='same-family',acceptance_status='provisional',goal_achieved=False)
(local/'terminal_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]} §20.376.42 published {heads[0]} fourlocal+remote SHA{digest}; actualheadonly pair local{local_hits} whole{whole_hits}, freshaudit{verdict}/0blocking samefamily/provisional, {deleted_bytes}B verifiedownedretired/noarchives. GoalACTIVE_UNMET.\n')
print(json.dumps(record),flush=True)
'''
text = header + guards + section + sync + footer
ast.parse(text)
output.write_text(text,encoding='utf-8')
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status='PREPARED_NOT_EXECUTED',
    publisher_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    actual_full_pair_required=True,actual_fresh_audit_required=True,actual_next_decision_required=True,
    remote_reads=0,model_forwards=0,weights_deleted=0,experimental_source_modified=False)
(local/'terminal_publisher_preparation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
