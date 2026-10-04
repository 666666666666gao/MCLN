"""Publish actual completed pair only after native closure and fresh audit."""
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
workspace = Path('C:/Users/gb')
repos = [workspace/'.codex_mcln_g0_20260905', workspace/'.codex_pvground_cs_20261002',
    workspace/'.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo/doc for repo in repos] + [workspace/'Desktop/document'/Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
guard = workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

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
assert all(not (repo/relative).exists() for repo in repos[:2] for relative in payloads)
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
    password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp(); remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote+'/'+doc,'rb') as stream:
    assert stream.read() == old
directories = sorted({str(Path(relative).parent).replace('\\','/') for relative in payloads},
    key=lambda item:(item.count('/'),item))
created = {'refine-logs','refine-logs/pvground_range_head_only_20261004'}
for directory in directories:
    parts = directory.split('/')
    for count in range(1,len(parts)+1):
        current = '/'.join(parts[:count])
        if current not in created:
            sftp.mkdir(remote+'/'+current); created.add(current)
for relative, raw in payloads.items():
    for repo in repos[:2]:
        target = repo/relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
    with sftp.open(remote+'/'+relative,'wx') as stream:
        stream.write(raw)
    with sftp.open(remote+'/'+relative,'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote+'/'+doc,'wb') as stream:
    stream.write(new)
assert all(path.read_bytes() == new for path in copies)
with sftp.open(remote+'/'+doc,'rb') as stream:
    assert stream.read() == new
sftp.close(); client.close()
heads = []
for index, repo in enumerate(repos):
    if index < 2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write(f'\n- {stamp} head-only pair complete9508 local{local_hits} whole{whole_hits}; fresh audit {verdict}, owned nonbest retired.\n')
    stage = [doc] + (['MANIFEST.md',*payloads] if index < 2 else [])
    changed = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check','--',*stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git','-C',str(repo),'show',':'+relative]) == raw
    subprocess.check_call(['git','-C',str(repo),'commit','-m','Record complete stable G range comparison and fresh terminal audit'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0] == heads[0]
for relative in (doc,*payloads):
    assert subprocess.check_output(['git','-C',str(repos[0]),'rev-parse','HEAD:'+relative]) == subprocess.check_output(['git','-C',str(repos[1]),'rev-parse','HEAD:'+relative])
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.42',
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
