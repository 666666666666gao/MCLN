"""Publish actual completed boundary results and the actual fresh audit."""
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
assert all(not (repo/relative).exists() for repo in repos[:2] for relative in payloads)
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
    password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp(); remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote+'/'+doc,'rb') as stream:
    assert stream.read() == old
directories = sorted({str(Path(relative).parent).replace('\\','/') for relative in payloads},
    key=lambda item:(item.count('/'),item))
created = {'refine-logs','refine-logs/pvground_boundary_distribution_20261004'}
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
            stream.write(f'\n- {stamp} frozen-G boundary pair complete9508 residual{residual_hits} distribution{distribution_hits}; fresh audit {verdict}, verified owned retention.\n')
    stage = [doc] + (['MANIFEST.md',*payloads] if index < 2 else [])
    changed = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check','--',*stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git','-C',str(repo),'show',':'+relative]) == raw
    subprocess.check_call(['git','-C',str(repo),'commit','-m','Record complete frozen G boundary comparison and fresh terminal audit'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0] == heads[0]
for relative in (doc,*payloads):
    assert subprocess.check_output(['git','-C',str(repos[0]),'rev-parse','HEAD:'+relative]) == subprocess.check_output(['git','-C',str(repos[1]),'rev-parse','HEAD:'+relative])
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.45',
    heads=heads,github_main=heads[0],handoff_sha256=digest,handoff_bytes=len(new),
    four_local_and_remote_equal=True,exact_committed_payloads=True,payload_count=len(payloads),
    residual_bbs_hits=residual_hits,distribution_bbs_hits=distribution_hits,full_pair_complete=True,
    retained_metric_best=status['retained_best']['name'],owned_deleted_bytes=deleted_bytes,
    weights_downloaded=0,fresh_pair_audit_complete=True,audit_verdict=verdict,
    review_independence='same-family',acceptance_status='provisional',goal_achieved=False)
(local/'terminal_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]} section20.376.45 published {heads[0]} fourlocal+remote SHA{digest}; actual boundary pair residual{residual_hits} distribution{distribution_hits}, fresh audit{verdict}/0blocking samefamily/provisional, {deleted_bytes}B verified owned retired/no archives. GoalACTIVE_UNMET.\n')
print(json.dumps(record),flush=True)
