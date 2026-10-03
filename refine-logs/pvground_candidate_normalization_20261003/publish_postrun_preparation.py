"""Publish the minimal prepared post-run readers without touching training source."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko


workspace = Path(r'C:\Users\gb')
local = Path(__file__).parent
receipt_path = local / 'postrun_publication.json'
assert not receipt_path.exists()
previous = json.loads((local / 'launch_publication.json').read_bytes())
preparation = json.loads((local / 'postrun_preparation.json').read_bytes())
assert preparation['status'] == 'PREPARED_NOT_EXECUTED'
assert preparation['scripts_compile'] and preparation['intended_spec_differences_only']
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
prefix = 'refine-logs/pvground_candidate_normalization_20261003/'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert len(old) == previous['handoff_bytes']
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
now = datetime.datetime.now().astimezone().isoformat()
addition = ('\n\n## 20.376.30 归一化控制的终态采集与三组分析准备（' + now + '）\n\n'
    '新增只读终态采集器collect_complete.py及三组分析器analyze_complete.py，实际本地编译通过，并核对相对已完成扩展对应组的配置差异仅为预定归一化及描述字段。'
    '两项处于PREPARED_NOT_EXECUTED；当前没有本轮正式精度结果，也没有新GPU前向、优化器更新、远端轮询或训练源码变更。\n\n'
    '采集器要求原控制器完成、退出0且已不在进程表，再核对训练3723更新、29778输入各一次与9508完整评估；'
    '带回日志、配置、源码快照、逐行评估、终点和原G的SHA凭证，未下载或删除模型权重。'
    '分析器复用已完成对照的原生行指标核验函数，分别比较归一化组与原G续训控制、原分母扩展对应组；'
    '检查同一fit顺序及输入/GT，记录实际起点连续值、Query、候选覆盖与REC/Mask阈值差异，不能预设起点逐位或阈值一致。\n\n'
    '结果保留bbs/bbf、Mask三项、修复/破坏、最终框GT上界及相对原G的真实差值；'
    '归一化改动解释为整个扩展对比项从N改为N+A的权重变化，而非纯标签变化。'
    '分析不执行权重删除；完整结果核验后按Acc@0.50优先及Acc@0.25同分决策，仅保留指标最佳、必要父权重和一个活动恢复点。'
    '首个远端检查仍定为19:32:02，之后240秒一次，预计19:35:02完成仅为前次耗时估计；现有等待会话45449在17:01实际读取仍存活，没有重启模型或另开轮询器。\n')
new = old + addition.encode('utf-8')
names = ['collect_complete.py', 'analyze_complete.py', 'prepare_postrun.py',
         'postrun_preparation.json', 'publish_postrun_preparation.py']
payloads = {prefix + name: (local / name).read_bytes() for name in names}
for name, expected in preparation['sources'].items():
    assert hashlib.sha256(payloads[prefix + name]).hexdigest() == expected['sha256']
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == old
for relative in payloads:
    assert not any((repo / relative).exists() for repo in repos[:2])
for relative, raw in payloads.items():
    for repo in repos[:2]:
        (repo / relative).write_bytes(raw)
    with sftp.open(remote + '/' + relative, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + relative, 'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote + '/' + doc, 'wb') as stream:
    stream.write(new)
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == new
assert all(path.read_bytes() == new for path in copies)
sftp.close()
client.close()
heads = []
for index, repo in enumerate(repos):
    stage = [doc] + (list(payloads) if index < 2 else [])
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in status.splitlines()} == set(stage)
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    if index < 2:
        subprocess.check_call(['git', '-C', str(repo), 'add', '--renormalize', '--', *payloads])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Prepare normalization terminal intake and three-arm result comparison'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
for relative in (doc, *payloads):
    assert subprocess.check_output(['git', '-C', str(repos[0]), 'rev-parse', 'HEAD:' + relative]) == subprocess.check_output(['git', '-C', str(repos[1]), 'rev-parse', 'HEAD:' + relative])
digest = hashlib.sha256(new).hexdigest()
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.30', heads=heads,
    github_main=heads[0], handoff_sha256=digest, handoff_bytes=len(new),
    prior_prefix_unchanged=True, four_local_and_remote_equal=True, exact_committed_payloads=True,
    payload_count=len(payloads), postrun_status='PREPARED_NOT_EXECUTED', new_accuracy_available=False,
    active_training_source_modified=False, active_controller_restarted=False,
    new_GPU_forwards=0, new_optimizer_updates=0, checkpoints_downloaded=False, weights_deleted=False)
receipt_path.write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
