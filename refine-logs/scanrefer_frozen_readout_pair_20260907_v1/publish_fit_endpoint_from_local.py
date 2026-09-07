import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_frozen_readout_pair_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_frozen_readout_pair_20260907_v1'
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document') / master.name
digest = lambda raw: hashlib.sha256(raw).hexdigest()
old = master.read_bytes()
assert digest(old) == '49c4aa641735c8a35d1064feacb111d007b7a3d748ae1d358b880ccc7cf6e537'
assert desktop.read_bytes() == old
fit = json.loads((local / 'fit_complete.json').read_bytes())
check_raw = (local / 'fit_endpoint_check.json').read_bytes()
check = json.loads(check_raw)
assert check['steps_per_arm'] == fit['steps_per_arm'] == 2482
assert check['fit_rows'] == 29778 and check['last_batch_rows'] == 6
assert check['fit_rows_exactly_once_and_holdout_disjoint'] and check['endpoint_hashes_independently_verified']
assert check['checkpoints'] == fit['checkpoints']
assert check['fit_complete_sha256'] == digest((local / 'fit_complete.json').read_bytes())
assert check['fit_batches_sha256'] == digest((local / 'fit_point_batches.json').read_bytes())
observation = json.loads((local / 'observation_112318.json').read_bytes())
assert observation['progress']['SCANREFER FROZEN READOUT EVAL']['stage'] == 'terminal'
assert observation['controller_exit'] is None and observation['queue_controller_exit'] is None
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
_, output, error = client.exec_command('ps -p 52529,52535 -o pid,ppid,stat,etime,args', timeout=30)
processes = output.read().decode()
assert output.channel.recv_exit_status() == 0, error.read().decode()
assert '52529' in processes and '52535' in processes
for name in ['fit_complete.json', 'fit_point_batches.json']:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=(local / name).stat().st_size)
        assert stream.read() == (local / name).read_bytes()
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when = now.strftime('%Y-%m-%d %H:%M CST')
addition = '\n\n### 20.116 冻结读出配对完成固定训练，终点已落盘（' + when + '）\n\n'
addition += '原52529训练及52535接续队列持续运行，没有重启或改变设置。11:23:17 CST实际观察，两臂均完成2482/2482次更新，并已进入terminal评估；最新已打印记录为12/6887行，仅证明终点评估已启动，不能据此报告完整质量。实际训练更新耗时10171.826秒，含终点保存的fit记录为10178.659秒。\n\n'
addition += '11:23:45的独立CPU收取检查逐批验证2482个记录、29778条fit表达各出现一次、末批6行、与6887条holdout无交集，并在服务器重新计算两份终点SHA256、核对大小与fit记录。没有下载权重到本地，没有增加GPU推理或训练。该检查尚不包括终点张量、冻结读出及optimizer状态的完整独立审计；这些按原队列在terminal结束后执行。\n\n'
addition += '| 固定终点 | 文件大小 bytes | SHA256 |\n|---|---:|---|\n'
for arm in ['native_only', 'frozen_gt']:
    item = fit['checkpoints'][arm]
    addition += '| ' + arm + ' | ' + str(item['bytes']) + ' | `' + item['sha256'] + '` |\n'
addition += '\n终点目录仍为`' + remote + '`。fit_complete SHA`' + check['fit_complete_sha256'] + '`；fit_point_batches SHA`' + check['fit_batches_sha256'] + '`；独立收取回执SHA`' + digest(check_raw) + '`。本次仅新增两份固定终点，共1237204430 bytes；11:23观察磁盘剩余8104755200 bytes，约7.548 GiB，没有再次删除权重。\n\n'
addition += '当前没有terminal完整指标、正式9508结果或晋级结论，历史最好保持不变。根据同轮baseline约1836秒评估耗时，预计terminal约11:51结束；这是时间估计，不是完成证据。继续等待固定6887评估与原定独立审计；只有预先固定frozen_gt在系统REC两阈值相对起点和native_only均不退化，才接续唯一9508正式评估。Scan通过既定保护线后尽快进行Nr/Sr REC训练，不等待59/51，不设置Nr/Sr Mask门槛。总体目标继续active。\n'
new = old + addition.encode()
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. Section20.116: fixed pair completed2482 updates/arm;both endpoint hashes/batch traversal verified;terminal6887 running,no new quality or formal result.'
for i, line in enumerate(lines):
    if line.startswith('| Frozen protected readout compatibility |'):
        lines[i] = '| Frozen protected readout compatibility | Fit2482/2482 each complete;two fixed endpoints hashchecked;terminal6887 running | Frozen_gt fixed candidate;independent terminal audit and module REC gate pending;no new formal result or Nr/Sr training |'
tracker_raw = ('\n'.join(lines) + '\n').encode()
for name, raw in [('docs/' + master.name, new), ('refine-logs/EXPERIMENT_TRACKER.md', tracker_raw)]:
    with sftp.open(runtime + name, 'wb') as stream:
        stream.set_pipelined(True)
        stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw
master.write_bytes(new)
desktop.write_bytes(new)
tracker.write_bytes(tracker_raw)
proof = {'time_cst': now.isoformat(), 'section': '20.116', 'bytes': len(new), 'sha256': digest(new),
         'three_master_copies_equal': master.read_bytes() == desktop.read_bytes() == new,
         'verified_live_processes': processes, 'fit_endpoint_check_sha256': digest(check_raw),
         'fixed_steps_per_arm': 2482, 'terminal_quality_result_available': False, 'goal_complete': False}
proof_raw = (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()
(local / 'handoff_sync_20_116.json').write_bytes(proof_raw)
(local / 'publish_fit_endpoint_from_local.py').write_bytes(Path(__file__).read_bytes())
for name in ['fit_endpoint_check.json', 'collect_fit_from_local.py', 'handoff_sync_20_116.json', 'publish_fit_endpoint_from_local.py']:
    raw = (local / name).read_bytes()
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw
sftp.close()
client.close()
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=repo).strip()
subprocess.run(['git', 'add', '--', 'docs/' + master.name, 'refine-logs/EXPERIMENT_TRACKER.md', str(local.relative_to(repo))], cwd=repo, check=True)
staged = subprocess.check_output(['git', 'diff', '--cached'], cwd=repo)
assert os.environ['MCLN_SSH_PASSWORD'].encode() not in staged
names = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=repo, text=True).splitlines()
assert not any(Path(name).suffix in ['.pt', '.pth'] for name in names)
for name in names:
    if name.startswith('refine-logs/scanrefer_frozen_readout_pair_'):
        assert subprocess.check_output(['git', 'show', ':' + name], cwd=repo) == (repo / name).read_bytes(), name
subprocess.run(['git', 'diff', '--cached', '--check'], cwd=repo, check=True)
subprocess.run(['git', 'commit', '-m', 'Record fixed ScanRefer compatibility endpoints and terminal evaluation start'], cwd=repo, check=True)
subprocess.run(['git', 'push', 'origin', 'HEAD:main'], cwd=repo, check=True)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=repo, text=True).split()[0] == head
proof['published_main'] = head
print(json.dumps(proof), flush=True)
