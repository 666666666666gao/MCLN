import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_range_pair_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_range_pair_20260907_v1'
archive = '/root/autodl-tmp/mcln_scanrefer_range_fit_check_20260907_v1'
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document') / master.name
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == 'b7c1a80beee411e703a9214202be5593776bd0045e9d79db4d98daadf3a07a70'
assert desktop.read_bytes() == old
check_raw = (local / 'fit_endpoint_check.json').read_bytes()
check = json.loads(check_raw)
assert check['steps_per_arm'] == 2482 and check['fit_rows'] == 29778
assert check['fit_rows_exactly_once_and_holdout_disjoint'] and check['endpoint_hashes_independently_verified']
assert hashlib.sha256((local / 'fit_complete.json').read_bytes()).hexdigest() == check['fit_complete_sha256']
assert hashlib.sha256((local / 'fit_point_batches.json').read_bytes()).hexdigest() == check['fit_batches_sha256']
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
with sftp.open(remote + '/fit_complete.json', 'rb') as stream:
    assert stream.read() == (local / 'fit_complete.json').read_bytes()
assert 'receipt.json' not in sftp.listdir(remote)
_, output, error = client.exec_command('ps -p 47112,47242,48128 -o pid,stat,etime,args', timeout=30)
processes = output.read().decode()
assert output.channel.recv_exit_status() == 0, error.read().decode()
for pid, name in [(47112, 'mcln_scanrefer_range_pair_v1'), (47242, 'mcln_scanrefer_range_posttraining_v1'), (48128, 'mcln_native_range_preflight_queue_v1')]:
    assert any(line.split()[0] == str(pid) and name in line for line in processes.splitlines()[1:])
with sftp.open(remote + '/run.log', 'rb') as stream:
    stream.seek(max(0, stream.stat().st_size - 64000))
    tail = stream.read()
prefix = 'SCANREFER RANGE EVAL '
latest = json.loads([line[len(prefix):] for line in tail.decode().splitlines() if line.startswith(prefix + '{')][-1])
assert latest['stage'] == 'terminal' and latest['total'] == 6887
_, output, error = client.exec_command('df -B1 /root/autodl-tmp', timeout=30)
disk_output = output.read().decode()
assert output.channel.recv_exit_status() == 0, error.read().decode()
free_bytes = int(disk_output.splitlines()[-1].split()[3])
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when = now.strftime('%Y-%m-%d %H:%M CST')
addition = '\n\n### 20.109 范围配对完成2482更新并保存固定终点；终态评估进行中（' + when + '）\n\n'
addition += ('05:34:42直接检查原screen47112及64KB日志尾部：两臂已完成2482/2482次更新，训练更新耗时7122.62秒；'
    '随后已进入6887条terminal模块留出评估。当时只看到首12行，不能将baseline的6684/6426或77.81%写成终态结果。'
    '本次文档写入前重新确认47112/47242/48128实际进程均存活，最新terminal日志为'
    + str(latest['rows']) + '/6887行，尚无完整terminal receipt或正式9508结果。\n\n'
    '**固定终点已实际保存并复核。** control(center)与local(extent)各620352201 bytes，分别SHA`'
    + check['checkpoints']['control']['sha256'] + '`、`' + check['checkpoints']['local']['sha256'] + '`。'
    '两个文件位于原配对目录，保留完整model、原readout及optimizer，未传回本地或提交Git。'
    '本轮通过远端sha256sum独立核对内容摘要和实际文件大小；同时读取完整2482个批次记录，确认29778个固定fit row ID各出现一次、与holdout无交集、前2481批各12行、最后一批6行。'
    '此项检查未重新解释模型张量语义或计算预测IoU，完整终态独立审计仍由原队列接续。\n\n'
    '**接续与结果边界不变。** 模块holdout仍是预训练主干见过的训练场景；终态评估完成后执行原定CPU审计和固定三组正式评估，再按历史REC5572/4797、同次保护不退化及ScanMask底线判定。'
    '本轮没有新增训练、延长步数、选择中途epoch或改动晋级标准。Nr/Sr原生GPU预检继续由48128等待正式通过后触发，当前尚未开始，整体目标仍active。\n\n')
addition += ('写出两个终点后，磁盘实查剩余' + str(free_bytes) + ' bytes（' + format(free_bytes / 1024**3, '.3f')
    + 'GiB），本轮未再删除权重。fit_complete SHA`' + check['fit_complete_sha256'] + '`；fit批次记录SHA`'
    + check['fit_batches_sha256'] + '`；独立终点检查回执SHA`' + hashlib.sha256(check_raw).hexdigest() + '`。\n')
new = old + addition.encode()
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. §20.109: both Scan range arms completed2482updates;two fixed checkpoints and fit-row coverage verified;terminal6887 evaluation running, no formal result.'
for index, line in enumerate(lines):
    if line.startswith('| ScanRefer matched center/extent range reading |'):
        lines[index] = '| ScanRefer matched center/extent range reading | Both2482updates complete;two620352201B endpoints SHA verified;terminal ' + str(latest['rows']) + '/6887 running | Same145008params/64slots;fixed9508 protected/center/extent still pending;no quality conclusion |'
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
proof = {'time_cst': now.isoformat(), 'section': '20.109', 'bytes': len(new), 'sha256': hashlib.sha256(new).hexdigest(),
    'three_master_copies_equal': master.read_bytes() == desktop.read_bytes() == new, 'processes': processes,
    'terminal_progress': latest, 'free_bytes': free_bytes, 'disk_output': disk_output,
    'fit_endpoint_check_sha256': hashlib.sha256(check_raw).hexdigest(), 'formal_results_present': False, 'goal_complete': False}
sftp.mkdir(archive)
for name, raw in [('fit_endpoint_check.json', check_raw), ('fit_complete.json', (local / 'fit_complete.json').read_bytes()),
                  ('handoff_sync_20_109.json', (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()),
                  ('publish_fit_from_local.py', Path(__file__).read_bytes()),
                  ('collect_fit_from_local.py', (local / 'collect_fit_from_local.py').read_bytes())]:
    (local / name).write_bytes(raw)
    with sftp.open(archive + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(archive + '/' + name, 'rb') as stream:
        assert stream.read() == raw
sftp.close()
client.close()
print(json.dumps(proof), flush=True)
