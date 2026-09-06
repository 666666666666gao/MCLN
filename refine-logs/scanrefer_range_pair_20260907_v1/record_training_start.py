import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_range_pair_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_range_pair_20260907_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
files = {}
for name in ['baseline_rows.json', 'baseline_metrics.json', 'protocol.json']:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=stream.stat().st_size)
        raw = stream.read()
    (local / name).write_bytes(raw)
    files[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
rows = json.loads((local / 'baseline_rows.json').read_bytes())
metrics = json.loads((local / 'baseline_metrics.json').read_bytes())
protocol = json.loads((local / 'protocol.json').read_bytes())
assert rows['control'] == rows['local'] and metrics['control'] == metrics['local']
assert [row['row_id'] for row in rows['control']] == protocol['row_ids']['holdout']
assert len(rows['control']) == len(set(row['row_id'] for row in rows['control'])) == 6887
assert sorted(set(row['physical_space'] for row in rows['control'])) == protocol['physical_spaces']['holdout']
assert len(protocol['physical_spaces']['holdout']) == 106
assert not set(protocol['physical_spaces']['fit']).intersection(protocol['physical_spaces']['holdout'])
value = {'rows': len(rows['control']), 'mask_miou': sum(row['mask_iou'] for row in rows['control']) / 6887 * 100.}
for suffix, threshold in [('025', .25), ('050', .5)]:
    for field in ['rec', 'mask']:
        value[field + '_hits' + suffix] = sum(row[field + '_iou'] > threshold for row in rows['control'])
for key, actual in value.items():
    if key == 'mask_miou':
        assert abs(actual - metrics['control'][key]) < 1e-8
    else:
        assert actual == metrics['control'][key]
assert (value['rec_hits025'], value['rec_hits050'], value['mask_hits025'], value['mask_hits050']) == (6684, 6426, 6511, 6097)
assert abs(value['mask_miou'] - 77.8108607870114) < 1e-8
progress = json.loads((local / 'progress_20260907_033641.json').read_bytes())
assert progress['screen_live'] and progress['progress']['SCANREFER RANGE TRAIN']['step'] == 64
_, output, error = client.exec_command('ps -p 47112,47242 -o pid,stat,etime,args', timeout=30)
processes = output.read().decode()
assert output.channel.recv_exit_status() == 0, error.read().decode()
assert '47112' in processes and '47242' in processes
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
record = {'schema': 'mcln-scanrefer-range-baseline-row-check-v1', 'status': 'pass', 'time_cst': now.isoformat(),
    'files': files, 'module_holdout_rows': 6887, 'module_holdout_physical_spaces': 106,
    'both_arms_rows_exactly_equal': True, 'row_metrics_recomputed': value,
    'matches_previous_correct_mesh_initial_metrics': True, 'first_training_observation': progress,
    'processes_reverified': processes, 'formal_rows': 0,
    'limits': 'Checks saved baseline row identity and metric counts; not a new IoU-from-box audit, terminal audit, or formal result.'}
raw = (json.dumps(record, indent=2, sort_keys=True) + '\n').encode()
(local / 'baseline_row_check.json').write_bytes(raw)
with sftp.open(remote + '/baseline_row_check.json', 'wx') as stream:
    stream.write(raw)
for name, content in [('record_training_start.py', Path(__file__).read_bytes()),
                      ('progress_20260907_033641.json', (local / 'progress_20260907_033641.json').read_bytes()),
                      ('progress_20260907_033641.txt', (local / 'progress_20260907_033641.txt').read_bytes())]:
    (local / name).write_bytes(content)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(content)
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document') / master.name
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == '5d491265e1a3603b1646430e39627fc95780debbdff35342597a583f56677d29'
assert desktop.read_bytes() == old
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
when = now.strftime('%Y-%m-%d %H:%M CST')
addition = '\n\n### 20.106 范围配对零更新基线完整一致，实际进入训练（' + when + '）\n\n'
addition += ('03:36:41实际核验原screen47112仍存活，日志已完成6887条baseline并推进到每臂64/2482更新。'
    '零更新评估耗时1929.63秒，两臂保存的逐条row dict完全一致；本轮收集并独立复核6887个固定row ID、106个模块holdout物理场景及计数。'
    'REC6684/6426，Mask6511/6097，mIoU77.8108607870114%，与此前正确mesh起点相同。'
    '该检查是保存行与指标计数复核，没有重新根据框/GT计算IoU，不能冒充完整终态审计；formal_rows=0。\n\n'
    '64次实际配对更新耗时184.90秒，约2.889秒/配对更新；control/local当前native loss分别10.2222729/10.2223883，'
    '裁剪前梯度范数1.14780/1.17615，均有限。它只证明固定GT训练路径正在更新，不是验证集改善。'
    '按首64步估算剩余训练约6985.82秒（116.4分钟），约05:30左右结束fit、写两个固定终点；'
    '随后另需约32分钟终态模块评估，正式三组9508评估与独立审计尚在其后。'
    '首次时间估算会随吞吐变化，不承诺固定完成时刻。没有中途改epoch、学习率、候选、权重或晋级规则。\n\n'
    '原posttraining screen47242在本次记录时也确认存活，继续240秒轮询、完成原定终态和正式接续。'
    '无新的Nr/Sr GPU任务。后续本地深查优先安排在估计fit终点前数分钟（约05:25），'
    '远端队列期间仍按240秒保留原进程观察；如队列记录终止/异常，则按实际状态提前处理。'
    '本轮是完成基线证据检查后的已核验等待，主目标仍active，正式成绩及上一轮CPU接入结论均保持原口径。\n\n')
addition += 'baseline_rows SHA`' + files['baseline_rows.json']['sha256'] + '`；baseline_row_check SHA`' + hashlib.sha256(raw).hexdigest() + '`。\n'
new = old + addition.encode()
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. §20.106:range baseline6887 exactly equal;actual64/2482 updates/arm observed03:36:41;estimated fit end~05:30;queue240s remains live.'
tracker_raw = ('\n'.join(lines) + '\n').encode()
for name, content in [('docs/' + master.name, new), ('refine-logs/EXPERIMENT_TRACKER.md', tracker_raw)]:
    with sftp.open(runtime + name, 'wb') as stream:
        stream.set_pipelined(True)
        stream.write(content)
    with sftp.open(runtime + name, 'rb') as stream:
        stream.prefetch(file_size=len(content))
        assert stream.read() == content
master.write_bytes(new)
desktop.write_bytes(new)
tracker.write_bytes(tracker_raw)
proof = {'time_cst': now.isoformat(), 'section': '20.106', 'bytes': len(new), 'sha256': hashlib.sha256(new).hexdigest(),
    'three_master_copies_equal': master.read_bytes() == desktop.read_bytes() == new,
    'training_screen': 47112, 'queue_screen': 47242, 'actual_steps_per_arm_observed': 64,
    'baseline_row_check_sha256': hashlib.sha256(raw).hexdigest(), 'goal_complete': False}
proof_raw = (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()
(local / 'handoff_sync_20_106.json').write_bytes(proof_raw)
with sftp.open(remote + '/handoff_sync_20_106.json', 'wx') as stream:
    stream.write(proof_raw)
sftp.close()
client.close()
print(json.dumps(proof), flush=True)
