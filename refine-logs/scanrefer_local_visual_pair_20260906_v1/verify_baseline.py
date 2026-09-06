import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_local_visual_pair_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_pair_20260906_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
digests = {}
for name in ['baseline_rows.json', 'baseline_metrics.json', 'protocol.json']:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=sftp.stat(remote + '/' + name).st_size)
        raw = stream.read()
    (local / name).write_bytes(raw)
    digests[name] = hashlib.sha256(raw).hexdigest()
sftp.close()
client.close()
rows = json.loads((local / 'baseline_rows.json').read_bytes())
recorded = json.loads((local / 'baseline_metrics.json').read_bytes())
protocol = json.loads((local / 'protocol.json').read_bytes())
assert rows['control'] == rows['local'] and len(rows['control']) == 6887
assert [row['row_id'] for row in rows['control']] == protocol['row_ids']['holdout']
assert len(protocol['row_ids']['fit']) == 29778
assert len(set(row['physical_space'] for row in rows['control'])) == 106
assert not set(protocol['physical_spaces']['fit']).intersection(protocol['physical_spaces']['holdout'])
actual = {'rows': 6887, 'mask_miou': sum(row['mask_iou'] for row in rows['control']) / 6887 * 100.}
for field, prefix in [('rec_iou', 'rec'), ('mask_iou', 'mask')]:
    assert all(math.isfinite(row[field]) and 0 <= row[field] <= 1 for row in rows['control'])
    for threshold, suffix in [(.25, '025'), (.5, '050')]:
        actual[prefix + '_hits' + suffix] = sum(row[field] > threshold for row in rows['control'])
for arm in ['control', 'local']:
    for key, value in actual.items():
        if key == 'mask_miou':
            assert abs(value - recorded[arm][key]) < 1e-8
        else:
            assert value == recorded[arm][key]
old_path = repo / 'refine-logs/scanrefer_joint_readout_pair_20260906_v1/baseline_rows.json'
old_raw = old_path.read_bytes()
old_rows = json.loads(old_raw)['detached']
assert len(old_rows) == len(rows['control'])
keys = ['row_id', 'scan_id', 'physical_space', 'point_sha256',
        'rec_iou', 'mask_iou', 'selected_variant_position']
assert all(before[key] == after[key] for before, after in zip(old_rows, rows['control']) for key in keys)
observation_path = local / 'progress_20260906_172007.json'
observation = json.loads(observation_path.read_bytes())
assert observation['process_live']
progress = observation['progress']['SCANREFER LOCAL VISUAL TRAIN']
assert progress['step'] == 128 and progress['total'] == 2482
assert all(math.isfinite(value) for arm in progress['arms'].values() for value in arm.values())
observed = datetime.datetime.fromisoformat(observation['time_cst'])
baseline_seconds = observation['progress']['SCANREFER LOCAL VISUAL EVAL COMPLETE']['elapsed_seconds']
remaining = progress['estimated_training_remaining_seconds']
high = observed + datetime.timedelta(seconds=remaining + baseline_seconds)
low = high - datetime.timedelta(seconds=progress['since_previous_report_seconds'])
result = {'status': 'pass', 'time_cst': datetime.datetime.now(observed.tzinfo).isoformat(),
    'rows': 6887, 'physical_spaces': 106, 'formal_rows': 0,
    'baseline_arms_all_rows_equal': True, 'baseline_matches_previous_e71_v99_start': True,
    'previous_baseline_sha256': hashlib.sha256(old_raw).hexdigest(),
    'compared_previous_row_fields': keys, 'actual_metrics': actual,
    'source_sha256': digests, 'backbone_has_seen_development_scenes': True,
    'observed_steps_per_arm': 128, 'training_process_live_at': observation['time_cst'],
    'seconds_per_paired_step_recent': progress['since_previous_report_seconds'] / 64,
    'seconds_per_paired_step_average': progress['elapsed_seconds'] / progress['step'],
    'baseline_evaluation_seconds': baseline_seconds,
    'estimated_terminal_completion_cst': [low.isoformat(), high.isoformat()],
    'eta_assumptions': 'Recent training throughput remains stable; terminal6887 evaluation takes the same time as baseline. Last report age bounded by one64-step interval.',
    'eta_is_not_measured_completion': True,
    'next_check_cst': '2026-09-06T19:24:00+08:00', 'subsequent_interval_seconds': 240}
with (local / 'baseline_verification.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps(result))
