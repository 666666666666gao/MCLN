import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_pair_20260906_v1'
observation_path = max(local.glob('progress_*.json'))
observation = json.loads(observation_path.read_bytes())
assert observation['process_live']
progress = observation['progress']['SCANREFER LOCAL VISUAL TRAIN']
assert progress['step'] >= 64 and progress['total'] == 2482
assert all(math.isfinite(value) for arm in progress['arms'].values() for value in arm.values())
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
digests = {}
for name in ['baseline_rows.json', 'baseline_metrics.json', 'protocol.json']:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=stream.stat().st_size)
        raw = stream.read()
    (local / name).write_bytes(raw)
    digests[name] = hashlib.sha256(raw).hexdigest()
rows = json.loads((local / 'baseline_rows.json').read_bytes())
recorded = json.loads((local / 'baseline_metrics.json').read_bytes())
protocol = json.loads((local / 'protocol.json').read_bytes())
manifest = json.loads((local / 'input_manifest.json').read_bytes())
assert rows['control'] == rows['local'] and len(rows['control']) == 6887
assert [row['row_id'] for row in rows['control']] == protocol['row_ids']['holdout']
assert len(protocol['row_ids']['fit']) == 29778
assert len(set(row['physical_space'] for row in rows['control'])) == 106
assert not set(protocol['physical_spaces']['fit']).intersection(protocol['physical_spaces']['holdout'])
assert protocol['data_root'] == manifest['data_root'] == '/root/autodl-tmp/DATA_ROOT_mcln_meshsp/'
assert protocol['superpoint_files_sha256'] == hashlib.sha256(json.dumps(manifest['superpoint_files'], sort_keys=True).encode()).hexdigest()

def metrics(records):
    value = {'rows': len(records), 'mask_miou': sum(row['mask_iou'] for row in records) / len(records) * 100.}
    for field, prefix in [('rec_iou', 'rec'), ('mask_iou', 'mask')]:
        assert all(math.isfinite(row[field]) and 0 <= row[field] <= 1 for row in records)
        for threshold, suffix in [(.25, '025'), (.5, '050')]:
            value[prefix + '_hits' + suffix] = sum(row[field] > threshold for row in records)
    return value

actual = metrics(rows['control'])
for arm in ['control', 'local']:
    for key, value in actual.items():
        if key == 'mask_miou':
            assert abs(value - recorded[arm][key]) < 1e-8
        else:
            assert value == recorded[arm][key]
old_path = repo / 'refine-logs/scanrefer_local_visual_pair_20260906_v1/baseline_rows.json'
old_raw = old_path.read_bytes()
old_rows = json.loads(old_raw)['control']
identity_fields = ['row_id', 'scan_id', 'physical_space', 'point_sha256']
assert len(old_rows) == len(rows['control'])
assert all(before[key] == after[key] for before, after in zip(old_rows, rows['control']) for key in identity_fields)
effects = {}
for field, prefix in [('rec_iou', 'rec'), ('mask_iou', 'mask')]:
    for threshold, suffix in [(.25, '025'), (.5, '050')]:
        repair = sum(old[field] <= threshold < new[field] for old, new in zip(old_rows, rows['control']))
        damage = sum(new[field] <= threshold < old[field] for old, new in zip(old_rows, rows['control']))
        effects[prefix + suffix] = {'repair': repair, 'damage': damage, 'net': repair - damage}
observed = datetime.datetime.fromisoformat(observation['time_cst'])
baseline_seconds = observation['progress']['SCANREFER LOCAL VISUAL EVAL COMPLETE']['elapsed_seconds']
average_step_seconds = progress['elapsed_seconds'] / progress['step']
latest = observed + datetime.timedelta(seconds=progress['estimated_training_remaining_seconds'] + baseline_seconds)
earliest = latest - datetime.timedelta(seconds=64 * average_step_seconds)
result = {'status': 'pass', 'time_cst': datetime.datetime.now(observed.tzinfo).isoformat(),
    'rows': 6887, 'physical_spaces': 106, 'formal_rows': 0,
    'baseline_arms_all_rows_equal': True, 'old_new_row_identity_and_points_equal': True,
    'compared_old_new_fields': identity_fields, 'old_baseline_sha256': hashlib.sha256(old_raw).hexdigest(),
    'old_root_metrics': metrics(old_rows), 'correct_mesh_metrics': actual, 'correct_mesh_minus_old_root': effects,
    'interpretation': 'Data-only initial-output comparison; not a trained-method improvement or formal benchmark result.',
    'source_sha256': digests, 'data_root': manifest['data_root'],
    'superpoint_files_sha256': protocol['superpoint_files_sha256'],
    'backbone_has_seen_development_scenes': True, 'observed_steps_per_arm': progress['step'],
    'training_process_live_at': observation['time_cst'], 'observed_progress_sha256': hashlib.sha256(observation_path.read_bytes()).hexdigest(),
    'seconds_per_paired_step_average': average_step_seconds, 'baseline_evaluation_seconds': baseline_seconds,
    'estimated_terminal_completion_cst': [earliest.isoformat(), latest.isoformat()],
    'eta_assumptions': 'Average fit throughput stays similar, terminal evaluation takes baseline duration; last progress report age up to64 steps.',
    'next_check_cst': (earliest - datetime.timedelta(minutes=5)).isoformat(), 'subsequent_interval_seconds': 240,
    'eta_is_not_measured_completion': True}
raw = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
with (local / 'baseline_verification.json').open('xb') as stream:
    stream.write(raw)
with sftp.open(remote + '/baseline_verification.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(result))
