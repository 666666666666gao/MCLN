import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
pair = repo / 'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1'
previous = repo / 'refine-logs/scanrefer_local_visual_pair_20260906_v1'
before = json.loads((previous / 'input_manifest.json').read_bytes())
after = json.loads((pair / 'input_manifest.json').read_bytes())
keys = ['artifacts','batch_size','clip_norm','core_learning_rate','epochs','local_learning_rate','loss',
    'model_source','readouts_frozen','source_manifest_sha256','split_protocol_sha256','split_salt',
    'steps_per_arm','weight_decay','fixed_endpoint_formal_evaluation']
assert all(before[key] == after[key] for key in keys)
for name, digest in after['files'].items():
    assert hashlib.sha256((pair / name).read_bytes()).hexdigest() == digest
    assert (pair / name).read_bytes() == (repo / name).read_bytes()
assert hashlib.sha256((pair / 'plan.md').read_bytes()).hexdigest() == after['plan_sha256']
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
verification = {'time_cst': now.isoformat(), 'unchanged_experimental_fields': keys,
    'old_manifest_sha256': hashlib.sha256((previous / 'input_manifest.json').read_bytes()).hexdigest(),
    'new_manifest_sha256': hashlib.sha256((pair / 'input_manifest.json').read_bytes()).hexdigest(),
    'entry_archive_matches_current_source': True, 'pre_registered_plan_sha256_verified': after['plan_sha256'],
    'result': 'input/provenance comparison pass; not a quality result'}
check_raw = (json.dumps(verification, indent=2, sort_keys=True) + '\n').encode()
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + now.strftime('%Y-%m-%d %H:%M CST') + '. Acceptance: §20.79; corrected-data Scan result and repeat: §20.97.'
for index, line in enumerate(lines):
    if line.startswith('| Weight storage cleanup |'):
        lines[index] = '| Weight storage cleanup | Complete14 explicit actions across two receipts;9.529GiB cumulative freed;10.10GiB free at22:21 | Ten duplicate copies share originals;four failed endpoints retired after audits;all logs/rows and six protected hashes retained |'
    if line.startswith('| Nr/Sr native real-input CPU probe |'):
        lines[index] = '| Nr/Sr native real-input CPU probe | Correct-mesh32 records/64 actual samples PASS20:57;all64 old/new pointSHA equal;input metadata boundaries recorded | GPU0/updates0/checkpoints0;v2 GPU entry prepared;Scan v3 REC failed so conditional GPU launch not executed;see20.97 |'
tracker_raw = ('\n'.join(lines) + '\n').encode()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open('/root/autodl-tmp/mcln_' + pair.name + '/same_setting_check.json', 'wx') as stream:
    stream.write(check_raw)
remote_tracker = '/home/gb/new butd/butd_detr-main/MCLN-main/refine-logs/EXPERIMENT_TRACKER.md'
with sftp.open(remote_tracker, 'wb') as stream:
    stream.write(tracker_raw)
with sftp.open(remote_tracker, 'rb') as stream:
    assert stream.read() == tracker_raw
sftp.close()
client.close()
(pair / 'same_setting_check.json').write_bytes(check_raw)
tracker.write_bytes(tracker_raw)
(repo / ('refine-logs/EXPERIMENT_TRACKER_' + now.strftime('%Y%m%d_%H%M%S') + '.md')).write_bytes(tracker_raw)
(pair / 'finalize_records.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'same_setting_fields_verified': len(keys), 'tracker_sha256': hashlib.sha256(tracker_raw).hexdigest(), 'time_cst': now.isoformat()}))
