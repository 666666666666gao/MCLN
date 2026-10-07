"""Execute one reviewed CPU-only inspection after actual closed-array recount."""
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).resolve().parents[1]
destination = local / 'closed_weight_inspection.json'
assert not destination.exists()
review = json.loads((local / 'postrun/INSPECTION_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY' and review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
summary_path = local / 'analysis/SUMMARY.json'
summary = json.loads(summary_path.read_bytes())
assert summary['status'] == 'ACTUAL_CLOSED_SHARED_PARENT_FACE_PAIR_RECOUNTED'
assert summary['formal_rows_per_result'] == 9508 and summary['metric_best_candidate']['arm'] == 'protected_geometry_parent'
for stage in summary['cpu_stage_recounts'].values():
    assert all(not any(values.values()) for values in stage['cpu_selected_threshold_flips_by_box_type'].values())
    assert all(not any(values.values()) for values in stage['cpu_full256_oracle_label_mismatches'].values())
eligible = [row for row in summary['table'] if row['stage'] in ('protected', 'formal')]
decision = dict(winner='protected_geometry_parent',
    hits={row['arm'] if row['arm'] == 'protected_geometry_parent' else row['arm'] + '/formal':
        [row['rec_hits25'], row['rec_hits50']] for row in eligible},
    summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest())
wait = json.loads((local / 'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode'] == 0
spec = json.loads((local / 'pair_spec.json').read_bytes())
root = spec['root']
assert root == '/root/autodl-tmp/pvground_face_support_20261007'
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime'] + '/env_spec.json', 'rb') as stream:
    environment = json.loads(stream.read())
assert hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
source = (local / 'postrun/inspect_closed_face_weights.py').read_bytes()
with sftp.open(root + '/inspect_closed_face_weights.py', 'wx') as stream:
    stream.write(source)
with sftp.open(root + '/inspect_closed_face_weights.py', 'rb') as stream:
    assert stream.read() == source
command = shlex.join(['env'] + [key + '=' + value for key, value in environment['env'].items()] +
    [spec['runtime'] + '/venv/bin/python', '-B', root + '/inspect_closed_face_weights.py', json.dumps(decision)])
_, stdout, stderr = client.exec_command(command, timeout=300)
raw = stdout.read(); assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
assert record['status'] == 'CLOSED_FIXED_FACE_CHECKPOINTS_CPU_INSPECTED' and record['winner'] == decision['winner']
assert record['script_sha256'] == hashlib.sha256(source).hexdigest()
assert record['weights_created'] == record['weights_deleted'] == record['optimizer_updates'] == 0
with sftp.open(root + '/closed_weight_inspection.json', 'rb') as stream:
    receipt = stream.read()
assert json.loads(receipt) == record
destination.write_bytes(receipt)
sftp.close(); client.close()
print(json.dumps(dict(status=record['status'], winner=record['winner'], checkpoints=len(record['identities']),
    GPU_forward_replayed=False, weights_deleted=0)), flush=True)
