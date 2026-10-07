"""Use standing user authorization only after closed audit, CPU witness and hashes."""
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


local = Path(__file__).resolve().parents[1]
assert not (local / 'weight_retention.json').exists()
assert not (local / 'archived_array_cleanup_receipt.json').exists()
review = json.loads((local / 'postrun/CLOSED_CONSUMERS_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY' and review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert sha(Path(item['path'])) == item['sha256']
summary_path = local / 'analysis/SUMMARY.json'
inspection_path = local / 'closed_weight_inspection.json'
audit_path = local / 'analysis/EXPERIMENT_AUDIT.json'
summary = json.loads(summary_path.read_bytes())
inspection = json.loads(inspection_path.read_bytes())
audit = json.loads(audit_path.read_bytes())
assert summary['metric_best_candidate']['arm'] == inspection['winner'] == 'protected_geometry_parent'
assert audit['fresh_context'] and audit['execution_scope'] == 'TERMINAL_SOURCE_AND_ACTUAL_ARTIFACTS'
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
for name, path in (('analysis/SUMMARY.json', summary_path), ('closed_weight_inspection.json', inspection_path)):
    assert audit['actual_file_digests'][name]['sha256'] == sha(path)
policy_path = local / 'CLEANUP_POLICY.json'
policy = json.loads(policy_path.read_bytes())
assert policy['future_repeated_approval_required'] is False and policy['user_reply'] == '清理，以后不用我审批无用的权重这些'
manifest_path = local / 'archived_array_cleanup_manifest.json'
manifest = json.loads(manifest_path.read_bytes())
intake = json.loads((local / 'complete_fit/INTAKE.json').read_bytes())
expected = {item['name']: item for item in intake['files']}
assert manifest['count'] == 2378 and manifest['bytes'] == 330047051
for item in manifest['files']:
    assert item == expected[item['name']]
    path = local / 'complete_fit' / item['name']
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
wait = json.loads((local / 'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode'] == 0
spec = json.loads((local / 'pair_spec.json').read_bytes())
root = spec['root']; assert root == manifest['root'] == '/root/autodl-tmp/pvground_face_support_20261007'
decision = dict(winner=inspection['winner'], summary_sha256=sha(summary_path), inspection_sha256=sha(inspection_path),
    audit_sha256=sha(audit_path), manifest_sha256=sha(manifest_path), policy_sha256=sha(policy_path))
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime'] + '/env_spec.json', 'rb') as stream:
    environment = json.loads(stream.read())
assert hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
for name, path in (('cleanup_closed_face_pair.py', local / 'postrun/cleanup_closed_face_pair.py'),
                   ('archived_array_cleanup_manifest.json', manifest_path), ('CLEANUP_POLICY.json', policy_path)):
    raw = path.read_bytes()
    with sftp.open(root + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(root + '/' + name, 'rb') as stream:
        assert stream.read() == raw
command = shlex.join(['env'] + [key + '=' + value for key, value in environment['env'].items()] +
    [spec['runtime'] + '/venv/bin/python', '-B', root + '/cleanup_closed_face_pair.py', json.dumps(decision)])
_, stdout, stderr = client.exec_command(command, timeout=180)
raw = stdout.read(); assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
assert record['weights_deleted'] == 2 and record['arrays_deleted'] == 2378 and record['array_bytes'] == 330047051
for name in ('weight_retention.json', 'archived_array_cleanup_receipt.json'):
    with sftp.open(root + '/' + name, 'rb') as stream:
        receipt = stream.read()
    (local / name).write_bytes(receipt)
sftp.close(); client.close()
print(json.dumps(record), flush=True)
