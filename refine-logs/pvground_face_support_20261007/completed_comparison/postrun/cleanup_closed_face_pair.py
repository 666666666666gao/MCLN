"""Clear this audited negative pair; preserve protected weights and local arrays."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


root = Path('/root/autodl-tmp/pvground_face_support_20261007')
assert root.resolve() == root
decision = json.loads(sys.argv[1])
assert decision['winner'] == 'protected_geometry_parent'
assert not (root / 'weight_retention.json').exists()
assert not (root / 'archived_array_cleanup_receipt.json').exists()
assert (root / 'fit_controller.exit').read_text().strip() == '0'
status = json.loads((root / 'fit_status.json').read_bytes())
assert status['status'] == 'complete' and status['protected_parents_exact']
spec = json.loads((root / 'pair_spec.json').read_bytes())
pattern = '^' + spec['runtime'] + '/venv/bin/python -B -u ' + str(root) + '/controller.py --phase fit$'
assert subprocess.run(['pgrep', '-af', pattern], stdout=subprocess.PIPE).returncode == 1
inspection_path = root / 'closed_weight_inspection.json'
assert sha(inspection_path) == decision['inspection_sha256']
inspection = json.loads(inspection_path.read_bytes())
assert inspection['winner'] == decision['winner'] and inspection['decision']['summary_sha256'] == decision['summary_sha256']
assert inspection['weights_created'] == inspection['weights_deleted'] == inspection['optimizer_updates'] == 0
parent = Path(spec['selected_terminal'])
assert sha(parent) == spec['selected_terminal_sha256']
assert sha(spec['base_terminal']) == spec['base_terminal_sha256']
manifest_path = root / 'archived_array_cleanup_manifest.json'
policy_path = root / 'CLEANUP_POLICY.json'
assert sha(manifest_path) == decision['manifest_sha256'] and sha(policy_path) == decision['policy_sha256']
manifest = json.loads(manifest_path.read_bytes())
policy = json.loads(policy_path.read_bytes())
assert policy['user_reply'] == '清理，以后不用我审批无用的权重这些'
assert policy['future_repeated_approval_required'] is False
assert manifest['root'] == str(root) and manifest['directories'] == ['initial_formal', 'formal']
assert manifest['count'] == len(manifest['files']) == 2378
assert manifest['bytes'] == sum(item['bytes'] for item in manifest['files']) == 330047051
assert len({item['name'] for item in manifest['files']}) == 2378
arrays = []
for item in manifest['files']:
    relative = Path(item['name'])
    assert len(relative.parts) == 2 and relative.parts[0] in manifest['directories']
    assert relative.name.startswith('batch_') and relative.suffix == '.npz'
    path = root / relative
    assert path.resolve() == path and root in path.parents
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    arrays.append(path)
weights = []
for arm in ('face_center', 'face_region'):
    identifier = arm + '/formal'
    identity = inspection['identities'][identifier]
    path = root / arm / 'terminal.pth'
    assert path.resolve() == path and str(path) == identity['path'] and root in path.parents
    assert path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256']
    weights.append((identifier, path, identity))
# The complete fixed list is checked before either deletion loop begins.
before = shutil.disk_usage(root).free
for _, path, _ in weights:
    path.unlink()
for path in arrays:
    path.unlink()
assert all(not path.exists() for _, path, _ in weights) and all(not path.exists() for path in arrays)
assert sha(parent) == spec['selected_terminal_sha256'] and sha(spec['base_terminal']) == spec['base_terminal_sha256']
stamp = datetime.datetime.now().astimezone().isoformat()
retention = dict(time_cst=stamp, status='CLOSED_NONBEST_WEIGHTS_REMOVED',
    retained_best=dict(candidate='protected_geometry_parent', hits=[5598, 4848], **inspection['identities']['protected_geometry_parent']),
    deleted=[dict(candidate=identifier, **identity) for identifier, _, identity in weights],
    released_bytes=sum(identity['bytes'] for _, _, identity in weights), negative_weight_archive_created=False,
    original_G_preserved=True, official_PV_paths_touched=0, V99_paths_touched=0,
    inspection_sha256=decision['inspection_sha256'], fresh_audit_sha256=decision['audit_sha256'])
archive = dict(time_cst=stamp, status='STANDING_AUTHORIZED_ARCHIVED_ARRAYS_REMOVED', root=str(root),
    deleted_count=len(arrays), released_file_bytes=manifest['bytes'], deleted_files=manifest['files'],
    deleted_manifest_sha256=decision['manifest_sha256'], policy_sha256=decision['policy_sha256'],
    local_archive=manifest['local_archive'], local_archive_preserved=True, free_bytes_before=before,
    free_bytes_after=shutil.disk_usage(root).free, best_weights_touched=0, text_logs_touched=0, datasets_touched=0)
(root / 'weight_retention.json').write_text(json.dumps(retention, indent=2) + '\n')
(root / 'archived_array_cleanup_receipt.json').write_text(json.dumps(archive, indent=2) + '\n')
print(json.dumps(dict(weights_deleted=len(weights), weight_bytes=retention['released_bytes'],
    arrays_deleted=len(arrays), array_bytes=manifest['bytes'], free_bytes_after=archive['free_bytes_after'])), flush=True)
