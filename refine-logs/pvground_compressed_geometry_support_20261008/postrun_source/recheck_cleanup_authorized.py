"""Check remaining closed project weights without inspecting the active experiment."""
import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko


root = Path(__file__).resolve().parent
prior_path = root.parent / 'pvground_mask_support_correction_20261008_v2/postrun/cleanup_receipt.json'
prior = json.loads(prior_path.read_bytes())
policy_path = root.parent / 'pvground_support_boundary_cases_20261007/CLEANUP_STANDING_AUTHORIZATION_20261008.json'
policy = json.loads(policy_path.read_bytes())
assert policy['future_repeated_approval_required'] is False
preserved = prior['preserved_weights']
best = prior['best_checkpoint']
active_name = 'pvground_compressed_geometry_support_20261008'

code = '''import hashlib,json,os
from pathlib import Path
base=Path('/root/autodl-tmp')
weights=[]
for task in sorted(base.iterdir()):
    if not task.is_dir() or not task.name.startswith(('pvground_', 'mcln_pvground_', 'cs_pvground_')):
        continue
    if task.name in ('mcln_pvground_runtime_20260908_v1','pvground_compressed_geometry_support_20261008'):
        continue
    for folder,dirs,names in os.walk(str(task),followlinks=False):
        dirs[:]=[name for name in dirs if name not in ('.git','venv','__pycache__','data','datasets')]
        for name in sorted(names):
            if name.endswith(('.pth','.ckpt')):
                path=Path(folder)/name
                assert not path.is_symlink()
                weights.append(dict(path=str(path),bytes=path.stat().st_size))
best_path=Path(BEST_PATH)
print(json.dumps(dict(weights=weights,best_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest())))
'''.replace('BEST_PATH', repr(best['path']))

client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
stdin, stdout, stderr = client.exec_command(
    '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python -', timeout=120)
stdin.write(code)
stdin.channel.shutdown_write()
remote = json.loads(stdout.read())
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
client.close()
expected = {row['path']: row['bytes'] for row in preserved}
assert {row['path']: row['bytes'] for row in remote['weights']} == expected
assert remote['best_sha256'] == best['sha256']
archive = Path(prior['best_local_archive'])
assert archive.stat().st_size == best['bytes']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == best['sha256']

temporary_weights = []
for task in root.parent.iterdir():
    if task.is_dir() and task.name.startswith(('pvground_', 'mcln_pvground_', 'cs_pvground_')) and task.name != active_name:
        for pattern in ('*.pth', '*.ckpt', '*.pt'):
            temporary_weights.extend(str(path) for path in task.rglob(pattern) if path.is_file())
assert not temporary_weights
stamp = datetime.datetime.now().astimezone().isoformat()
receipt = dict(time_cst=stamp, status='CLOSED_NONBEST_WEIGHTS_ALREADY_CLEAN_NO_NEW_ELIGIBLE_FILES',
    standing_user_instruction='清理，以后不用我审批无用的权重这些', repeated_user_approval_required=False,
    remaining_required_weights=remote['weights'], required_weight_count=len(expected),
    best_remote_and_local_sha256=best['sha256'], local_temporary_weights=temporary_weights,
    active_experiment_excluded=active_name, training_progress_queries=0,
    new_deleted_files=0, new_released_bytes=0,
    previous_cleanup_receipt=str(prior_path), previous_released_bytes=prior['released_file_bytes'],
    neural_forwards=0, optimizer_updates=0)
destination = root / 'CLEANUP_RECHECK.json'
assert not destination.exists()
destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
note = ('\nPV-Ground '+stamp+': user reconfirmed automatic cleanup without per-file approval. '
    'Read-only closed project inventory and protected best local/remote hash agree: six necessary weights, '
    'no generated local temporary weights and no new closed nonbest remote weights. Prior completed '
    'cleanup188659678B; no new deletions. Active compressed support experiment entirely excluded from '
    'remote inspection; zero training progress queries. Best, restoration dependencies, data and unique '
    'evidence retained. Receipt '+str(destination)+'.\n')
for path in (Path('C:/Users/gb/MEMORY.md'), Path('C:/Users/gb/memory/2026-10-08.md')):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(status=receipt['status'], required_weights=len(expected),
    new_deleted_files=0, previous_released_bytes=prior['released_file_bytes'], receipt=str(destination))))
