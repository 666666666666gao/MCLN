"""Read a closed arm's small receipts while the other arm continues."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import paramiko

local = Path(__file__).resolve().parent
observation_path=local/'fit_observations/observation_11.json'
observation=json.loads(observation_path.read_bytes())
assert observation['controller_alive'] and observation['status']['arm'] == 'member_target'
assert observation['status']['completed_runs'] == ['control/train', 'control/formal']
launch = json.loads((local / 'fit_launch.json').read_bytes())
destination = local / 'closed_control_receipts'
assert not destination.exists()
destination.mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
names = ['control/receipt.json', 'control/formal/receipt.json', 'control/formal_restore.json',
         'control/train.exit', 'control/formal.exit', 'control/formal/rows.jsonl',
         'control/initial/receipt.json', 'control/terminal/receipt.json']
identities = {}
for name in names:
    with sftp.open(launch['root'] + '/' + name, 'rb') as stream:
        raw = stream.read()
    assert len(raw) == sftp.stat(launch['root'] + '/' + name).st_size
    path = destination / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    identities[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
sftp.close()
client.close()
fit = json.loads((destination / 'control/receipt.json').read_bytes())
formal = json.loads((destination / 'control/formal/receipt.json').read_bytes())
restore = json.loads((destination / 'control/formal_restore.json').read_bytes())
assert all((destination / ('control/' + name + '.exit')).read_text().strip() == '0' for name in ('train', 'formal'))
assert fit['status'] == 'complete' and fit['training_steps'] == 3723 and fit['fit_rows'] == 29778
assert fit['parent_and_zero_R_states_exact'] and fit['fit_seen_exactly_once']
assert formal['status'] == 'pass' and formal['rows'] == 9508
assert restore['strict_model_restore'] and restore['optimizer']['all_keys_moments_steps_and_groups_exact']
assert restore['terminal_sha256'] == fit['terminal_sha256']
assert identities['control/formal/rows.jsonl']['sha256']==formal['rows_sha256']
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_phase_observation_cst=observation['time_cst'], status='CLOSED_CONTROL_RECEIPTS_READ', files=identities,
    fit_updates=3723, fit_rows=29778, formal_rows=9508, formal_finished_cst=formal['time_cst'],
    formal_elapsed_seconds=formal['elapsed_seconds'], native_bbs=formal['metrics']['bbs'],
    parent_and_zero_R_states_exact=True, actual_formal_restore_pass=True,
    full_pair_complete=False, CPU_full_rows_recount_pending=True, integrity_review_pending=True,
    best_weight_promoted=False, downloaded_weights=0, downloaded_full_rows=9508,
    inference_or_optimizer_replayed=False, weights_deleted=0)
(local / 'CONTROL_CLOSED_INTAKE.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
