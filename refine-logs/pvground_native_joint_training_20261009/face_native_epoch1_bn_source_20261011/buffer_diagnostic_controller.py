"""One serial diagnostic stage, admitted only after normal training closes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
admission = json.loads((root / 'admission.json').read_bytes())
assert admission['stage'] in ('sanity', 'full')
assert admission['normal_exit_code'] == 0 and admission['saved_epoch'] == 3
assert not (root / 'diagnostic_status.json').exists()
environment = json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == admission['env_spec_sha256']
variables = dict(os.environ)
variables.update(environment['env'])
variables['PYTHONPATH'] = admission['source'] + ':' + variables['PYTHONPATH']
variables.update(CUDA_VISIBLE_DEVICES='0', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
                 MASTER_ADDR='127.0.0.1', MASTER_PORT='29693')
record = dict(status='running', stage=admission['stage'],
    started_cst=datetime.datetime.now().astimezone().isoformat(),
    normal_training_restarted=False, optimizer_updates=0, written_weights=0)
(root / 'diagnostic_status.json').write_text(json.dumps(record, indent=2) + '\n')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
started = time.monotonic()


def execute(name, command):
    with (root / (name + '.log')).open('wb') as stream:
        process = subprocess.Popen(command, cwd=admission['source'], env=variables,
            stdout=stream, stderr=subprocess.STDOUT)
        record.update(phase=name, child_pid=process.pid, child_argv=command)
        (root / 'diagnostic_status.json').write_text(json.dumps(record, indent=2) + '\n')
        exit_code = process.wait()
    (root / (name + '.exit')).write_text(str(exit_code) + '\n')
    assert exit_code == 0


if admission['stage'] == 'sanity':
    execute('cold_recovery', [runtime, '-B', '-u', str(root / 'recover_face_normal_terminal.py'),
        '--protocol', admission['protocol'], '--terminal', admission['terminal'],
        '--output', str(root / 'cold_recovery')])
    recovery = json.loads((root / 'cold_recovery/RECOVERY_RECEIPT.json').read_bytes())
    assert recovery['status'] == 'FACE_NORMAL_BEST_AND_FIXED_E3_FULL_COLD_RECOVERY_COMPLETE'
    assert all(row['full_recovery_exact'] for row in recovery['checkpoints'])
    latest = [row for row in recovery['checkpoints'] if row['name'] == 'latest']
    assert len(latest) == 1 and latest[0]['identity'] == admission['checkpoint']
execute('replay', [runtime, '-B', '-u', str(root / 'paired_native_buffer_replay.py'),
    '--protocol', admission['protocol'], '--source-manifest', admission['source_manifest'],
    '--checkpoint', admission['checkpoint']['path'], '--output', str(root / 'replay'),
    '--max-batches', '1' if admission['stage'] == 'sanity' else '0'])
result = json.loads((root / 'replay/RESULT.json').read_bytes())
assert result['checkpoint'] == admission['checkpoint']
assert result['full_trained_state_restored_exactly'] and result['all_non_bn_states_unchanged']
assert result['scope'] == ('REAL_FIRST_BATCH_8_SANITY' if admission['stage'] == 'sanity' else 'FULL_9508_DIAGNOSTIC_REPLAY')
record.update(status='complete', completed_cst=datetime.datetime.now().astimezone().isoformat(),
    elapsed_seconds=time.monotonic() - started, exit_code=0,
    result=result, full_goal_complete=False, best_promoted=False)
(root / 'diagnostic_status.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
