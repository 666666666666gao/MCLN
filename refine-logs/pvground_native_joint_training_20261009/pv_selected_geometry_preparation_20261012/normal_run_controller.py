"""Run one admitted arm through the native trainer under the existing GPU lock."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
protocol = json.loads((root / 'protocol.json').read_bytes())
admission = json.loads((root / 'admission.json').read_bytes())
assert admission['actual_pair_preflight_complete']
assert admission['fresh_initialization_without_preflight_states']
assert admission['MCLN_terminal_and_device_released']
assert protocol['arm'] in ('native_control', 'selected_geometry')
assert protocol['max_epochs'] == 8 and protocol['patience'] == 2
assert protocol['target_hits'] == [5705, 4945]
assert protocol['primary_score'] == 'last/bbs' and protocol['formal_rows'] == 9508
spec = json.loads(Path(protocol['native_init_spec']).read_bytes())
assert spec['use_selected_geometry_supervision'] == (protocol['arm'] == 'selected_geometry')
assert spec['use_g_supervision'] and not spec['use_selected_mask_supervision']
environment = json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
variables = dict(os.environ)
variables.update(environment['env'])
variables['PYTHONPATH'] = protocol['model_source'] + ':' + variables['PYTHONPATH']
variables.update(CUDA_VISIBLE_DEVICES='0', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
                 MASTER_ADDR='127.0.0.1', MASTER_PORT='29679')
assert not (root / 'normal_status.json').exists() and not (root / 'logs').exists()
record = dict(status='running', phase='native_normal_training', arm=protocol['arm'],
    started_cst=datetime.datetime.now().astimezone().isoformat(), max_epochs=8,
    patience=2, actual_metrics=None, preflight_states_used=False)
(root / 'normal_status.json').write_text(json.dumps(record, indent=2) + '\n')
command = ['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python', '-B', '-u',
    protocol['model_source'] + '/train_dist_mod.py'] + protocol['common_arguments'] + [
    '--native_init_spec', protocol['native_init_spec'], '--log_dir', str(root / 'logs'),
    '--exp', protocol['arm'], '--print_freq', '128']
assert '--checkpoint_path' not in command and '--frozen' not in command
started = time.monotonic()
with (root / 'train.log').open('wb') as stream:
    process = subprocess.Popen(command, cwd=protocol['model_source'], env=variables,
                               stdout=stream, stderr=subprocess.STDOUT)
    record.update(child_pid=process.pid, child_argv=command)
    (root / 'normal_status.json').write_text(json.dumps(record, indent=2) + '\n')
    exit_code = process.wait()
(root / 'train.exit').write_text(str(exit_code) + '\n')
assert exit_code == 0
metrics_paths = list((root / 'logs').rglob('native_metrics.jsonl'))
assert len(metrics_paths) == 1
metrics = [json.loads(line) for line in metrics_paths[0].read_text().splitlines()]
assert 2 <= len(metrics) <= 9
assert [row['epoch'] for row in metrics] == list(range(len(metrics)))
assert all(row['rows'] == 9508 and row['primary_score'] == 'last/bbs' for row in metrics)
stop = json.loads((metrics_paths[0].parent / 'native_stop.json').read_bytes())
assert stop['history'] == metrics and stop['max_epochs'] == 8
assert stop['reason'] in ('max_epochs', 'two_consecutive_strict_declines')
if stop['reason'] == 'two_consecutive_strict_declines':
    assert len(metrics) >= 3
    assert metrics[-1]['hits050'] < metrics[-2]['hits050'] < metrics[-3]['hits050']
else:
    assert metrics[-1]['epoch'] == 8
weights = []
for name in ('best.pth', 'latest.pth'):
    path = metrics_paths[0].parent / name
    assert path.is_file()
    weights.append(dict(path=str(path), bytes=path.stat().st_size))
record.update(status='complete', completed_cst=datetime.datetime.now().astimezone().isoformat(),
    elapsed_seconds=time.monotonic() - started, exit_code=0, actual_metrics=metrics,
    stop=stop, weights=weights, formal_result_audit_pending=True,
    actual_full_checkpoint_cold_recovery_pending=True, new_result_promoted=False,
    full_goal_complete=False)
(root / 'normal_status.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
