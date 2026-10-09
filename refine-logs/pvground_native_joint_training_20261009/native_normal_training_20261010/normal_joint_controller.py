"""Run one bounded arm through the unchanged ordinary PV training entry."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
admission = json.loads((root / 'admission.json').read_bytes())
assert admission['status'] == 'NATIVE_NORMAL_JOINT_TRAINING_ADMITTED'
assert admission['actual_M0_complete_and_audited'] and admission['preflight_states_not_used']
assert admission['source_mode'] in ('whole_support', 'extremal_support')
protocol = json.loads((root / 'NORMAL_NATIVE_RUN_PROTOCOL.json').read_bytes())
environment = json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest() == admission['env_spec_sha256']
for name, digest in admission['helper_files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
variables = dict(os.environ)
variables.update(environment['env'])
variables['PYTHONPATH'] = protocol['model_source'] + ':' + variables['PYTHONPATH']
variables.update(CUDA_VISIBLE_DEVICES='0', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
                 MASTER_ADDR='127.0.0.1', MASTER_PORT='29691')
mode = admission['source_mode']
assert not (root / 'normal_status.json').exists() and not (root / 'logs').exists()
record = dict(status='running', phase='ordinary_joint_training', source_mode=mode,
    started_cst=datetime.datetime.now().astimezone().isoformat(), planned_epochs=3,
    formal_result=None, original_core_frozen=False, paired_other_arm_training_started=False)
(root / 'normal_status.json').write_text(json.dumps(record,indent=2)+'\n')
command = ['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-B','-u',
    protocol['model_source'] + '/train_dist_mod.py'] + protocol['common_arguments'] + [
    '--native_init_spec',protocol['model_source']+'/init_manifests/'+mode+'.json',
    '--log_dir',str(root/'logs'),'--exp',mode,'--print_freq','128']
assert '--checkpoint_path' not in command and '--frozen' not in command
started = time.monotonic()
with (root/'train.log').open('wb') as stream:
    process = subprocess.Popen(command,cwd=protocol['model_source'],env=variables,
        stdout=stream,stderr=subprocess.STDOUT)
    record.update(child_pid=process.pid,child_argv=command)
    (root/'normal_status.json').write_text(json.dumps(record,indent=2)+'\n')
    exit_code = process.wait()
(root/'train.exit').write_text(str(exit_code)+'\n')
assert exit_code == 0
metrics_paths = list((root/'logs').rglob('native_metrics.jsonl'))
assert len(metrics_paths) == 1
metrics = [json.loads(line) for line in metrics_paths[0].read_text().splitlines()]
assert [row['epoch'] for row in metrics] == [0,1,2,3]
assert all(row['rows']==9508 and row['primary_score']=='last/bbs' for row in metrics)
weights = []
for name in ('best.pth','latest.pth'):
    path = metrics_paths[0].parent/name
    assert path.is_file()
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    weights.append(dict(path=str(path),bytes=path.stat().st_size,sha256=value.hexdigest()))
record.update(status='complete',completed_cst=datetime.datetime.now().astimezone().isoformat(),
    elapsed_seconds=time.monotonic()-started,exit_code=0,metrics=metrics,weights=weights,
    preflight_states_used=False,actual_full_checkpoint_cold_recovery_pending=True,
    formal_result_audit_pending=True,no_result_promoted=True)
(root/'normal_status.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
