"""Sequential two-arm preflight or fit/formal, sharing the existing GPU lock."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('--phase',choices=['preflight','fit'],required=True)
args=parser.parse_args()
phase=args.phase
assert not (root/(phase+'_status.json')).exists()
base=json.loads((root/'control_spec.json').read_bytes())
env=json.loads((Path(base['runtime'])/'env_spec.json').read_bytes())
variables=dict(env['env'])
variables['PYTHONPATH']=str(root)+':'+base['helper_root']+':'+variables['PYTHONPATH']
parents={Path(base[key]):base[key+'_sha256'] for key in ('base_terminal','geometry_terminal')}
official=env['weight_dirs']['scanrefer']
parents[Path(official['path'])]=official['sha256']
assert all(hashlib.sha256(path.read_bytes()).hexdigest()==digest for path,digest in parents.items())
start=time.monotonic()
record=dict(status='running',phase=phase,started_cst=datetime.datetime.now().astimezone().isoformat(),
    protected_best_hits=[5614,4509],completed_runs=[])

def write_status():
    (root/(phase+'_status.json')).write_text(json.dumps(record,indent=2)+'\n')

write_status()
if phase=='fit':
    assert json.loads((root/'preflight_status.json').read_bytes())['status']=='complete'
    assert all(json.loads((root/arm/'preflight.json').read_bytes())['status']=='pass' for arm in ('control','member_target'))
for arm in ('control','member_target'):
    output=root/arm
    modes=('preflight',) if phase=='preflight' else ('train','formal')
    for mode in modes:
        command=['env']+[key+'='+value for key,value in variables.items()]+[
            base['runtime']+'/venv/bin/python','-B','-u',str(root/'run_geometry_fit.py'),
            '--spec',str(root/(arm+'_spec.json')),'--mode',mode]
        with (output/(mode+'.log')).open('x') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            record.update(arm=arm,mode=mode,child_pid=child.pid)
            write_status()
            code=child.wait()
        (output/(mode+'.exit')).write_text(str(code)+'\n')
        if code:
            record.update(status='failed',exit_code=code,finished_cst=datetime.datetime.now().astimezone().isoformat())
            write_status()
            raise SystemExit(code)
        record['completed_runs'].append(arm+'/'+mode)
        write_status()
assert all(hashlib.sha256(path.read_bytes()).hexdigest()==digest for path,digest in parents.items())
record.update(status='complete',exit_code=0,finished_cst=datetime.datetime.now().astimezone().isoformat(),
    elapsed_seconds=time.monotonic()-start,protected_parents_exact=True,
    optimizer_steps_per_arm=2 if phase=='preflight' else 3723,accuracy_result=phase=='fit')
write_status()
print(json.dumps(record),flush=True)
