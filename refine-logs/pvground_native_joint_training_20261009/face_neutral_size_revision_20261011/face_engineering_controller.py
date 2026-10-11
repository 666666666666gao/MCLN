"""Run exactly one reviewed matcher mode through two real normal updates."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['source_conditioned','without_additional_source'],required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
admission=json.loads((root/'admission.json').read_bytes())
assert admission['status']=='NATIVE_FACE_ENGINEERING_ADMITTED'
assert args.mode==admission['mode']
protocol_path=root/'admitted_protocol.json'
protocol=json.loads(protocol_path.read_bytes())
assert protocol['face_residual_mode']==args.mode and protocol['serial_gpu_preflight_admitted']
assert not protocol['full_training_admitted']
assert hashlib.sha256(protocol_path.read_bytes()).hexdigest()==admission['protocol_sha256']
helper=root.parent.parent/'native_face_preflight.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()==admission['helper_sha256']
source=Path(protocol['model_source'])
for name,digest in protocol['source_sha256'].items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
env=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==admission['env_spec_sha256']
assert shutil.disk_usage(source).free>admission['reserve_bytes']
variables=dict(os.environ)
variables.update(env['env'])
variables['PYTHONPATH']=str(source)+':'+variables['PYTHONPATH']
variables.update(CUDA_VISIBLE_DEVICES='0',WORLD_SIZE='1',RANK='0',LOCAL_RANK='0',MASTER_ADDR='127.0.0.1',MASTER_PORT='29741')
assert not (root/'engineering_status.json').exists()
status=dict(status='running',mode=args.mode,started_cst=datetime.datetime.now().astimezone().isoformat(),
    formal_accuracy=None,full_training_started=False)
(root/'engineering_status.json').write_text(json.dumps(status,indent=2)+'\n')
command=[admission['runtime'],'-B','-u',str(helper),'--protocol',str(protocol_path),
    '--mode','extremal_support','--initial_checkpoint',admission['initial_checkpoint'],
    '--output',str(root/'witness')]
start=time.monotonic()
with (root/'preflight.log').open('wb') as stream:
    child=subprocess.Popen(command,cwd=str(source),env=variables,stdout=stream,stderr=subprocess.STDOUT)
    status.update(child_pid=child.pid,child_argv=command)
    (root/'engineering_status.json').write_text(json.dumps(status,indent=2)+'\n')
    result=child.wait()
(root/'preflight.exit').write_text(str(result)+'\n')
status.update(status='failed' if result else 'checking_receipt',exit_code=result,
    child_closed_cst=datetime.datetime.now().astimezone().isoformat(),
    elapsed_seconds=time.monotonic()-start)
(root/'engineering_status.json').write_text(json.dumps(status,indent=2)+'\n')
assert result==0
receipt=json.loads((root/'witness/NATIVE_M0_RECEIPT.json').read_bytes())
assert receipt['face_residual_mode']==args.mode and receipt['actual_updates']==2
assert receipt['full_recovery_exact'] and receipt['face_architecture_saved_and_reconstructed']
assert receipt['retained_1295_states_exactly_match_normal_E0']
assert receipt['full_state_tensors']==1301
assert len(receipt['input_batch_sha256'])==2
status.update(status='complete',completed_cst=datetime.datetime.now().astimezone().isoformat(),
    elapsed_seconds=time.monotonic()-start,exit_code=0,full_recovery_exact=True)
(root/'engineering_status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status),flush=True)
