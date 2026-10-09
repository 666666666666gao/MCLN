"""One serial recovery-only witness and one fresh ordinary trainer M0."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parent
admission=json.loads((root/'admission.json').read_bytes())
assert admission['status']=='ISOLATED_NATIVE_V2_ENGINEERING_ADMITTED'
protocol=json.loads((root/'NORMAL_NATIVE_RUN_PROTOCOL.json').read_bytes())
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==admission['env_spec_sha256']
for name,digest in admission['helpers'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
variables=dict(os.environ);variables.update(environment['env'])
variables['PYTHONPATH']=protocol['model_source']+':'+variables['PYTHONPATH']
variables.update(CUDA_VISIBLE_DEVICES='0',WORLD_SIZE='1',RANK='0',LOCAL_RANK='0',MASTER_ADDR='127.0.0.1')
record=dict(status='running',phase='targeted_native_engineering',
    started_cst=datetime.datetime.now().astimezone().isoformat(),completed_stages=[],formal_accuracy=None)
started=time.monotonic()
stages=[('previous_whole_recovery','recover_previous_whole.py',[
    '--checkpoint-identity',str(root/'PREVIOUS_ENGINEERING_CHECKPOINT.json')]),
    ('extremal_support','native_joint_preflight.py',['--mode','extremal_support'])]
for index,(stage,helper,extra) in enumerate(stages):
    variables['MASTER_PORT']=str(29717+index)
    record['stage']=stage
    (root/'engineering_status.json').write_text(json.dumps(record,indent=2)+'\n')
    argv=[admission['runtime'],'-B','-u',str(root/helper),'--protocol',
        str(root/'NORMAL_NATIVE_RUN_PROTOCOL.json'),'--output',str(root/stage)]+extra
    with (root/(stage+'.log')).open('wb') as stream:
        process=subprocess.run(argv,cwd=protocol['model_source'],env=variables,
            stdout=stream,stderr=subprocess.STDOUT)
    (root/(stage+'.exit')).write_text(str(process.returncode)+'\n')
    assert process.returncode==0
    receipt_path=root/stage/('RECOVERY_RECEIPT.json' if index==0 else 'NATIVE_M0_RECEIPT.json')
    receipt=json.loads(receipt_path.read_bytes())
    assert receipt['full_recovery_exact']
    if index==0:
        assert receipt['new_optimizer_steps']==0
    else:
        assert receipt['actual_updates']==2
    record['completed_stages'].append(stage)
record.update(status='complete',completed_cst=datetime.datetime.now().astimezone().isoformat(),
    elapsed_seconds=time.monotonic()-started,normal_training_started=False)
(root/'engineering_status.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
