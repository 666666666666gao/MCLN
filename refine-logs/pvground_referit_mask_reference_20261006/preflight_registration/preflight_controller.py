"""Run each registered mixed-row real-model preflight once, sequentially on GPU0."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
args=parser.parse_args();root=args.root
assert str(root)=='/root/autodl-tmp/pvground_referit_mask_reference_20261006'
status=dict(status='running',started_cst=datetime.datetime.now().astimezone().isoformat(),completed=[])
start=time.perf_counter()
def record():
    (root/'preflight_status.json').write_text(json.dumps(status,indent=2)+'\n')
record()
for dataset in ('nr3d','sr3d'):
    for mode in ('native','fused_mask'):
        name=dataset+'_'+mode;directory=root/name
        assert not (directory/'preflight.exit').exists() and not (directory/'receipt.json').exists()
        with (directory/'run.log').open('xb') as stream:
            process=subprocess.Popen([sys.executable,'-B','-u',str(root/'referit_model_preflight.py'),
                '--spec',str(directory/'spec.json')],stdout=stream,stderr=subprocess.STDOUT)
            status.update(dataset=dataset,reference_mode=mode,child_pid=process.pid);record()
            exitcode=process.wait()
        (directory/'preflight.exit').write_text(str(exitcode)+'\n')
        assert exitcode==0,name
        receipt=json.loads((directory/'receipt.json').read_bytes())
        assert receipt['status']=='pass' and receipt['actual_optimizer_steps']==2 and receipt['formal_rows']==0
        assert receipt['scanrefer_weights_loaded'] is False
        status['completed'].append(name);record()
status.update(status='complete',finished_cst=datetime.datetime.now().astimezone().isoformat(),
    elapsed_seconds=time.perf_counter()-start,formal_rows=0,new_training_checkpoints=0)
record();print(json.dumps(status),flush=True)
