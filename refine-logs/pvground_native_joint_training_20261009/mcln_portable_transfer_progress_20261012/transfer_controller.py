"""One explicitly admitted CPU/sanity/full MCLN transfer stage."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
admission = json.loads((root/'admission.json').read_bytes())
stage = admission['stage']
assert stage in ('cpu','sanity','full')
source = admission['source']
environment = json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
variables = dict(os.environ)
variables.update(environment['env'])
variables['PYTHONPATH'] = source+':'+source+'/pointnet2:'+variables['PYTHONPATH']
variables.update(CUDA_VISIBLE_DEVICES='' if stage=='cpu' else '0', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
entry = 'check_mcln_transfer_cpu.py' if stage=='cpu' else 'evaluate_mcln_portable_pair.py'
command = [runtime,'-B','-u',str(Path(source)/entry),'--source',source,
           '--spec',admission['spec'],'--output',str(root/'CPU_RESULT.json' if stage=='cpu' else root/'replay')]
if stage!='cpu': command += ['--max-batches','1' if stage=='sanity' else '0']
record = dict(status='running',stage=stage,started_cst=datetime.datetime.now().astimezone().isoformat(),
              source=source,optimizer_updates=0,weights_written=0)
started = time.monotonic()
with (root/'stage.log').open('wb') as stream:
    child = subprocess.Popen(command,cwd=source,env=variables,stdout=stream,stderr=subprocess.STDOUT)
    record.update(child_pid=child.pid,child_argv=command)
    (root/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    exit_code = child.wait()
(root/'child.exit').write_text(str(exit_code)+'\n')
record.update(status='complete' if exit_code==0 else 'failed',exit_code=exit_code,
              completed_cst=datetime.datetime.now().astimezone().isoformat(),elapsed_seconds=time.monotonic()-started)
if exit_code==0:
    result_path = root/'CPU_RESULT.json' if stage=='cpu' else root/'replay/RESULT.json'
    record['result'] = json.loads(result_path.read_bytes())
(root/'status.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
assert exit_code==0
