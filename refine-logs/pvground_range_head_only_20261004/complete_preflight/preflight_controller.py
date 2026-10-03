"""Run one reviewed stable-G engineering probe, two updates and no disk weights."""
import datetime
import json
from pathlib import Path
import subprocess
import time

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

root=Path(__file__).parent
assert not (root/'status.json').exists()
directory=root/'whole_range'
spec=json.loads((directory/'spec.json').read_bytes())
assert spec['head_only'] and spec['use_whole_range']
environment=json.loads((Path(spec['runtime'])/'env_spec.json').read_bytes())
variables=dict(environment['env'])
variables['PYTHONPATH']=str(directory)+':'+variables['PYTHONPATH']
python=spec['runtime']+'/venv/bin/python'
command=['env']+[key+'='+value for key,value in variables.items()]+[
    python,'-B','-u',str(root/'run_range_head_only.py'),'--spec',str(directory/'spec.json'),'--mode','preflight']
(root/'status.json').write_text(json.dumps(dict(status='running',phase='whole_range',started_cst=now()),indent=2)+'\n')
begin=time.monotonic()
with (directory/'preflight.log').open('x') as output:
    code=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT).returncode
(directory/'preflight.exit').write_text(str(code)+'\n')
if code!=0:
    (root/'status.json').write_text(json.dumps(dict(status='failed',phase='whole_range',exit_code=code,finished_cst=now()),indent=2)+'\n')
    raise SystemExit(code)
receipt=json.loads((directory/'preflight.json').read_bytes())
assert receipt['status']=='pass' and receipt['batch_size']==8 and receipt['optimizer_steps']==2
assert receipt['head_only'] and receipt['original_g_state_unchanged']
assert receipt['weight_files_created']==0
record=dict(status='complete',finished_cst=now(),seconds=time.monotonic()-begin,
    exit_code=code,batch_size=8,optimizer_steps=2,original_g_state_unchanged=True,
    formal_training_started=False,weights_created=0)
(root/'status.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
