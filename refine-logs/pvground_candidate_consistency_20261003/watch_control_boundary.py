"""Wait until the estimated control boundary, then observe every 240 seconds."""
import datetime
import json
import os
from pathlib import Path
import time

local=Path(__file__).parent
due=datetime.datetime.fromisoformat('2026-10-03T11:15:00+08:00')
launch=json.loads((local/'pair_launch.json').read_bytes())
source=local/'observe_pair_once.py'
code=source.read_text(encoding='utf-8')
assert code.count("'pair_live_observation.json'")==1
started=local/'control_boundary_observer_started.json'
assert not started.exists()
activation=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),local_pid=os.getpid(),
    due_cst=due.isoformat(),interval_seconds=240,controller_pid=int(launch['process'].split()[0]),
    scope='read-only stage/liveness/capacity; no model launch/restart or cleanup')
with started.open('x',encoding='utf-8') as stream:json.dump(activation,stream,indent=2)
print(json.dumps(activation),flush=True)
time.sleep(max(0,(due-datetime.datetime.now().astimezone()).total_seconds()))
index=0
while True:
    name='control_boundary_observation_%03d.json'%index
    observed=code.replace("'pair_live_observation.json'",repr(name))
    exec(compile(observed,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
    receipt=json.loads((local/name).read_bytes())
    if receipt['status']['status']!='running' or receipt['status']['stage']!='g_control/train' or not receipt['controller_live']:
        break
    index+=1
    time.sleep(240)
finished=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),observations=index+1,
    final_observation=name,status=receipt['status'],controller_live=receipt['controller_live'])
with (local/'control_boundary_observer_finished.json').open('x',encoding='utf-8') as stream:json.dump(finished,stream,indent=2)
print(json.dumps(finished),flush=True)
