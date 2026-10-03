"""Observe the original pair near its estimated end, then every 240 seconds."""
import datetime
import json
import os
from pathlib import Path
import time


local = Path(__file__).parent
due = datetime.datetime.fromisoformat('2026-10-03T14:45:00+08:00')
boundary = json.loads((local/'control_boundary_observer_finished.json').read_bytes())
assert boundary['status']['stage'] == 'g_consistent/train'
assert boundary['controller_live']
assert not (local/'pair_completion_observer_started.json').exists()
source = local/'observe_pair_once.py'
code = source.read_text(encoding='utf-8')
assert code.count("'pair_live_observation.json'") == 1
started = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
               local_pid=os.getpid(), due_cst=due.isoformat(), interval_seconds=240,
               controller_pid=320934, prior_observer_finished=True,
               estimate_basis='control stage10422.462s; two formal evaluations about31min total',
               scope='read-only original pair stage/liveness/capacity; no launch/restart/cleanup')
with (local/'pair_completion_observer_started.json').open('x', encoding='utf-8') as stream:
    json.dump(started, stream, indent=2)
print(json.dumps(started), flush=True)
time.sleep(max(0, (due-datetime.datetime.now().astimezone()).total_seconds()))
index = 0
while True:
    name = 'pair_completion_observation_%03d.json' % index
    observed = code.replace("'pair_live_observation.json'", repr(name))
    exec(compile(observed, str(source), 'exec'), {'__file__':str(source), '__name__':'__main__'})
    receipt = json.loads((local/name).read_bytes())
    if receipt['status']['status'] != 'running' or not receipt['controller_live']:
        break
    index += 1
    time.sleep(240)
finished = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
                observations=index+1, final_observation=name,
                status=receipt['status'], controller_live=receipt['controller_live'])
with (local/'pair_completion_observer_finished.json').open('x', encoding='utf-8') as stream:
    json.dump(finished, stream, indent=2)
print(json.dumps(finished), flush=True)
