"""One scheduled read-only observer, first near the estimated end, then240seconds."""
import datetime
import json
from pathlib import Path
import runpy
import time

local=Path(__file__).parent
assert not (local/'preflight_wait.json').exists()
launch=json.loads((local/'preflight_launch.json').read_bytes())
start=datetime.datetime.fromisoformat(launch['time_cst'])
first=start+datetime.timedelta(seconds=launch['first_check_seconds'])
record=dict(status='waiting',first_check_cst=first.isoformat(),observations=0,
    estimated_finish_cst=(start+datetime.timedelta(seconds=launch['estimate_seconds'])).isoformat(),
    poll_interval_seconds=launch['later_poll_seconds'])
(local/'preflight_wait.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
time.sleep(max(0,(first-datetime.datetime.now().astimezone()).total_seconds()))
while True:
    witness=runpy.run_path(str(local/'observe_preflight_authorized.py'))
    observation=witness['record']
    record.update(status=observation['status']['status'],observations=record['observations']+1,
        last_observation=witness['name'],observed_cst=observation['observed_cst'])
    (local/'preflight_wait.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    if record['status'] in ('complete','failed') and not observation['controller_alive']:
        assert observation['controller_exit'] is not None
        raise SystemExit(observation['controller_exit'])
    time.sleep(launch['later_poll_seconds'])
