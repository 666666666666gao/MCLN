"""Observe near the measured finish estimate, then at 240-second intervals."""
import datetime
import json
from pathlib import Path
import runpy
import time

local = Path(__file__).parent
launch = json.loads((local / 'launch.json').read_bytes())
estimated = datetime.datetime.fromisoformat(launch['estimated_finish_cst'])
first_check = estimated - datetime.timedelta(seconds=180)
record = dict(status='waiting', first_check_cst=first_check.isoformat(), poll_interval_seconds=240,
              estimated_finish_cst=estimated.isoformat(), observations=0)
(local / 'completion_wait.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
time.sleep(max(0, (first_check - datetime.datetime.now().astimezone()).total_seconds()))
while True:
    witness = runpy.run_path(str(local / 'observe_authorized.py'))
    status = witness['record']['status']['status']
    record.update(status=status, observations=record['observations'] + 1,
                  last_observation=str(witness['path']), observed_cst=witness['record']['observed_cst'])
    (local / 'completion_wait.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    if status in ('complete', 'failed'):
        raise SystemExit(0 if status == 'complete' else 1)
    time.sleep(240)
