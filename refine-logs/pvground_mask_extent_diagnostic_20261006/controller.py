"""One fixed read-only phase, running in the already installed PV environment."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True, choices=['preflight', 'formal'])
args = parser.parse_args()
root = Path(__file__).resolve().parent
spec = json.loads((root/'spec.json').read_bytes())
if args.mode == 'formal':
    check = json.loads((root/'preflight/CPU_SUMMARY.json').read_bytes())
    assert check['rows'] == check['preflight_actual_raw_member_rows_replayed'] == 8
    assert not any(value for item in check['CPU_stored_threshold_flips'].values() for value in item.values())
env = dict(os.environ)
env.update(json.loads((Path(spec['runtime'])/'env_spec.json').read_bytes())['env'])
status = dict(controller_pid=os.getpid(), mode=args.mode,
              started_cst=datetime.datetime.now().astimezone().isoformat(), completed=False)
path = root/(args.mode+'_status.json')
assert not path.exists()
with (root/(args.mode+'.log')).open('xb') as stream:
    child = subprocess.Popen([spec['runtime']+'/venv/bin/python','-B','-u',str(root/'run_extent_diagnostic.py'),
                              '--spec',str(root/'spec.json'),'--mode',args.mode],env=env,stdout=stream,stderr=subprocess.STDOUT)
    status['child_pid'] = child.pid
    path.write_text(json.dumps(status, indent=2)+'\n')
    code = child.wait()
status.update(completed=True, exit_code=code, finished_cst=datetime.datetime.now().astimezone().isoformat())
path.write_text(json.dumps(status, indent=2)+'\n')
(root/(args.mode+'.exit')).write_text(str(code)+'\n')
sys.exit(code)
