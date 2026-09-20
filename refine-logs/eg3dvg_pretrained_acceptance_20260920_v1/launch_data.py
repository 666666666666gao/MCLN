import json,os,shlex,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
env=dict(os.environ)
with (r/'data.log').open('xb') as f:
 p=subprocess.Popen(['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-u',str(r/'prepare_data.py')],cwd=str(r/'source'),env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rec={'pid':p.pid,'script':str(r/'prepare_data.py'),'log':str(r/'data.log')}
(r/'data_launch.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
