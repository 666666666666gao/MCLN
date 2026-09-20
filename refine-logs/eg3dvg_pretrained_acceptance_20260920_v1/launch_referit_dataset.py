import json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
runner=r/'wait_referit_dataset.py'
runner.write_text("import subprocess,time\nfrom pathlib import Path\nr=Path("+repr(str(r))+")\nwhile not (r/'referit_annotation_prep.exit').exists():time.sleep(180)\nassert int((r/'referit_annotation_prep.exit').read_text())==0\ncode=subprocess.call(['/root/mcln_eg3dvg_torch112_20260920_v1/venv/bin/python','-u',str(r/'preflight_referit_dataset.py')])\n(r/'referit_dataset_preflight.exit').write_text(str(code)+'\\n')\nraise SystemExit(code)\n")
with (r/'referit_dataset_preflight.log').open('xb') as f:p=subprocess.Popen(['/root/mcln_eg3dvg_torch112_20260920_v1/venv/bin/python',str(runner)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'model_forwards':0,'training_steps':0,'purpose':'full CPU Dataset input check after original annotation parsing'}))
