import subprocess,time
from pathlib import Path
r=Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
while not (r/'referit_annotation_prep.exit').exists():time.sleep(180)
assert int((r/'referit_annotation_prep.exit').read_text())==0
code=subprocess.call(['/root/mcln_eg3dvg_torch112_20260920_v1/venv/bin/python','-u',str(r/'preflight_referit_dataset.py')])
(r/'referit_dataset_preflight.exit').write_text(str(code)+'\n')
raise SystemExit(code)
