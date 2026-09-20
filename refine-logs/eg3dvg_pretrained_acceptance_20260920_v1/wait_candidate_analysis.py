import subprocess,time
from pathlib import Path
r=Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
while not (r/'controller.exit').exists():time.sleep(180)
code=int((r/'controller.exit').read_text())
if code:raise SystemExit(code)
code=subprocess.call(['/root/mcln_eg3dvg_torch112_20260920_v1/venv/bin/python','-u',str(r/'analyze_candidates.py'),'--root',str(r)])
(r/'candidate_analysis.exit').write_text(str(code)+'\n')
raise SystemExit(code)
