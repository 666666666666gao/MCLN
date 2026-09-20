import hashlib,json,os,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
assert not (r/'formal/candidate_analysis.json').exists()
runner=r/'wait_candidate_analysis.py'
runner.write_text("import subprocess,time\nfrom pathlib import Path\nr=Path("+repr(str(r))+")\nwhile not (r/'controller.exit').exists():time.sleep(180)\ncode=int((r/'controller.exit').read_text())\nif code:raise SystemExit(code)\ncode=subprocess.call(['/root/mcln_eg3dvg_torch112_20260920_v1/venv/bin/python','-u',str(r/'analyze_candidates.py'),'--root',str(r)])\n(r/'candidate_analysis.exit').write_text(str(code)+'\\n')\nraise SystemExit(code)\n")
with (r/'candidate_analysis.log').open('xb') as f:p=subprocess.Popen(['/root/mcln_eg3dvg_torch112_20260920_v1/venv/bin/python',str(runner)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rec={'pid':p.pid,'poll_seconds':180,'wait_for':'controller.exit=0 after existing formal9508+CPUaudit','analysis_sha256':hashlib.sha256((r/'analyze_candidates.py').read_bytes()).hexdigest(),'model_forwards':0,'optimizer_steps':0}
(r/'candidate_analysis_launch.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
