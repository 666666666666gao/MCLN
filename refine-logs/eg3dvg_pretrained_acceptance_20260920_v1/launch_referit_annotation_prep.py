import json,os,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
cmd=['/root/mcln_eg3dvg_torch112_20260920_v1/venv/bin/python','-u',str(r/'prepare_referit_annotations.py')]
runner=r/'run_referit_annotation_prep.py'
runner.write_text('import subprocess\nfrom pathlib import Path\nr=Path('+repr(str(r))+')\ncode=subprocess.call('+repr(cmd)+')\n(r/"referit_annotation_prep.exit").write_text(str(code)+"\\n")\nraise SystemExit(code)\n')
with (r/'referit_annotation_prep.log').open('xb') as f:p=subprocess.Popen([cmd[0],str(runner)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rec={'pid':p.pid,'model_forwards':0,'training_steps':0,'datasets':['nr3d','sr3d'],'purpose':'CPU original author annotation parsing; leave Scan source, spec, GPU run unchanged'}
(r/'referit_annotation_prep_launch.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
