import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_sourcequery_v1')
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
runtime=Path(spec['runtime'])
env=json.loads((runtime/'env_spec.json').read_bytes())['env']
command=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',str(runtime/'venv/bin/python'),'-u',str(root/'train.py'),'--spec',str(root/'spec.json')]
result=subprocess.run(command,cwd=str(runtime/'PV-Ground'),env=dict(os.environ,**env))
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
