import json,os,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_pretrained_forward_20260908_v1')
spec=json.loads((root/'spec.json').read_bytes())
runtime=Path(spec['runtime'])
env=json.loads((runtime/'env_spec.json').read_bytes())['env']
command=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',str(runtime/'venv/bin/python'),'-u',str(root/'verify_forward.py'),'--runtime',str(runtime),'--fixtures',spec['fixtures'],'--output',str(root/'results')]
result=subprocess.run(command,cwd=str(runtime/'PV-Ground'),env=dict(os.environ,**env))
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
