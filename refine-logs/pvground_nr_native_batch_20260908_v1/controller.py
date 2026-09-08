import json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\n')
spec=json.loads((root/'spec.json').read_bytes());runtime=Path(spec['runtime'])
env=os.environ.copy();env.update(json.loads((runtime/'env_spec.json').read_bytes())['env'])
env['CUDA_VISIBLE_DEVICES']=''
with (root/'run.log').open('xb') as log:
    result=subprocess.run([str(runtime/'venv/bin/python'),'-u',str(root/'prepare.py'),'--spec',str(root/'spec.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
