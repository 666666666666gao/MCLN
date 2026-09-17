import json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\n')
spec=json.loads((root/'spec.json').read_bytes());runtime=Path(spec['runtime'])
env=dict(os.environ,**json.loads((runtime/'env_spec.json').read_bytes())['env'])
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'run.log').open('w') as log:
 result=subprocess.run([str(runtime/'venv/bin/python'),'-u',str(root/'prepare.py'),'--spec',str(root/'spec.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
