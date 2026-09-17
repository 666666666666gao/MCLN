import json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\n')
spec=json.loads((root/'spec.json').read_bytes());runtime=Path(spec['runtime'])
env=dict(os.environ,**json.loads((runtime/'env_spec.json').read_bytes())['env'])
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(root))
with (root/'tests.log').open('w') as log:
 test=subprocess.run([str(runtime/'venv/bin/python'),str(root/'tests/test_pvground_referit_endpoint_audit.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'tests.exit').write_text(str(test.returncode)+'\n')
if test.returncode!=0:
 (root/'controller.exit').write_text(str(test.returncode)+'\n')
 raise SystemExit(test.returncode)
with (root/'run.log').open('w') as log:
 result=subprocess.run([str(runtime/'venv/bin/python'),'-u',str(root/'prepare_pvground_referit_formal_inputs.py'),'--spec',str(root/'spec.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
