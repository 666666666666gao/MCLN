import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_training_interface_20260908_v1')
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
runtime=Path(spec['runtime'])
env=json.loads((runtime/'env_spec.json').read_bytes())['env']
export=['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'export_fixtures.py'),'--manifest',spec['manifest'],'--output',str(root/'fixtures'),'--training-labels','--reference-fixtures',spec['reference_fixtures']]
result=subprocess.run(export,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'))
(root/'export.exit').write_text(str(result.returncode)+'\n')
if result.returncode:
    (root/'controller.exit').write_text(str(result.returncode)+'\n')
    raise SystemExit(result.returncode)
check=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',str(runtime/'venv/bin/python'),'-u',str(root/'check_training.py'),'--runtime',str(runtime),'--fixtures',str(root/'fixtures'),'--output',str(root/'results')]
result=subprocess.run(check,cwd=str(runtime/'PV-Ground'),env=dict(os.environ,**env))
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
