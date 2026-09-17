import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
runtime=Path(spec['runtime'])
environment=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
with (root/'check.log').open('x') as stream:
    result=subprocess.run([str(runtime/'venv/bin/python'),'-u',str(root/'check.py')],stdout=stream,stderr=subprocess.STDOUT,
        env=dict(os.environ,**dict(environment['env'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')))
(root/'check.exit').write_text(str(result.returncode)+'\n')
if result.returncode==0:
    rows=[json.loads(line.split(' ',1)[1]) for line in (root/'check.log').read_text().splitlines() if line.startswith('PVG_FIXED_MEMORY_LATEST_CPU_PASS ')]
    assert len(rows)==1 and rows[0]['status']=='pass'
    (root/'receipt.json').write_text(json.dumps(rows[0],indent=2)+'\n')
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
