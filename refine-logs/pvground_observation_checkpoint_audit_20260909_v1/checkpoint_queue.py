import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_observation_checkpoint_audit_20260909_v1')
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
training=Path(spec['training_root'])
time.sleep(max(0,datetime.datetime.fromisoformat(spec['first_check_cst']).timestamp()-time.time()))
while not (training/'latest.pth').is_file():
    original=Path('/proc')/str(spec['training_pid'])/'cmdline'
    assert original.is_file() and (str(training)+'/controller.py').encode() in original.read_bytes()
    time.sleep(300)
runtime=Path(spec['runtime'])
environment=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
command=[str(runtime/'venv/bin/python'),'-u',str(root/'check.py')]
with (root/'check.log').open('x') as stream:
    result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
        env=dict(os.environ,**dict(environment['env'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')))
(root/'check.exit').write_text(str(result.returncode)+'\n')
assert result.returncode==0
lines=(root/'check.log').read_text().splitlines()
record=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('PVG_OBSERVATION_LATEST_CPU_PASS ')]
assert len(record)==1 and record[0]['status']=='pass'
(root/'receipt.json').write_text(json.dumps(record[0],indent=2)+'\n')
print(json.dumps(record[0]),flush=True)
