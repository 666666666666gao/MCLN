import datetime,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\n')
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
while not (root/'preflight.exit').is_file():
    process=Path('/proc/12002/cmdline')
    assert process.is_file() and (str(root)+'/preflight_controller.py').encode() in process.read_bytes()
    time.sleep(300)
assert (root/'preflight.exit').read_text().strip()=='0'
check=json.loads((root/'preflight_receipt.json').read_bytes())
assert check['status']=='pass' and check['model_states_restored'] and check['optimizer_steps']==0
assert check['spec_sha256']==hashlib.sha256((root/'spec.json').read_bytes()).hexdigest()
assert check['script_sha256']==hashlib.sha256((root/'train.py').read_bytes()).hexdigest()
while shutil.disk_usage(root).free<1759741824:
    print('REC_COMPETITION_WAIT_DISK '+str(shutil.disk_usage(root).free),flush=True)
    time.sleep(300)
runtime=Path(spec['runtime'])
environment=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
start=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),disk_free=shutil.disk_usage(root).free,required_free=1759741824,training_steps=3723,seed=2027)
(root/'training_start.json').write_text(json.dumps(start)+'\n')
print('REC_COMPETITION_TRAINING_START '+json.dumps(start),flush=True)
command=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',str(runtime/'venv/bin/python'),'-u',str(root/'train.py'),'--spec',str(root/'spec.json')]
result=subprocess.run(command,cwd=str(runtime/'PV-Ground'),env=dict(os.environ,**environment['env']))
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
