"""Deploy a reviewed loss-only change into the existing warm GPU environment."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).resolve().parent
assert not (local/'preflight_launch.json').exists()
review = json.loads((local/'SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
spec = json.loads((local/'quality_preflight_spec.json').read_bytes())
root = '/root/autodl-tmp/pvground_final_quality_20261005'
assert spec['root'] == root+'/preflight'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
    password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:
    environment = json.loads(stream.read())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest() == spec['env_spec_sha256']
python = spec['runtime']+'/venv/bin/python'
probe = '''
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);spec=json.loads(sys.argv[2]);closed=Path('/root/autodl-tmp/pvground_readback_fit_20261004')
assert not root.exists()
assert json.loads((closed/'status.json').read_bytes())['status']=='complete'
assert (closed/'controller.exit').read_text().strip()=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for name in ('base_terminal','geometry_terminal'):
    assert hashlib.sha256(Path(spec[name]).read_bytes()).hexdigest()==spec[name+'_sha256']
assert hashlib.sha256(Path(spec['source_port']).read_bytes()).hexdigest()==spec['source_port_sha256']
assert shutil.disk_usage(root.parent).free >= 512*1024**2
print(json.dumps(dict(GPU_idle=True,prior_controller_closed=True,protected_best=[5616,4506],
    data_free_bytes=shutil.disk_usage(root.parent).free,system_free_bytes=shutil.disk_usage('/').free,
    warm_environment_reused=True,packages_installed=0,weight_files_created=0)))
'''
_,stdout,stderr = client.exec_command(shlex.join([python,'-c',probe,root,json.dumps(spec)]),timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
resources = json.loads(raw)
(local/'preflight_resource_check.json').write_bytes(raw)
sftp.mkdir(root)
for name in ('preflight','quality'):
    sftp.mkdir(root+'/'+name)
payloads = {name:(local/name).read_bytes() for name in ('controller.py','EXPERIMENT_PLAN.md','SOURCE_REVIEW.md','SOURCE_REVIEW.json')}
for arm, spec_name in (('preflight','quality_preflight_spec.json'),('quality','quality_fit_spec.json')):
    payloads[arm+'/spec.json'] = (local/spec_name).read_bytes()
    for name,digest in spec['runner_files'].items():
        raw = (local/'runtime_bundle'/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        payloads[arm+'/'+name] = raw
for name,raw in payloads.items():
    with sftp.open(root+'/'+name,'wx') as stream:
        stream.write(raw)
    with sftp.open(root+'/'+name,'rb') as stream:
        assert stream.read() == raw
command = shlex.join(['flock','-n',environment['resource_limits']['gpu_lock'],python,'-B','-u',root+'/controller.py','--stage','preflight'])
inner = command+' > '+shlex.quote(root+'/preflight_controller.log')+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(root+'/preflight_controller.exit')+'; exit "$code"'
screen = 'pvg_final_quality_preflight_20261005'
_,stdout,stderr = client.exec_command(shlex.join(['screen','-dmS',screen,'bash','-c',inner]),timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
pattern = '^'+python+' -B -u '+root+'/controller.py --stage preflight$'
_,stdout,stderr = client.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),root=root,screen=screen,process=process,
    resources=resources,status='LAUNCHED_NOT_COMPLETED',formal_training_started=False,accuracy_result=False,
    weight_files_created=0,estimated_seconds=550,first_check_seconds=400,later_poll_seconds=240,
    estimate_basis='warm reconstruction and two real-batch updates, reusing actual completed R model; no whole validation pass')
(local/'preflight_launch.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with sftp.open(root+'/preflight_launch.json','wx') as stream:
    stream.write((local/'preflight_launch.json').read_bytes())
sftp.close()
client.close()
print(json.dumps(record),flush=True)
