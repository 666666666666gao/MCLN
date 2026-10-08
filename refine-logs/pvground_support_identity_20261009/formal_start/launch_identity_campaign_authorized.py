"""Seal/deploy one reviewed paired identity campaign in the unchanged warm env."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import time
import paramiko


local = Path(__file__).resolve().parent
assert not (local / 'CAMPAIGN_LAUNCH.json').exists()
spec = json.loads((local / 'campaign_spec.json').read_bytes())
review = json.loads((local / 'FORMAL_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
assert hashlib.sha256((local / 'actual/preflight.json').read_bytes()).hexdigest() == spec['preflight_sha256']
assert json.loads((local / 'actual/preflight.json').read_bytes())['status'] == 'PASS_ACTUAL_B8_TWO_STEP_SUPPORT_IDENTITY_PREFLIGHT'
parent = json.loads((local.parent / 'pvground_compressed_geometry_support_20261008/pair_spec.json').read_bytes())
files = ('run_identity_campaign.py', 'campaign_spec.json', 'FORMAL_SOURCE_REVIEW.json', 'FORMAL_SOURCE_REVIEW.md', 'FORMAL_SOURCE_REVIEW_REQUEST.txt', 'FORMAL_PLAN.md')
assert all(hashlib.sha256((local / name).read_bytes()).hexdigest() == digest for name, digest in spec['files'].items())
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
check = """import hashlib,json,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);r=Path(b['root'])
assert r==Path('/root/autodl-tmp/pvground_support_identity_20261009')
assert int((r/'preflight.exit').read_text())==0
assert hashlib.sha256((r/'preflight.json').read_bytes()).hexdigest()==b['preflight_sha256']
assert not (r/'campaign').exists() and not (r/'campaign.exit').exists()
assert all(hashlib.sha256((r/name).read_bytes()).hexdigest()==digest for name,digest in b['already_staged'].items())
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip().split(',')
assert int(gpu[0])==0 and int(gpu[2])<500
def free(path):
    import os
    s=os.statvfs(path);return s.f_bavail*s.f_frsize
data_free=free('/root/autodl-tmp');system_free=free('/')
assert data_free>400000000
print(json.dumps(dict(gpu=dict(index=int(gpu[0]),name=gpu[1].strip(),used_mib=int(gpu[2]),total_mib=int(gpu[3])),data_free=data_free,system_free=system_free,environment_rebuilt=False,new_packages=0,preflight_closed_exit0=True)))
"""
stdin, stdout, stderr = client.exec_command(shlex.join([parent['runtime'] + '/venv/bin/python', '-B', '-c', check]), timeout=60)
stdin.write(json.dumps(dict(root=spec['root'], preflight_sha256=spec['preflight_sha256'], already_staged={'support_identity_readout.py':spec['files']['support_identity_readout.py']})))
stdin.channel.shutdown_write()
resources = json.loads(stdout.read())
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
(local / 'CAMPAIGN_RESOURCE_CHECK.json').write_text(json.dumps(resources, indent=2) + '\n', encoding='utf-8')
sftp = client.open_sftp()
for name in files:
    path = spec['root'] + '/' + name
    with sftp.open(path, 'wxb') as stream:
        stream.write((local / name).read_bytes())
sftp.close()
launch_code = """import hashlib,json,re,shlex,subprocess,sys,time
from pathlib import Path
b=json.load(sys.stdin);r=Path(b['root']);spec=json.loads((r/'campaign_spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name
review=json.loads((r/'FORMAL_SOURCE_REVIEW.json').read_bytes());assert not review['blocking_findings'] and review['execution_scope']=='SOURCE_ONLY'
parent=json.loads(Path(spec['parent_spec']).read_bytes());runtime=parent['runtime'];env=json.loads((Path(runtime)/'env_spec.json').read_bytes())
variables=dict(env['env']);variables['PYTHONPATH']=str(r)+':'+parent['root']+':'+parent['helper_root']+':'+variables['PYTHONPATH']
command=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock','env']+[key+'='+value for key,value in variables.items()]+[runtime+'/venv/bin/python','-B','-u',str(r/'run_identity_campaign.py'),'--spec',str(r/'campaign_spec.json')]
shell=' '.join(shlex.quote(x) for x in command)+' > '+shlex.quote(str(r/'campaign.log'))+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(str(r/'campaign.exit'))
subprocess.check_call(['screen','-dmS',b['screen'],'bash','-lc',shell]);time.sleep(2)
pattern='^'+re.escape(runtime+'/venv/bin/python -B -u '+str(r/'run_identity_campaign.py')+' --spec '+str(r/'campaign_spec.json'))+'$'
pids=subprocess.check_output(['pgrep','-f',pattern]).decode().split();assert len(pids)==1,pids
print(json.dumps(dict(pid=int(pids[0]),screen=b['screen'],command=command,status='ACTUAL_PAIRED_IDENTITY_CAMPAIGN_STARTED_NOT_COMPLETE')))
"""
stdin, stdout, stderr = client.exec_command(shlex.join([parent['runtime'] + '/venv/bin/python', '-B', '-c', launch_code]), timeout=60)
stdin.write(json.dumps(dict(root=spec['root'], screen='pvg_support_identity_fit_20261009')))
stdin.channel.shutdown_write()
result = json.loads(stdout.read())
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
client.close()
now = datetime.datetime.now().astimezone()
estimate = now + datetime.timedelta(seconds=spec['estimated_seconds'])
first = estimate - datetime.timedelta(seconds=300)
result.update(root=spec['root'], time_cst=now.isoformat(), first_observation_cst=first.isoformat(), estimated_finish_cst=estimate.isoformat(), later_poll_seconds=240, optimizer_steps_planned_per_arm=3723, fit_examples_planned_per_arm=29778, seed=2027, multiseed=False, historical_best_hits=[5599,4859], target_hits=[5658,4850], resources=resources, preflight_updates_carried=False, formal_accuracy_result=False)
(local / 'CAMPAIGN_LAUNCH.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result), flush=True)
