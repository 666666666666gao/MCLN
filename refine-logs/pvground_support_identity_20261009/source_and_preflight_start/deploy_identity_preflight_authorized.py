"""Launch the reviewed two-step sanity job once in the unchanged warm runtime."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

root = Path(__file__).resolve().parent
assert not (root/'PREFLIGHT_LAUNCH.json').exists()
spec = json.loads((root/'preflight_spec.json').read_bytes())
review = json.loads((root/'SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
parent = json.loads((root.parent/'pvground_compressed_geometry_support_20261008/pair_spec.json').read_bytes())
for name,digest in spec['new_files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
runtime = parent['runtime']
files = list(spec['new_files'])+['preflight_spec.json','EXPERIMENT_PLAN.md','SOURCE_REVIEW.json','SOURCE_REVIEW.md']
client = paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
with sftp.open(runtime+'/env_spec.json','rb') as stream:env = json.loads(stream.read())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest() == parent['env_spec_sha256']
check = """import hashlib,json,shutil,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root'])
assert root==Path('/root/autodl-tmp/pvground_support_identity_20261009') and not root.exists()
assert (Path('/root/autodl-tmp/pvground_pretrained_protocol_20261009/preflight_r3/formal.exit').read_text().strip()=='0')
assert not Path('/proc/958304').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip().splitlines()
assert len(rows)==1
index,name,used,total=[x.strip() for x in rows[0].split(',')]
assert int(index)==0 and 'A100' in name and int(used)<500 and 40000<=int(total)<=45000
assert shutil.disk_usage('/root/autodl-tmp').free>134217728
assert hashlib.sha256(Path(b['support_terminal']).read_bytes()).hexdigest()==b['support_terminal_sha256']
assert hashlib.sha256(Path(b['parent_spec']).read_bytes()).hexdigest()==b['parent_spec_sha256']
root.mkdir()
print(json.dumps(dict(gpu=dict(index=int(index),name=name,used_mib=int(used),total_mib=int(total)),data_free=shutil.disk_usage(root).free,system_free=shutil.disk_usage('/').free,environment_rebuilt=False,new_packages=0,prior_parent_evaluation_closed=True)))
"""
stdin,stdout,stderr = client.exec_command(shlex.join([runtime+'/venv/bin/python','-B','-c',check]),timeout=60)
stdin.write(json.dumps(spec));stdin.channel.shutdown_write()
resource_raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
(root/'RESOURCE_CHECK.json').write_bytes(resource_raw)
for name in files:
    data=(root/name).read_bytes()
    with sftp.open(spec['root']+'/'+name,'wx') as stream:stream.write(data)
    with sftp.open(spec['root']+'/'+name,'rb') as stream:assert stream.read()==data
command=shlex.join(['flock','-n',env['resource_limits']['gpu_lock'],'env']+[k+'='+v for k,v in env['env'].items()]+[runtime+'/venv/bin/python','-B','-u',spec['root']+'/run_identity_preflight.py','--spec',spec['root']+'/preflight_spec.json'])
body=command+' >'+shlex.quote(spec['root']+'/preflight.log')+' 2>&1\nresult=$?\nprintf "%s\\n" "$result" >'+shlex.quote(spec['root']+'/preflight.exit')+'\nexit "$result"\n'
with sftp.open(spec['root']+'/run_preflight.sh','wx') as stream:stream.write(body.encode())
sftp.close()
screen='pvg_support_identity_preflight_20261009'
_,stdout,stderr=client.exec_command(shlex.join(['screen','-dmS',screen,'bash',spec['root']+'/run_preflight.sh']),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
pattern='^'+runtime+'/venv/bin/python -B -u '+spec['root']+'/run_identity_preflight.py --spec '+spec['root']+'/preflight_spec.json$'
_,stdout,stderr=client.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
process=stdout.read().decode().strip()
assert stdout.channel.recv_exit_status()==0 and len(process.splitlines())==1,stderr.read().decode()
client.close()
stamp=datetime.datetime.now().astimezone()
record=dict(status='ACTUAL_B8_IDENTITY_PREFLIGHT_STARTED_NOT_PASSED',time_cst=stamp.isoformat(),root=spec['root'],process=process,pid=int(process.split()[0]),screen=screen,resources=json.loads(resource_raw),optimizer_steps_planned_per_arm=2,accuracy_result=False,formal_training_started=False,new_weight_files_expected=0,first_observation_cst=(stamp+datetime.timedelta(seconds=spec['first_check_seconds'])).isoformat(),estimated_finish_cst=(stamp+datetime.timedelta(seconds=spec['estimated_seconds'])).isoformat(),later_poll_seconds=240,estimate_basis='Prior warm paired B8 sanity source/data load and full CPU rebuild typically several minutes; current estimate8min, first7min. This is not a completion claim.')
(root/'PREFLIGHT_LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
