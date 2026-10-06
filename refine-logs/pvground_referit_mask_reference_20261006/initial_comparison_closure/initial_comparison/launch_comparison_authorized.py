"""Run the reviewed three-forward measurement once, after the failed R1 is closed."""
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

local=Path(__file__).resolve().parent
assert not (local/'launch.json').exists()
review=json.loads((local/'SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
previous=json.loads((local.parent/'preflight_wait.json').read_bytes())
assert previous['observer_closed'] and previous['terminal']['controller_exit']==1
assert not previous['terminal']['controller_alive'] and not previous['terminal']['receipts']
spec=json.loads((local/'spec.json').read_bytes())
root='/root/autodl-tmp/pvground_referit_initial_comparison_20261006'
assert spec['root']==root+'/nr3d_native'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:environment=json.loads(stream.read())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',', ':')).encode()).hexdigest()==spec['env_spec_sha256']
create='''import json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);assert not root.exists()
old=Path('/root/autodl-tmp/pvground_referit_mask_reference_20261006');assert (old/'preflight_controller.exit').read_text().strip()=='1'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip()
assert len(gpu.splitlines())==1 and int(gpu)<500,gpu
assert shutil.disk_usage(root.parent).free>128*1024*1024
root.mkdir();(root/'nr3d_native').mkdir()
print(json.dumps(dict(gpu_memory_before_MiB=int(gpu),free_bytes=shutil.disk_usage(root).free)))
'''
_,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',create,root]),timeout=30)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
resource=json.loads(raw)
for name,digest in spec['files'].items():
    raw=(local/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
    with sftp.open(root+'/'+name,'wx') as stream:stream.write(raw)
    with sftp.open(root+'/'+name,'rb') as stream:assert stream.read()==raw
for name,remote in [('spec.json','nr3d_native/spec.json'),('SOURCE_REVIEW.json','SOURCE_REVIEW.json'),('SOURCE_REVIEW.md','SOURCE_REVIEW.md')]:
    raw=(local/name).read_bytes()
    with sftp.open(root+'/'+remote,'wx') as stream:stream.write(raw)
    with sftp.open(root+'/'+remote,'rb') as stream:assert stream.read()==raw
command=shlex.join(['env']+[key+'='+value for key,value in environment['env'].items()]
    +[spec['runtime']+'/venv/bin/python','-B','-u',root+'/compare_initial.py','--spec',root+'/nr3d_native/spec.json'])
shell=command+' > '+shlex.quote(root+'/run.log')+' 2>&1; status=$?; printf "%s\\n" "$status" > '+shlex.quote(root+'/controller.exit')+'; exit "$status"'
launch='''import datetime,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);p=subprocess.Popen(['bash','-c',sys.argv[2]],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),controller_pid=p.pid,root=str(root));assert p.poll() is None
(root/'launch.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
_,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',launch,root,shell]),timeout=30)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record=json.loads(raw);record.update(resource=resource,zero_update_only=True,actual_result_pending=True)
(local/'launch.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
sftp.close();client.close();print(json.dumps(record),flush=True)
