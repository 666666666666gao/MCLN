"""Install reviewed isolated ReferIt preflight source and launch the single controller."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local=Path(__file__).resolve().parent
assert not (local/'preflight_launch.json').exists()
review=json.loads((local/'SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
plan=json.loads((local/'preflight_spec.json').read_bytes())
sources={name:(local/name).read_bytes() for name in plan['files']}
assert all(hashlib.sha256(raw).hexdigest()==plan['files'][name] for name,raw in sources.items())
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(plan['runtime']+'/env_spec.json','rb') as stream:environment=json.loads(stream.read())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',', ':')).encode()).hexdigest()==plan['env_spec_sha256']
root=plan['remote_root'];assert root=='/root/autodl-tmp/pvground_referit_mask_reference_20261006'
code='''import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);assert not root.exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip()
assert len(gpu.splitlines())==1 and int(gpu)<500,gpu
assert shutil.disk_usage('/root/autodl-tmp').free>128*1024*1024
root.mkdir()
for name in ('nr3d_native','nr3d_fused_mask','sr3d_native','sr3d_fused_mask'):(root/name).mkdir()
print(json.dumps(dict(gpu_memory_before_MiB=int(gpu),free_bytes=shutil.disk_usage('/root/autodl-tmp').free)))
'''
_,stdout,stderr=client.exec_command(shlex.join([plan['runtime']+'/venv/bin/python','-B','-c',code,root]),timeout=30)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
resource=json.loads(raw)
payloads={**sources,'preflight_spec.json':(local/'preflight_spec.json').read_bytes(),
    'SOURCE_REVIEW.json':(local/'SOURCE_REVIEW.json').read_bytes(),
    'SOURCE_REVIEW.md':(local/'SOURCE_REVIEW.md').read_bytes()}
for name,raw in payloads.items():
    with sftp.open(root+'/'+name,'wx') as stream:stream.write(raw)
    with sftp.open(root+'/'+name,'rb') as stream:assert stream.read()==raw
for name,spec in plan['runs'].items():
    raw=(json.dumps(spec,indent=2)+'\n').encode()
    with sftp.open(root+'/'+name+'/spec.json','wx') as stream:stream.write(raw)
    with sftp.open(root+'/'+name+'/spec.json','rb') as stream:assert stream.read()==raw
    directory=local/name;directory.mkdir();(directory/'spec.json').write_bytes(raw)
command=shlex.join(['env']+[key+'='+value for key,value in environment['env'].items()]
    +[plan['runtime']+'/venv/bin/python','-B','-u',root+'/preflight_controller.py','--root',root])
shell=command+' > '+shlex.quote(root+'/preflight_controller.log')+' 2>&1; status=$?; printf "%s\\n" "$status" > '+shlex.quote(root+'/preflight_controller.exit')+'; exit "$status"'
launch='''import datetime,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);p=subprocess.Popen(['bash','-c',sys.argv[2]],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),controller_pid=p.pid,root=str(root))
(root/'preflight_launch.json').write_text(json.dumps(record,indent=2)+'\\n')
assert p.poll() is None
print(json.dumps(record))
'''
_,stdout,stderr=client.exec_command(shlex.join([plan['runtime']+'/venv/bin/python','-B','-c',launch,root,shell]),timeout=30)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record=json.loads(raw);record.update(resource=resource,source_review='SOURCE_ONLY',formal_rows=0,
    actual_preflight_terminal_pending=True,protection='existing best Scan weights not loaded or modified')
(local/'preflight_launch.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
sftp.close();client.close();print(json.dumps(record),flush=True)
