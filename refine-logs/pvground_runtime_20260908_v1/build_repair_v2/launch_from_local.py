import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo / 'refine-logs/pvground_runtime_20260908_v1/build_repair_v2'
temporary = Path('C:/Users/gb/.codex/tmp')
spec_data = (repo / 'configs/pvground_runtime_env_20260908.json').read_bytes()
spec = json.loads(spec_data)
root = spec['root']
remote_repair = root + '/build_repair_v2'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
check = "from pathlib import Path;import shutil;root=Path("+repr(root)+");assert (root/'controller.exit').read_text().strip()=='1';assert not (root/'build_repair_v2').exists();assert shutil.disk_usage(str(root)).free>2500000000"
_, stdout, stderr = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(check), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
for name in ('run.log', 'controller.exit', 'base_packages_before.json', 'source_port.json'):
    sftp.get(root+'/'+name, str(archive / ('original_'+name)))
for name in ('pvg_cuda_toolkit_probe_20260908.json','pvg_build_packaging_probe_20260908.json'):
    (archive/name).write_bytes((temporary/name).read_bytes())
sftp.mkdir(remote_repair)
with sftp.open(root+'/env_spec.json','rb') as stream:
    previous=stream.read()
assert hashlib.sha256(json.dumps(json.loads(previous),sort_keys=True,separators=(',',':')).encode()).hexdigest()=='c73eed09e0e952c981d616ea55ff5da683c99ea7a0ebd21f8f8740896a501ae4'
with sftp.open(remote_repair+'/previous_env_spec.json','wx') as stream:
    stream.write(previous)
with sftp.open(root+'/env_spec.json','wb') as stream:
    stream.write(spec_data)
for name in ('package_source.json','env_spec.json','spec_sha256.txt'):
    sftp.put(str(archive/name), remote_repair+'/'+name)
package=json.loads((archive/'package_source.json').read_bytes())
sftp.put(str(temporary/package['filename']),remote_repair+'/'+package['filename'])
source=(temporary/'resume_pvg_compile_v2_20260908.py').read_bytes()
(archive/'resume_compile_v2.py').write_bytes(source)
with sftp.open(root+'/resume_compile_v2.py','wx') as stream:
    stream.write(source)
controller='#!/bin/bash\ncd '+shlex.quote(root)+'\nflock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/resume_compile_v2.py')+'\nresult=$?\nprintf "%s\\n" "$result" > '+shlex.quote(remote_repair+'/controller.exit')+'\nexit "$result"\n'
(archive/'controller.sh').write_bytes(controller.encode())
with sftp.open(remote_repair+'/controller.sh','wx') as stream:
    stream.write(controller.encode())
inner='exec bash '+shlex.quote(remote_repair+'/controller.sh')+' > '+shlex.quote(remote_repair+'/run.log')+' 2>&1'
_, stdout, stderr=client.exec_command('screen -dmS mcln_pvg_compile_v2 bash -c '+shlex.quote(inner),timeout=30)
assert stdout.channel.recv_exit_status()==0, stderr.read().decode()
_, stdout, stderr=client.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/resume_compile_v2.py$'),timeout=30)
process=stdout.read().decode()
assert stdout.channel.recv_exit_status()==0 and process.strip(),stderr.read().decode()
receipt={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'process':process.strip(),'screen':'mcln_pvg_compile_v2','spec_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'estimated_seconds':[300,600],'training_steps':0,'formal_rows':0}
raw=(json.dumps(receipt,indent=2)+'\n').encode()
(archive/'launch.json').write_bytes(raw)
with sftp.open(remote_repair+'/launch.json','wx') as stream:
    stream.write(raw)
(archive/'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
sftp.close()
client.close()
print('PVG_REPAIR_LAUNCHED '+json.dumps(receipt),flush=True)
