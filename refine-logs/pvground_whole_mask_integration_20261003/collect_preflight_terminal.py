"""Collect actual closed engineering-preflight text evidence, with no model call."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).parent
output = local / 'complete'
assert not output.exists()
launch = json.loads((local / 'preflight_launch.json').read_bytes())
spec = json.loads((local / 'whole_range_spec.json').read_bytes())
probe = r'''
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);pid=int(sys.argv[2])
assert root==Path('/root/autodl-tmp/pvground_whole_mask_preflight_20261003')
assert not Path('/proc/%d'%pid).exists()
status=json.loads((root/'status.json').read_bytes())
assert status['status'] in ('complete','failed')
code=int((root/'controller.exit').read_text())
assert (code==0)==(status['status']=='complete')
files={}
for path in sorted(root.rglob('*')):
    if path.is_file():
        assert path.suffix in ('.py','.md','.json','.log','.exit'),str(path)
        raw=path.read_bytes()
        files[path.relative_to(root).as_posix()]=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
env=Path(sys.argv[3]);parent=Path(sys.argv[4])
print(json.dumps(dict(status=status,controller_exit=code,controller_alive=False,files=files,
    directory_free_bytes=shutil.disk_usage(root).free,gpu_compute=gpu,
    environment_spec_sha256=hashlib.sha256(json.dumps(json.loads(env.read_bytes()),sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    original_g_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),weight_files_created=0)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
command = shlex.join([spec['runtime'] + '/venv/bin/python', '-c', probe, launch['root'],
    launch['process'].split()[0], spec['runtime'] + '/env_spec.json', spec['base_terminal']])
_, stdout, stderr = client.exec_command(command, timeout=180)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
assert record['environment_spec_sha256'] == '966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
assert record['original_g_sha256'] == spec['base_terminal_sha256']
output.mkdir()
sftp = client.open_sftp()
for relative, item in record['files'].items():
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(item['path'], str(path))
    content = path.read_bytes()
    assert len(content) == item['bytes'] and hashlib.sha256(content).hexdigest() == item['sha256']
sftp.close()
client.close()
record.update(time_cst=datetime.datetime.now().astimezone().isoformat(), remote_root=launch['root'],
    collection_model_forwards=0, collection_optimizer_updates=0, weights_downloaded=0, weights_deleted=0)
(output / 'INTAKE.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=record['status']['status'], files=len(record['files']),
    controller_exit=record['controller_exit'], weights_created=0, output=str(output))), flush=True)
