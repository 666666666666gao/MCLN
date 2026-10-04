"""Collect the closed boundary pair's text/row evidence; never download weights."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).parent
output = local/'complete'
assert not output.exists()
launch = json.loads((local/'launch.json').read_bytes())
spec = json.loads((local/'distribution_spec.json').read_bytes())
probe = r'''
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);pid=int(sys.argv[2])
assert root==Path('/root/autodl-tmp/pvground_boundary_fit_20261004')
assert not Path('/proc/%d'%pid).exists()
status=json.loads((root/'status.json').read_bytes())
assert status['status']=='complete' and (root/'controller.exit').read_text().strip()=='0'
assert [(r['arm'],r['mode']) for r in status['completed']]==[
    ('residual','train'),('residual','formal'),('distribution','train'),('distribution','formal')]
files={};weights={}
for path in sorted(root.rglob('*')):
    if not path.is_file():continue
    raw=path.read_bytes();item=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    relative=path.relative_to(root).as_posix()
    if path.suffix=='.pth':
        assert path.name=='terminal.pth' and path.parent in (root/'residual',root/'distribution')
        weights[relative]=item
    else:
        assert path.suffix in ('.py','.md','.json','.jsonl','.log','.exit'),str(path)
        files[relative]=item
best=status['retained_best']
assert len(weights)==(0 if best['protected_parent'] else 1)
if not best['protected_parent']:
    item=weights[str(Path(best['path']).relative_to(root))]
    assert item['sha256']==best['sha256']
parent=Path(sys.argv[4]);env=Path(sys.argv[3])
parent_sha=hashlib.sha256(parent.read_bytes()).hexdigest()
for arm in ('residual','distribution'):
    directory=root/arm
    assert (directory/'train.exit').read_text().strip()=='0'
    assert (directory/'formal.exit').read_text().strip()=='0'
    fit=json.loads((directory/'receipt.json').read_bytes())
    retention=json.loads((directory/'weight_retention.json').read_bytes())
    assert fit['terminal_sha256']==retention['terminal_sha256']
    assert retention['parent_sha256_after']==parent_sha and not retention['local_weight_archive_created']
    for item in retention['deleted']:
        path=Path(item['path'])
        assert path.parent in (root/'residual',root/'distribution') and path.name=='terminal.pth'
        assert not path.exists()
print(json.dumps(dict(status=status,controller_exit=0,controller_alive=False,files=files,
    weights_retained=weights,weight_downloads=0,
    system_disk_free_bytes=shutil.disk_usage('/').free,
    directory_free_bytes=shutil.disk_usage(root).free,
    gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],universal_newlines=True).strip(),
    gpu_compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],universal_newlines=True).strip(),
    environment_spec_sha256=hashlib.sha256(json.dumps(json.loads(env.read_bytes()),sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    original_g_sha256=parent_sha)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
command = shlex.join([spec['runtime']+'/venv/bin/python', '-c', probe, launch['root'],
    launch['process'].split()[0], spec['runtime']+'/env_spec.json', spec['base_terminal']])
_, stdout, stderr = client.exec_command(command, timeout=180)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
assert record['environment_spec_sha256'] == spec['env_spec_sha256']
assert record['original_g_sha256'] == spec['base_terminal_sha256']
output.mkdir()
sftp = client.open_sftp()
for relative, item in record['files'].items():
    path = output/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(item['path'], str(path))
    content = path.read_bytes()
    assert len(content) == item['bytes'] and hashlib.sha256(content).hexdigest() == item['sha256']
sftp.close()
client.close()
record.update(time_cst=datetime.datetime.now().astimezone().isoformat(), remote_root=launch['root'],
    collection_model_forwards=0, collection_optimizer_updates=0, weights_downloaded=0, weights_deleted=0)
(output/'INTAKE.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status='complete', files=len(record['files']), controller_exit=0,
    best=record['status']['retained_best']['name'], weights_downloaded=0, output=str(output))), flush=True)
