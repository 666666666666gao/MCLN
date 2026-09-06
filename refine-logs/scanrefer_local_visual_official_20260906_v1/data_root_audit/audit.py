import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_local_visual_official_20260906_v1/data_root_audit'
local.mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
historical = '/root/autodl-tmp/DATA_ROOT/output/v99_meshsp_official_20260814'
names = sftp.listdir(historical)
(local / 'historical_files.json').write_text(json.dumps(names, indent=2))
for name in names:
    if name.endswith('.json'):
        with sftp.open(historical + '/' + name, 'rb') as stream:
            (local / name).write_bytes(stream.read())
program = r'''
import datetime, hashlib, json
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

root=Path('/root/autodl-tmp')
views={}
for path in sorted(root.glob('DATA_ROOT*')):
    views[path.name]={'path':str(path),'resolved':str(path.resolve()),'superpoints':{}}
    for split in ['train','val']:
        sp=path/'superpoints'/split
        views[path.name]['superpoints'][split]={'exists':sp.exists(),'resolved':str(sp.resolve())}
old=root/'DATA_ROOT/superpoints/val'
new=root/'DATA_ROOT/superpoints_mesh_official/val'
assert old.is_dir() and new.is_dir()
old_files={path.name:path for path in old.iterdir() if path.is_file()}
new_files={path.name:path for path in new.iterdir() if path.is_file()}
common=sorted(set(old_files)&set(new_files))
same=[]
different=[]
for name in common:
    item={'name':name,'old_sha256':sha(old_files[name]),'mesh_sha256':sha(new_files[name])}
    (same if item['old_sha256']==item['mesh_sha256'] else different).append(item)
formal=root/'mcln_scanrefer_local_visual_official_20260906_v1'
protocol=json.loads((formal/'result/protocol.json').read_text())
command=protocol['authoritative_base_command']
data_root=command[command.index('--data_root')+1]
result={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'actual_formal_data_root':data_root,'views':views,'old_val_files':len(old_files),
        'mesh_val_files':len(new_files),'same_files':same,'different_files':different,
        'same_count':len(same),'different_count':len(different),
        'old_only':sorted(set(old_files)-set(new_files)),'mesh_only':sorted(set(new_files)-set(old_files)),
        'gpu_forwards':0,'optimizer_steps':0,'checkpoint_writes':0}
print(json.dumps(result))
'''
command = '/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(program)
_, output, error = client.exec_command(command, timeout=300)
raw = output.read()
error_text = error.read().decode()
assert output.channel.recv_exit_status() == 0, error_text
(local / 'receipt.json').write_bytes(raw)
(local / 'audit.py').write_bytes(Path(__file__).read_bytes())
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v1/data_root_audit'
sftp.mkdir(remote)
for path in local.iterdir():
    with sftp.open(remote + '/' + path.name, 'wx') as stream:
        stream.write(path.read_bytes())
sftp.close()
client.close()
result = json.loads(raw)
print(json.dumps({key:value for key,value in result.items() if key not in ['same_files','different_files']}))
print(json.dumps({'historical_files':names,'receipt_sha256':hashlib.sha256(raw).hexdigest()}))
