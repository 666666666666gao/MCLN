import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/weight_cleanup_20260906_v2'
remote = '/root/autodl-tmp/mcln_weight_cleanup_20260906_v2'
old_dir = repo / 'refine-logs/scanrefer_local_visual_pair_20260906_v1'
formal_dir = repo / 'refine-logs/scanrefer_local_visual_official_20260906_v3'
old = json.loads((old_dir / 'receipt.json').read_bytes())
audit = json.loads((old_dir / 'independent_audit.json').read_bytes())
formal = json.loads((formal_dir / 'result/independent_audit.json').read_bytes())
assert audit['integrity_pass'] and formal['integrity_pass']
assert not formal['promotion']['advance_to_nr3d_sr3d_rec']
protected = json.loads((repo / 'refine-logs/weight_cleanup_20260906_v1/cleanup_plan.json').read_bytes())['protected_files']
plan = {'user_authorized_unused_weight_cleanup': True,
    'old_directory': '/root/autodl-tmp/mcln_scanrefer_local_visual_pair_20260906_v1',
    'formal_directory': '/root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v3',
    'new_directory': '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_pair_20260906_v1',
    'old_receipt_sha256': hashlib.sha256((old_dir / 'receipt.json').read_bytes()).hexdigest(),
    'old_audit_sha256': hashlib.sha256((old_dir / 'independent_audit.json').read_bytes()).hexdigest(),
    'formal_audit_sha256': hashlib.sha256((formal_dir / 'result/independent_audit.json').read_bytes()).hexdigest(),
    'protected_files': protected, 'delete': old['checkpoints'],
    'reason': 'both fixed old-root endpoints fully audited; corrected formal local fails REC; new run initializes protected E71 and does not load either endpoint',
    'preserve': 'all logs, row metrics, receipts, source manifests and six protected weight files'}
code = '''import datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path
root=Path(__file__).parent
plan=json.loads((root/'cleanup_plan.json').read_text())
def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):d.update(block)
 return d.hexdigest()
old=Path(plan['old_directory']).resolve()
formal=Path(plan['formal_directory']).resolve()
assert (old/'controller.exit').read_text().strip()=='0'
assert (formal/'controller.exit').read_text().strip()=='0'
assert sha(old/'receipt.json')==plan['old_receipt_sha256']
assert sha(old/'independent_audit.json')==plan['old_audit_sha256']
assert sha(formal/'result/independent_audit.json')==plan['formal_audit_sha256']
assert not json.loads((formal/'result/receipt.json').read_text())['promotion']['advance_to_nr3d_sr3d_rec']
new=json.loads((Path(plan['new_directory'])/'input_manifest.json').read_text())
assert new['data_root']=='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/'
deleted_paths={item['path'] for item in plan['delete'].values()}
assert not deleted_paths.intersection(item['path'] for item in new['artifacts'].values())
rows=subprocess.check_output(['ps','-eo','pid,comm,args']).decode().splitlines()[1:]
for line in rows:
 parts=line.split(None,2)
 if len(parts)==3 and parts[1]=='python' and int(parts[0])!=os.getpid():
  assert not any(path in parts[2] for path in [str(old),str(formal)]),parts[0]
for path,digest in plan['protected_files'].items():assert sha(path)==digest,path
free_before=shutil.disk_usage('/root/autodl-tmp').free
actions=[]
for arm,item in plan['delete'].items():
 path=Path(item['path'])
 assert path.resolve()==path and path.parent==old and path.name==arm+'_local_visual_state.pt'
 assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
 assert path.stat().st_nlink==1
 actions.append(dict(item,arm=arm,allocated_bytes=path.stat().st_blocks*512))
with (root/'verified_before_delete.json').open('x') as f:json.dump({'actions':actions,'free_bytes':free_before},f,indent=2)
for item in actions:
 path=Path(item['path'])
 path.unlink()
 assert not path.exists()
for path,digest in plan['protected_files'].items():assert sha(path)==digest,path
receipt={'schema':'mcln-sealed-failed-local-weight-cleanup-v1','status':'complete',
 'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'deleted':actions,'freed_allocated_bytes':sum(item['allocated_bytes'] for item in actions),
 'free_bytes_before':free_before,'free_bytes_after':shutil.disk_usage('/root/autodl-tmp').free,
 'protected_sha256_before_and_after_verified':plan['protected_files'],
 'logs_rows_receipts_and_manifests_preserved':True,'plan_sha256':sha(root/'cleanup_plan.json')}
with (root/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True)
print(json.dumps(receipt))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
local.mkdir()
sftp.mkdir(remote)
for name, raw in {'cleanup_plan.json': (json.dumps(plan,indent=2,sort_keys=True)+'\n').encode(),
                  'cleanup.py': code.encode(), 'launch_helper.py': Path(__file__).read_bytes()}.items():
    with sftp.open(remote + '/' + name,'wx') as stream: stream.write(raw)
    (local / name).write_bytes(raw)
_, out, err = client.exec_command('/root/miniconda3/envs/bdetr/bin/python ' + shlex.quote(remote + '/cleanup.py'), timeout=60)
stdout = out.read().decode()
stderr = err.read().decode()
(local / 'stdout.txt').write_text(stdout)
(local / 'stderr.txt').write_text(stderr)
assert out.channel.recv_exit_status() == 0, stderr
for name in ['verified_before_delete.json','receipt.json']:
    with sftp.open(remote + '/' + name,'rb') as stream: raw=stream.read()
    (local / name).write_bytes(raw)
sftp.close()
client.close()
receipt=json.loads((local/'receipt.json').read_bytes())
print(json.dumps({key:receipt[key] for key in ['status','time_cst','freed_allocated_bytes','free_bytes_before','free_bytes_after']}))
