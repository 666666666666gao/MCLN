import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/weight_cleanup_20260907_v3'
remote='/root/autodl-tmp/mcln_weight_cleanup_20260907_v3'
old='/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_pair_20260906_v1'
formal='/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_official_20260906_v1'
diagnostic='/root/autodl-tmp/mcln_scanrefer_stage_diagnostic_20260907_v1'
digest=lambda raw:hashlib.sha256(raw).hexdigest()
train=repo/'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1'
formal_local=repo/'refine-logs/scanrefer_local_visual_mesh_official_20260906_v1/result'
diagnostic_local=repo/'refine-logs/scanrefer_stage_diagnostic_20260907_v1/diagnostic_result'
assert json.loads((train/'independent_audit.json').read_bytes())['integrity_pass']
fa=json.loads((formal_local/'independent_audit.json').read_bytes())
assert fa['integrity_pass'] and not fa['promotion']['advance_to_nr3d_sr3d_rec']
assert json.loads((diagnostic_local/'receipt.json').read_bytes())['status']=='complete'
evidence={}
for directory,source,names in [
    (old,train,['receipt.json','independent_audit.json','terminal_rows.json']),
    (formal+'/result',formal_local,['receipt.json','independent_audit.json','rows.json','native_rows.json']),
    (diagnostic+'/diagnostic_result',diagnostic_local,['receipt.json','independent_audit.json','stage_rows.json','stage_summary.json'])]:
    for name in names:evidence[directory+'/'+name]=digest((source/name).read_bytes())
active={
 '/root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1/input_manifest.json':'2e7b7fa65afd056c0ad98a9a92e974e563df6e104624abdb30cfaf4716e1e72a',
 '/root/autodl-tmp/mcln_scanrefer_native_box_transfer_posttraining_20260907_v1/input_manifest.json':'c1760a8a6107ee1b8dbca5ba2fde536dfe60b501ec3e38f1358daa29c609e942'}
plan={'schema':'mcln-sealed-mesh-endpoint-cleanup-v1','user_authorized_unused_weight_cleanup':True,
 'run_directory':old,'formal_directory':formal,'diagnostic_directory':diagnostic,
 'delete':json.loads((train/'receipt.json').read_bytes())['checkpoints'],
 'protected_files':json.loads((repo/'refine-logs/weight_cleanup_20260907_v2/cleanup_plan.json').read_bytes())['protected_files'],
 'evidence_files':evidence,'active_manifests':active,
 'reason':'Correct-mesh local visual trial failed formal REC; all dependent formal/stage work finished; current E71 teacher-box pair uses neither endpoint.',
 'preserve':'all prediction rows, scientific negative results, source, manifests and six protected weights'}
code=r'''import datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
plan=json.loads((root/'cleanup_plan.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
old=Path(plan['run_directory']).resolve()
formal=Path(plan['formal_directory']).resolve()
diagnostic=Path(plan['diagnostic_directory']).resolve()
for p in [old,formal,diagnostic]:assert (p/'controller.exit').read_text().strip()=='0',str(p)
fa=json.loads((formal/'result/independent_audit.json').read_text())
assert fa['integrity_pass'] and not fa['promotion']['advance_to_nr3d_sr3d_rec']
assert json.loads((diagnostic/'diagnostic_result/receipt.json').read_text())['status']=='complete'
for path,digest in plan['evidence_files'].items():assert sha(path)==digest,path
delete_paths={v['path'] for v in plan['delete'].values()}
for path,digest in plan['active_manifests'].items():
 assert sha(path)==digest,path
 raw=Path(path).read_text()
 assert not any(target in raw for target in delete_paths)
rows=subprocess.check_output(['ps','-eo','pid,comm,args']).decode().splitlines()[1:]
for row in rows:
 fields=row.split(None,2)
 if len(fields)==3 and fields[1] in ('python','screen') and int(fields[0])!=os.getpid():
  assert not any(str(p) in fields[2] for p in [old,formal,diagnostic]),fields[0]
for path,digest in plan['protected_files'].items():assert sha(path)==digest,path
actions=[]
for arm,item in plan['delete'].items():
 p=Path(item['path'])
 assert p.resolve()==p and p.parent==old and p.name==arm+'_local_visual_state.pt'
 assert str(p) not in plan['protected_files']
 assert p.stat().st_size==item['bytes'] and p.stat().st_nlink==1 and sha(p)==item['sha256']
 actions.append(dict(item,arm=arm,allocated_bytes=p.stat().st_blocks*512))
assert len(actions)==2
before=shutil.disk_usage('/root/autodl-tmp').free
with (root/'verified_before_delete.json').open('x') as f:json.dump({'actions':actions,'free_bytes':before,'referenced_jobs_terminal':True},f,indent=2)
for item in actions:
 p=Path(item['path']);p.unlink();assert not p.exists()
for path,digest in plan['protected_files'].items():assert sha(path)==digest,path
for path,digest in plan['active_manifests'].items():assert sha(path)==digest,path
for path,digest in plan['evidence_files'].items():assert sha(path)==digest,path
result={'schema':'mcln-sealed-mesh-endpoint-cleanup-v1','status':'complete',
 'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'deleted':actions,'freed_allocated_bytes':sum(v['allocated_bytes'] for v in actions),
 'free_bytes_before':before,'free_bytes_after':shutil.disk_usage('/root/autodl-tmp').free,
 'protected_sha256_before_and_after_verified':plan['protected_files'],
 'active_manifests_unchanged':plan['active_manifests'],'evidence_sha256_preserved':plan['evidence_files'],
 'plan_sha256':sha(root/'cleanup_plan.json'),'training_interrupted':False}
with (root/'receipt.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
print(json.dumps(result),flush=True)
'''
local.mkdir()
files={'cleanup.py':code.encode(),'cleanup_plan.json':(json.dumps(plan,indent=2,sort_keys=True)+'\n').encode(),'launch_helper.py':Path(__file__).read_bytes(),
 'inspection.json':Path('C:/Users/gb/.codex/tmp/mcln_mesh_cleanup_inventory_20260907.json').read_bytes()}
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();s.mkdir(remote)
for name,raw in files.items():
 (local/name).write_bytes(raw)
 with s.open(remote+'/'+name,'wx') as f:f.write(raw)
 with s.open(remote+'/'+name,'rb') as f:assert f.read()==raw
_,o,e=c.exec_command('/root/miniconda3/envs/bdetr/bin/python '+shlex.quote(remote+'/cleanup.py'),timeout=60)
stdout=o.read();stderr=e.read();status=o.channel.recv_exit_status()
(local/'stdout.txt').write_bytes(stdout);(local/'stderr.txt').write_bytes(stderr)
assert status==0,stderr.decode()
for name in ['verified_before_delete.json','receipt.json']:
 with s.open(remote+'/'+name,'rb') as f:(local/name).write_bytes(f.read())
s.close();c.close()
result=json.loads((local/'receipt.json').read_bytes())
assert result['status']=='complete' and result['plan_sha256']==digest(files['cleanup_plan.json'])
print(json.dumps({k:result[k] for k in ['status','time_cst','freed_allocated_bytes','free_bytes_before','free_bytes_after','training_interrupted']}))
