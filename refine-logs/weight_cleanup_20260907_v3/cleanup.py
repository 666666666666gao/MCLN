import datetime,hashlib,json,os,shutil,subprocess
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
