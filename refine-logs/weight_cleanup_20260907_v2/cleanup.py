import datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path
root=Path(__file__).parent
plan=json.loads((root/'cleanup_plan.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
run=Path(plan['run_directory']).resolve()
queue=Path(plan['queue_directory']).resolve()
assert (run/'controller.exit').read_text().strip()=='0'
assert (queue/'controller.exit').read_text().strip()=='0'
assert json.loads((queue/'decision.json').read_text())['status']=='module_rec_screen_not_passed'
for path,digest in plan['evidence_files'].items():assert sha(path)==digest,path
rows=subprocess.check_output(['ps','-eo','pid,comm,args']).decode().splitlines()[1:]
for row in rows:
    fields=row.split(None,2)
    if len(fields)==3 and fields[1]=='python' and int(fields[0])!=os.getpid():
        assert str(run) not in fields[2] and str(queue) not in fields[2],fields[0]
for path,digest in plan['protected_files'].items():assert sha(path)==digest,path
actions=[]
for arm,item in plan['delete'].items():
    p=Path(item['path'])
    assert p.resolve()==p and p.parent==run and p.name==arm+'_frozen_readout_state.pt'
    assert str(p) not in plan['protected_files']
    assert p.stat().st_size==item['bytes'] and p.stat().st_nlink==1 and sha(p)==item['sha256']
    actions.append(dict(item,allocated_bytes=p.stat().st_blocks*512))
before=shutil.disk_usage('/root/autodl-tmp').free
with (root/'verified_before_delete.json').open('x') as f:json.dump({'actions':actions,'free_bytes':before},f,indent=2)
for item in actions:
    p=Path(item['path']);p.unlink();assert not p.exists()
for path,digest in plan['protected_files'].items():assert sha(path)==digest,path
receipt={'status':'complete','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'deleted':actions,'freed_allocated_bytes':sum(item['allocated_bytes'] for item in actions),
    'free_bytes_before':before,'free_bytes_after':shutil.disk_usage('/root/autodl-tmp').free,
    'protected_sha256_before_and_after_verified':plan['protected_files'],'logs_rows_receipts_and_manifests_preserved':True,
    'plan_sha256':sha(root/'cleanup_plan.json')}
with (root/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True)
print(json.dumps(receipt),flush=True)
