import datetime,fcntl,hashlib,json,os,shutil,subprocess
from pathlib import Path

directory=Path('/root/autodl-tmp/mcln_weight_cleanup_20260906_v1')
plan=json.loads((directory/'cleanup_plan.json').read_text())
assert plan['user_authorized_unused_weight_cleanup']
lock=Path('/root/autodl-tmp/mcln_v99_backbone_gpu0.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
assert not gpu,gpu
training=Path(plan['training_directory'])
assert (training/'controller.exit').read_text().strip()=='0'
receipt=json.loads((training/'receipt.json').read_text())
audit=json.loads((training/'independent_audit.json').read_text())
assert receipt['status']=='complete' and not receipt['eligible_for_fixed_terminal_formal_evaluation']
assert audit['integrity_pass'] and not audit['eligible_for_fixed_terminal_formal_evaluation']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

protected=plan['protected_files']
assert all(sha(path)==digest for path,digest in protected.items())
assert not set(item['path'] for item in plan['actions']).intersection(protected)
for item in plan['actions']:
    path=Path(item['path'])
    path.resolve().relative_to(Path('/root/autodl-tmp').resolve())
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    assert path.stat().st_nlink==1
    if item['action']=='replace_identical_copy_with_hardlink':
        canonical=Path(item['canonical'])
        assert protected[str(canonical)]==item['sha256']
        assert canonical.stat().st_dev==path.stat().st_dev
        assert canonical.stat().st_mode & 0o777==path.stat().st_mode & 0o777==0o444
        assert not path.with_name(path.name+'.dedup_20260906').exists()
    else:
        assert item['action']=='delete_failed_completed_checkpoint'
        assert path.parent==training
        assert any(record['path']==str(path) and record['sha256']==item['sha256'] for record in receipt['checkpoints'].values())
before=dict(zip(['total','used','free'],shutil.disk_usage('/root/autodl-tmp')))
completed=[]
for item in plan['actions']:
    path=Path(item['path'])
    if item['action']=='replace_identical_copy_with_hardlink':
        canonical=Path(item['canonical'])
        temporary=path.with_name(path.name+'.dedup_20260906')
        os.link(str(canonical),str(temporary))
        os.replace(str(temporary),str(path))
        assert path.stat().st_ino==canonical.stat().st_ino
        assert path.stat().st_size==item['bytes']
    else:
        path.unlink()
        assert not path.exists()
    completed.append(item)
    with (directory/'actions.jsonl').open('a') as f:f.write(json.dumps(item,sort_keys=True)+'\n')
protected_after={path:sha(path) for path in protected}
assert protected_after==protected
after=dict(zip(['total','used','free'],shutil.disk_usage('/root/autodl-tmp')))
result={'schema':'mcln-weight-cleanup-receipt-v1','status':'complete',
        'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'actions':completed,'disk_before':before,'disk_after':after,
        'actual_free_bytes_increase':after['free']-before['free'],
        'protected_sha256_after':protected_after,'gpu_processes_at_start':[],
        'source_data_logs_and_metrics_deleted':False,'cleanup_plan_sha256':sha(directory/'cleanup_plan.json')}
with (directory/'receipt.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'status':result['status'],'actions':len(completed),
                 'freed_gib':result['actual_free_bytes_increase']/1024**3,
                 'free_gib_after':after['free']/1024**3,'protected_hashes_unchanged':True}))
