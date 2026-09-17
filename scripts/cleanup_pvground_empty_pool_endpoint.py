"""Remove only the sealed empty-pool negative endpoint; retain its cached outputs."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


base=Path('/root/autodl-tmp').resolve()
records=[]
for suffix in ['20260908_emptypool_v1']:
    roots=[base/('mcln_pvground_scanrefer_'+kind+'_'+suffix) for kind in ['finetune','endpoint_audit','formal']]
    root,audit_root,formal=roots
    for directory in roots:
        assert (directory/'controller.exit').read_text().strip()=='0'
        launch=json.loads((directory/'launch.json').read_bytes())
        pid=int(launch['process'].split()[0]);cmd=Path('/proc')/str(pid)/'cmdline'
        assert not cmd.exists() or (str(directory)+'/controller.py').encode() not in cmd.read_bytes()
    receipt=json.loads((root/'receipt.json').read_bytes())
    audit=json.loads((audit_root/'audit.json').read_bytes())
    decision=json.loads((formal/'decision.json').read_bytes())
    assert receipt['status']=='complete' and receipt['training_steps']==3723
    assert not receipt['primary_rec_nonregression'] and audit['integrity_pass']
    assert audit['receipt_sha256']==sha(root/'receipt.json')
    assert decision['status']=='skipped_primary_rec_regression' and decision['formal_rows']==0
    path=root/'terminal.pth'
    assert path.resolve().parent==root.resolve() and root.resolve().parent==base
    assert path.is_file() and not path.is_symlink()
    assert sha(path)==receipt['terminal_sha256']=='e5cc9aa891e5386b74fe2c9d6390f2ce3935cf18e986c0144b1808b06f42e773'
    open_handles=[str(fd) for proc in Path('/proc').glob('[0-9]*')
                  for fd in (proc/'fd').glob('*') if os.path.realpath(str(fd))==str(path)]
    assert not open_handles,open_handles
    for stage in ['initial','terminal']:
        saved=json.loads((root/stage/'receipt.json').read_bytes())
        for name,ext in [('rows','jsonl'),('boxes','npy'),('scores','npy')]:
            assert sha(root/stage/(name+'.'+ext))==saved[name+'_sha256']
    records.append(dict(path=str(path),bytes=path.stat().st_size,sha256=receipt['terminal_sha256'],
        training_receipt_sha256=sha(root/'receipt.json'),decision=decision['status'],
        cached_outputs_preserved=True,reason='sealed empty-pool negative endpoint; D starts from official epoch81 and does not load this checkpoint'))
result=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            candidates=records,bytes=sum(r['bytes'] for r in records),disk_free_before=shutil.disk_usage(base).free,removed=False)
if records:
    for record in records:
        path=Path(record['path'])
        with (path.parent/'obsolete_terminal_cleanup_intent.json').open('x') as f:json.dump(record,f,indent=2)
        path.unlink();assert not path.exists()
    result.update(removed=True,disk_free_after=shutil.disk_usage(base).free)
    with (base/'mcln_pvg_empty_endpoint_cleanup_20260917.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
