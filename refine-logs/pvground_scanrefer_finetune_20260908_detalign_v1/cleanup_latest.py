import hashlib,json,shutil
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_detalign_v1')
audit=Path('/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_detalign_v1')
formal=Path('/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260908_detalign_v1')
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):d.update(chunk)
 return d.hexdigest()
assert (root/'controller.exit').read_text().strip()=='0'
assert (audit/'controller.exit').read_text().strip()=='0'
assert json.loads((audit/'audit.json').read_bytes())['integrity_pass']
assert json.loads((formal/'decision.json').read_bytes())['status']=='skipped_primary_rec_regression'
assert not Path('/proc/9517').exists()
r=json.loads((root/'receipt.json').read_bytes())
terminal=root/'terminal.pth';assert sha(terminal)==r['terminal_sha256']
latest=root/'latest.pth';assert latest.resolve().parent==root.resolve() and latest.is_file()
before=shutil.disk_usage(root).free
record=dict(path=str(latest),bytes=latest.stat().st_size,sha256=sha(latest),terminal_sha256=r['terminal_sha256'],reason='completed fixed-endpoint run; obsolete intermediate resume file; protected terminal retained')
latest.unlink()
assert sha(terminal)==r['terminal_sha256'] and not latest.exists()
record.update(disk_free_before=before,disk_free_after=shutil.disk_usage(root).free,terminal_unchanged=True)
(root/'latest_cleanup.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
