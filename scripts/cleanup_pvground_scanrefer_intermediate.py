"""Remove only the completed, sealed trial's superseded latest checkpoint."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


root=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_vsaorder_v1').resolve()
audit_root=Path('/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_vsaorder_v1')
formal=Path('/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260908_vsaorder_v1')
assert root.parent==Path('/root/autodl-tmp').resolve()
for directory in [root,audit_root,formal]:assert (directory/'controller.exit').read_text().strip()=='0'
assert not Path('/proc/14249').exists()
receipt=json.loads((root/'receipt.json').read_bytes())
audit=json.loads((audit_root/'audit.json').read_bytes())
assert audit['integrity_pass'] and audit['receipt_sha256']==sha(root/'receipt.json')
assert not audit['primary_rec_nonregression'] and receipt['training_steps']==3723
assert json.loads((formal/'decision.json').read_bytes())['status']=='skipped_primary_rec_regression'
latest=root/'latest.pth';terminal=root/'terminal.pth'
assert latest.resolve().parent==root and terminal.resolve().parent==root
assert latest.is_file() and terminal.is_file() and not latest.is_symlink() and not terminal.is_symlink()
terminal_sha=sha(terminal);assert terminal_sha==receipt['terminal_sha256']
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
payload=torch.load(str(latest),map_location='cpu')
assert payload['step']==3584 and payload['step']<receipt['training_steps']
assert payload['parent_checkpoint_sha256']==receipt['checkpoint_sha256']
assert payload['spec_sha256']==receipt['spec_sha256']
assert len(payload['row_ids'])==3584*8
record=dict(file=str(latest),sha256=sha(latest),bytes=latest.stat().st_size,step=payload['step'],
    superseded_by=str(terminal),terminal_sha256=terminal_sha,terminal_step=3723,
    independent_audit_sha256=sha(audit_root/'audit.json'),disk_free_before=shutil.disk_usage(root).free,
    model_forwards=0,optimizer_steps=0,torch_cuda_initialized=torch.cuda.is_initialized())
assert not record['torch_cuda_initialized']
with (root/'latest_cleanup_intent.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
del payload
latest.unlink()
assert not latest.exists() and terminal.is_file() and sha(terminal)==terminal_sha
record.update(removed=True,time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              disk_free_after=shutil.disk_usage(root).free,terminal_preserved=True)
with (root/'latest_cleanup.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps(record))
