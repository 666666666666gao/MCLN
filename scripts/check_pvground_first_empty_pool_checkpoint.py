import hashlib,json,shutil
from pathlib import Path
import torch

root=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_emptypool_v1')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
spec=json.loads((root/'spec.json').read_bytes())
checkpoint=root/'latest.pth';before=sha(checkpoint)
data=torch.load(str(checkpoint),map_location='cpu')
assert data['step']==512 and len(data['row_ids'])==4096
assert data['empty_pool_mask'] is True
assert data['empty_pool_module_sha256']==spec['empty_pool_module_sha256']==sha(root/'pvground_empty_pool_mask.py')
assert data['spec_sha256']==sha(root/'spec.json')
assert data['parent_checkpoint_sha256']==spec['checkpoint_sha256']
assert all(v.device.type=='cpu' for v in data['state_delta'].values())
assert not torch.cuda.is_initialized()
assert before==sha(checkpoint)
print(json.dumps(dict(status='metadata_pass',step=data['step'],rows=len(data['row_ids']),
    checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=before,
    empty_pool_mask=data['empty_pool_mask'],module_sha256=data['empty_pool_module_sha256'],
    spec_sha256=data['spec_sha256'],state_tensors=len(data['state_delta']),disk_free=shutil.disk_usage(str(root)).free,
    checkpoint_unchanged=True,gpu_initialized=False,model_forwards=0,optimizer_steps=0,formal_rows=0,
    scope='CPU saved metadata and tensor-location check; not a full restored-model forward or quality evaluation')))
