import hashlib,json,time
from pathlib import Path
import torch
r=Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v3')
spec=json.loads((r/'spec.json').read_text())
path=Path(spec['state_root'])/'latest.pth'
started=time.time()
checkpoint=torch.load(str(path),map_location='cpu')
parent=torch.load(spec['checkpoint'],map_location='cpu')
assert checkpoint['step']>=512 and checkpoint['step']%512==0
assert checkpoint['rows']==checkpoint['step']*8
assert checkpoint['spec_sha256']==hashlib.sha256((r/'spec.json').read_bytes()).hexdigest()
assert checkpoint['parent_checkpoint_sha256']==spec['checkpoint_sha256']
assert set(checkpoint['model'])==set(parent['model'])
assert all(checkpoint['model'][k].shape==v.shape and checkpoint['model'][k].dtype==v.dtype for k,v in parent['model'].items())
assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
assert len(checkpoint['optimizer']['state'])>0
assert len(checkpoint['optimizer']['param_groups'])==3
for key in ['random_state','numpy_state','torch_rng_state','cuda_rng_states']:
    assert key in checkpoint
h=hashlib.sha256()
with path.open('rb') as f:
    for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
report={'status':'load_verified','path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest(),
        'step':checkpoint['step'],'rows':checkpoint['rows'],'model_states':len(checkpoint['model']),
        'optimizer_states':len(checkpoint['optimizer']['state']),'seconds':time.time()-started,
        'rng_saved':True,'model_states_finite':True,'model_forwards':0,'optimizer_steps_executed':0,
        'claim_boundary':'CPU deserialization, model keys/shapes/finiteness and recovery payload verified; no resumed training step or validation selection.'}
dest=r/('recovery_checkpoint_'+str(checkpoint['step'])+'.json')
with dest.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report),flush=True)
