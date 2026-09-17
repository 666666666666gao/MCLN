"""Read-only CPU comparison of the protected parent and fixed D terminal buffers."""
import hashlib
import json
from pathlib import Path
import torch

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

training=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_task_observation_v1')
spec=json.loads((training/'spec.json').read_bytes())
receipt=json.loads((training/'receipt.json').read_bytes())
assert (training/'controller.exit').read_text().strip()=='0'
assert receipt['training_steps']==3723 and receipt['status']=='complete'
assert sha(training/'spec.json')==receipt['spec_sha256']
assert sha(training/'terminal.pth')==receipt['terminal_sha256']=='ce03188965491a82bcb1c5a6d26f590d3a243a01985457f220d5503c75b2fcf5'
environment=json.loads((Path(spec['runtime'])/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
checkpoint=environment['weight_dirs']['scanrefer']
assert sha(checkpoint['path'])==checkpoint['sha256']==receipt['checkpoint_sha256']
torch.set_num_threads(1)
parent={k[7:]:v for k,v in torch.load(checkpoint['path'],map_location='cpu')['model'].items()}
delta=torch.load(str(training/'terminal.pth'),map_location='cpu')['state_delta']
rows=[]
for name,initial in parent.items():
    if not name.endswith(('running_mean','running_var','num_batches_tracked')):
        continue
    assert name in delta,name
    terminal=delta[name]
    assert initial.shape==terminal.shape and initial.dtype==terminal.dtype
    diff=terminal.double()-initial.double()
    row=dict(name=name,shape=list(initial.shape),changed=not torch.equal(initial,terminal),
        initial_mean=float(initial.double().mean()),terminal_mean=float(terminal.double().mean()),
        max_abs_change=float(diff.abs().max()),rms_change=float(diff.square().mean().sqrt()))
    if name.endswith('running_var'):
        assert bool((initial>=0).all()) and bool((terminal>=0).all())
        row.update(initial_min=float(initial.min()),terminal_min=float(terminal.min()))
    rows.append(row)
assert rows
print(json.dumps(dict(status='complete_cpu_census',rows=rows,
    buffer_tensors=len(rows),changed_tensors=sum(r['changed'] for r in rows),
    parent_sha256=checkpoint['sha256'],terminal_sha256=receipt['terminal_sha256'],
    training_spec_sha256=sha(training/'spec.json'),script_sha256=sha(__file__),
    model_forwards=0,optimizer_steps=0,new_checkpoints=0,formal_rows=0),indent=2))
