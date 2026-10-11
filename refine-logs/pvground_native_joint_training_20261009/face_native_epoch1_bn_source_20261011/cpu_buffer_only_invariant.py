"""Constructed PyTorch state-invariant check; not a PV accuracy experiment."""
import datetime
import json
from pathlib import Path
import torch
from torch import nn
from batchnorm_buffer_intervention import capture_batchnorm_buffers,install_batchnorm_buffers

root=Path(__file__).resolve().parent
assert not (root/'CPU_BUFFER_ONLY_INVARIANT.json').exists()
torch.set_num_threads(1)
torch.manual_seed(2027)
model=nn.Sequential(nn.BatchNorm1d(3),nn.Linear(3,2),nn.BatchNorm1d(2),nn.LayerNorm(2))
reference=capture_batchnorm_buffers(model)
assert len(reference)==6
optimizer=torch.optim.SGD(model.parameters(),lr=0.05)
model.train()
for offset in (2.0,4.0):
    data=torch.arange(24,dtype=torch.float32).reshape(8,3)/5+offset
    loss=model(data).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()
before={key:value.detach().clone() for key,value in model.state_dict().items()}
trained=capture_batchnorm_buffers(model)
assert any(not torch.equal(trained[key],reference[key]) for key in trained)
rng=torch.get_rng_state().clone()
install_batchnorm_buffers(model,reference)
after=model.state_dict()
assert all(torch.equal(after[key],reference[key]) for key in reference)
assert all(torch.equal(after[key],value) for key,value in before.items() if key not in reference)
assert torch.equal(rng,torch.get_rng_state())
install_batchnorm_buffers(model,trained)
assert all(torch.equal(model.state_dict()[key],value) for key,value in before.items())
record=dict(status='CPU_BUFFER_ONLY_INTERVENTION_INVARIANT_PASS',
 time_cst=datetime.datetime.now().astimezone().isoformat(),torch_version=torch.__version__,
 execution_scope='Constructed CPU PyTorch model: two BatchNorms, Linear and LayerNorm; no PV/data/GPU',
 buffer_count=6,all_non_buffer_states_exact=True,affine_weights_unchanged=True,
 reference_buffers_installed_exactly=True,trained_buffers_restored_exactly=True,
 global_torch_rng_unchanged=True,actual_PV_forward=False,actual_formal_accuracy=None,
 causal_degradation_diagnosis_proven=False,active_training_source_changed=False)
(root/'CPU_BUFFER_ONLY_INVARIANT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
