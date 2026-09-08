"""Actual CUDA grouping/gradient check for one empty and one supported query."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_modules import StackSAModuleMSG

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from pvground_empty_pool_mask import EmptyPoolMaskedVSA

torch.set_num_threads(1)
torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
native = StackSAModuleMSG(radii=[.2], nsamples=[2], mlps=[[2, 4, 4]], use_xyz=True).cuda().eval()
with torch.no_grad():
    for layer in native.modules():
        if isinstance(layer, torch.nn.BatchNorm2d): layer.bias.fill_(1.)
control = EmptyPoolMaskedVSA(copy.deepcopy(native), False)
candidate = EmptyPoolMaskedVSA(copy.deepcopy(native), True)
state = {name: value.detach().clone() for name, value in native.state_dict().items()}
assert list(control.state_dict()) == list(candidate.state_dict()) == list(state)
xyz = torch.tensor([[0., 0., 0.], [.05, 0., 0.]], device='cuda')
centers = torch.tensor([[0., 0., 0.], [2., 2., 2.]], device='cuda')
counts = torch.tensor([2], dtype=torch.int32, device='cuda')
features = torch.tensor([[.2, .3], [.4, .5]], device='cuda', requires_grad=True)
arguments = (xyz, counts, centers, counts, features)
_, original = native(*arguments)
_, baseline = control(*arguments)
_, masked = candidate(*arguments)
assert torch.equal(original, baseline)
assert torch.equal(original[0], masked[0])
assert (original[1] != 0).any() and torch.equal(masked[1], torch.zeros_like(masked[1]))
masked[1].sum().backward(retain_graph=True)
assert torch.equal(features.grad, torch.zeros_like(features.grad))
assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in candidate.parameters())
features.grad = None
candidate.zero_grad()
masked[0].sum().backward()
assert features.grad is not None and torch.isfinite(features.grad).all() and (features.grad != 0).any()
assert all(torch.equal(value, state[name]) for name, value in candidate.state_dict().items())
record = dict(status='pass', seed=2027, native_equals_disabled=True, supported_query_exact=True,
              empty_native_nonzero=True, empty_candidate_exact_zero=True, empty_loss_gradients_zero=True,
              supported_feature_gradient_finite_nonzero=True, state_keys_and_values_unchanged=True,
              module_forwards=3, full_model_forwards=0, optimizer_steps=0, formal_rows=0,
              native_output=original.detach().cpu().tolist(), masked_output=masked.detach().cpu().tolist(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              module_sha256=hashlib.sha256((root/'pvground_empty_pool_mask.py').read_bytes()).hexdigest())
(root / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
print('EMPTY_POOL_CUDA_CHECK_PASS ' + json.dumps(record), flush=True)
