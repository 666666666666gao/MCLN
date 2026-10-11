"""Exercise the observed initialization invariant, not an accuracy evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from torch import nn

root = Path(__file__).resolve().parent
assert not torch.cuda.is_available()
torch.manual_seed(2027)
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FaceResidualSpanMixer

old_path = root.parent / 'face_native_gpu_20261011/source/face_residual_span_mixer.py'
new_path = root / 'source/face_residual_span_mixer.py'
old_class = load(old_path, 'failed_face')
new_class = load(new_path, 'repaired_face')
prior = nn.Identity()
prior.source_mode = 'extremal_support'
old = old_class(prior, True)
new = new_class(prior, True)
new.load_state_dict(old.state_dict(), strict=True)
query = torch.randn(256, 288)
native_center = torch.randn(256, 3)
native_size = torch.rand(256, 3) + 0.1
mask_center = torch.randn(256, 3)
mask_size = torch.rand(256, 3) + 0.1
# The prior's positive sizes are not finally floored. Exercise that exact
# invariant, with several magnitudes on either side of the newly added floor.
mask_size[:32] = torch.tensor([1e-7, 3e-7, 2e-6])
gate = torch.rand(256, 3)
gate[:32] = 0
prior_center = (1 - gate) * mask_center + gate * native_center
prior_size = (1 - gate) * mask_size + gate * native_size.clamp_min(1e-6)
evidence = dict(axis_gate=gate, face_tokens=torch.randn(256, 6, 32),
    source_fraction=torch.rand(256, 3, 2), reference_valid=torch.ones(256, dtype=torch.bool))
inputs = (query, native_center, native_size, mask_center, mask_size, prior_center, prior_size, evidence)
with torch.no_grad():
    old_center, old_size, old_evidence = old.refine_one(*inputs)
    new_center, new_size, new_evidence = new.refine_one(*inputs)
assert torch.equal(old_center, prior_center) and torch.equal(new_center, prior_center)
assert not torch.equal(old_size, prior_size)
assert torch.equal(new_size, prior_size)
assert torch.count_nonzero(new_evidence['face_movement']) == 0
control = new_class(prior, False)
control.load_state_dict(new.state_dict(), strict=True)
with torch.no_grad():
    control_center, control_size, _ = control.refine_one(*inputs)
assert torch.equal(control_center, new_center) and torch.equal(control_size, new_size)
record = dict(status='CPU_NEUTRAL_SIZE_REGRESSION_PASS', device='cpu',
    scope='Constructed initialization regression only; not dataset or GPU evidence',
    old_changed_size_elements=int(torch.count_nonzero(old_size - prior_size)),
    new_changed_size_elements=int(torch.count_nonzero(new_size - prior_size)),
    prior_minimum_size=float(prior_size.min()), old_minimum_size=float(old_size.min()),
    new_minimum_size=float(new_size.min()), initial_control_and_source_equal=True,
    old_source_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
    new_source_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
    torch_version=torch.__version__, CUDA_available=False, formal_accuracy=None)
(root / 'CPU_NEUTRAL_SIZE_REGRESSION.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
