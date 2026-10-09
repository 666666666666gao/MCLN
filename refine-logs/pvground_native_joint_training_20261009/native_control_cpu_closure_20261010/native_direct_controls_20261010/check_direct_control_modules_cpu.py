"""CPU module equivalence/gradient checks; synthetic inputs, no REC evaluation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bundle_path = Path(sys.argv[1])
bundle = json.loads(bundle_path.read_bytes())
root = bundle_path.parent
original = Path(bundle['original_source'])
assert original == Path('/root/autodl-tmp/pvground_native_joint_training_20261010/PV-Ground')
assert not (root / 'CPU_MODULE_WITNESS.json').exists()
torch.set_num_threads(1)
torch.manual_seed(2027)
np.random.seed(2027)
for name, expected in bundle['original_sha256'].items():
    assert digest(original / name) == expected
for name, expected in bundle['prepared_sha256'].items():
    assert digest(root / name) == expected
sys.path.insert(0, str(original))
old_A = load_module('original_A_cpu_check', original / 'mask_support_corrector.py')
old_B = load_module('original_B_cpu_check', original / 'extremal_span_mixer.py')
new_A = load_module('direct_A_cpu_check', root / 'source/mask_support_corrector.py')
new_B = load_module('direct_B_cpu_check', root / 'source/extremal_span_mixer.py')
geometry_module = load_module('original_geometry_cpu_check', original / 'native_mask_geometry.py')
state_spec = json.loads((root / 'common_behavior_check/init.json').read_bytes())
for role in ('support', 'span'):
    assert digest(state_spec[role + '_checkpoint']) == state_spec[role + '_checkpoint_sha256']
support_state = torch.load(state_spec['support_checkpoint'], map_location='cpu')
span_state = torch.load(state_spec['span_checkpoint'], map_location='cpu')
prefix = 'candidate_support_corrector.'
assert support_state['arm'] == 'selected_query' and support_state['geometry_encoding'] == 'zero'
assert span_state['source_mode'] == 'extremal_support'
a_state = {name[len(prefix):]: value for name, value in support_state['state_delta'].items()}
assert len(a_state) == 10 and len(span_state['mixer_state']) == 14

a_old = old_A.CandidateMaskSupportCorrector(False)
a_common = new_A.CandidateMaskSupportCorrector(False, 'full')
a_content = new_A.CandidateMaskSupportCorrector(False, 'content_only')
a_bypass = new_A.CandidateMaskSupportCorrector(False, 'bypass')
for module in (a_old, a_common, a_content, a_bypass):
    module.load_state_dict(a_state, strict=True)
    assert list(module.state_dict()) == list(a_old.state_dict())
    assert sum(p.numel() for p in module.parameters()) == 27841
a_bypass.requires_grad_(False)
b_old = old_B.ExtremalSpanMixer('extremal_support')
b_common = new_B.ExtremalSpanMixer('extremal_support', 'learned')
b_fixed = new_B.ExtremalSpanMixer('extremal_support', 'fixed_half')
for module in (b_old, b_common, b_fixed):
    module.load_state_dict(span_state['mixer_state'], strict=True)
    assert list(module.state_dict()) == list(b_old.state_dict())
    assert sum(p.numel() for p in module.parameters()) == 29793
b_fixed.requires_grad_(False)

# Explicit engineering fixture: random embeddings/probabilities and eight
# equally populated synthetic superpoints. No GT, scenes, or accuracy metrics.
queries = torch.randn(1, 256, 288)
supports = [torch.randn(288, 8)]
raw_points = torch.randn(1, 50000, 6)
native_center = torch.randn(1, 256, 3)
native_size = torch.rand(1, 256, 3) + .5
own = torch.randn(256, 8)
text = torch.randn(256, 8)
alpha = torch.tensor(.4)


def predictions(query_mask):
    return dict(superpoints=[torch.arange(50000) % 8],
        sp_last_pred_masks=[query_mask], last_pred_masks=[[text]],
        adaptive_weights=[alpha], last_center=native_center, last_pred_size=native_size)


captures = []
hook = a_content.member[0].register_forward_pre_hook(
    lambda module, args: captures.append(args[0][..., 64:70].detach().clone()))
with torch.no_grad():
    old_masks, old_geometry = a_old(queries, supports, raw_points,
        native_center, native_size, predictions(own))
    masks, geometries = a_common(queries, supports, raw_points,
        native_center, native_size, predictions(own))
    content_masks, _ = a_content(queries, supports, raw_points,
        native_center, native_size, predictions(own))
    bypass_masks, _ = a_bypass(queries, supports, raw_points,
        native_center, native_size, predictions(own))
hook.remove()
assert torch.equal(old_masks[0], masks[0])
assert torch.equal(bypass_masks[0], own) and bypass_masks[0] is own
assert len(captures) == 1 and not torch.count_nonzero(captures[0]).item()
for key in old_geometry[0]:
    assert np.array_equal(old_geometry[0][key], geometries[0][key]), key
p = predictions(masks[0])
geometry_module.native_mask_geometry(p, geometries)
with torch.no_grad():
    old_centers, old_sizes, old_evidence = b_old(queries, supports, p, geometries)
    centers, sizes, evidence = b_common(queries, supports, p, geometries)
    fixed_centers, fixed_sizes, fixed_evidence = b_fixed(queries, supports, p, geometries)
assert torch.equal(old_centers, centers) and torch.equal(old_sizes, sizes)
for key in old_evidence[0]:
    assert torch.equal(old_evidence[0][key], evidence[0][key]), key
assert torch.equal(fixed_centers, .5 * p['last_center'] + .5 * native_center)
assert torch.equal(fixed_sizes, .5 * p['last_pred_size'] + .5 * native_size.clamp_min(1e-6))
assert torch.equal(fixed_evidence[0]['axis_gate'], torch.full((256, 3), .5))
assert torch.equal(fixed_evidence[0]['raw_axis_gate'], torch.full((256, 3), .5))
assert not any(parameter.requires_grad for parameter in a_bypass.parameters())
assert not any(parameter.requires_grad for parameter in b_fixed.parameters())

gradient_query = queries.detach().clone().requires_grad_(True)
gradient_masks, _ = a_common(gradient_query, supports, raw_points,
    native_center, native_size, predictions(own))
gradient_masks[0].square().mean().backward()
assert gradient_query.grad is not None and torch.count_nonzero(gradient_query.grad).item() > 0
assert a_common.output.weight.grad is not None and torch.count_nonzero(a_common.output.weight.grad).item() > 0
gradient_center = native_center.detach().clone().requires_grad_(True)
gradient_size = native_size.detach().clone().requires_grad_(True)
fixed_prediction = dict(p, native_coarse_center=gradient_center, native_coarse_size=gradient_size)
fc, fs, _ = b_fixed(queries, supports, fixed_prediction, geometries)
(fc.sum() + fs.sum()).backward()
assert torch.equal(gradient_center.grad, torch.full_like(gradient_center, .5))
assert torch.equal(gradient_size.grad, torch.full_like(gradient_size, .5))
assert all(parameter.grad is None for parameter in b_fixed.parameters())

report = dict(
    status='ACTUAL_CPU_MODULE_CHECKS_COMPLETE_NO_FORMAL_MODEL_OR_ACCURACY',
    torch_version=torch.__version__, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
    tensor_devices=[str(queries.device), str(raw_points.device)],
    evaluation_type='synthetic_module_engineering_fixture_not_accuracy_evaluation',
    fixture=dict(seed=2027, batch=1, candidates=256, superpoints=8, points=50000),
    original_source_sha256=bundle['original_sha256'], prepared_sha256=bundle['prepared_sha256'],
    support_payload_sha256=state_spec['support_checkpoint_sha256'],
    span_payload_sha256=state_spec['span_checkpoint_sha256'],
    A_state_tensors=10, B_state_tensors=14,
    common_A_masks_bitwise_equal_original=True, common_B_boxes_and_evidence_bitwise_equal_original=True,
    bypass_A_returns_original_query_mask=True, content_only_observation_input_columns_exactly_zero=True,
    content_vs_full_mask_max_abs_difference=(content_masks[0] - masks[0]).abs().max().item(),
    fixed_B_gates_exactly_half=True, fixed_B_native_box_gradient_exactly_half=True,
    bypass_A_and_fixed_B_parameters_frozen=True,
    common_A_has_query_and_output_weight_gradients=True,
    full_PV_constructor_or_1295_state_checked=False, native_criterion_checked=False,
    optimizer_scheduler_or_full_cold_recovery_checked=False,
    source_or_geometry_extremal_accuracy_claim=False, formal_accuracy=None,
    current_training_status_reads=0, GPU_calls=0, full_goal_complete=False)
(root / 'CPU_MODULE_WITNESS.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
