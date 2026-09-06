"""CPU synthetic audit of the protected ScanRefer readout's gradient boundary."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

directory, source = map(Path, sys.argv[1:3])
os.chdir(str(source))
sys.path.insert(0, str(source))
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''

import torch
import scripts
scripts.__path__ = [str(directory / 'scripts')] + list(scripts.__path__)
from models.rec_pareto_contextual_hierarchy import ParetoContextualHierarchicalReranker
from models.rec_reranker import QueryReranker, compute_query_ious, compute_rec_reranker_loss
from scripts.run_v95_threshold_aligned_listwise_hierarchical import graded_listwise_loss
from train_dist_mod import (
    _build_rec_reranker_outputs_float32,
    _build_rec_geometry_runtime_outputs_float32,
    build_rec_reranker_outputs,
    build_rec_geometry_runtime_outputs,
)

torch.set_num_threads(1)
torch.manual_seed(7)
expected = json.loads((directory / 'expected.json').read_text())
artifacts, models = {}, {}
for name in ['parent', 'geometry', 'v99']:
    raw = Path(expected['artifacts'][name]['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected['artifacts'][name]['sha256']
    artifact = torch.load(expected['artifacts'][name]['path'], map_location='cpu')
    artifacts[name] = artifact
    if name == 'v99':
        model = ParetoContextualHierarchicalReranker()
    else:
        model = QueryReranker(**artifact['model_config'])
    model.load_state_dict(artifact['model_state_dict'], strict=True)
    models[name] = model.eval()
assert not artifacts['geometry']['filter_non_gt_boxes']
before = {name: {key: value.clone() for key, value in model.state_dict().items()}
          for name, model in models.items()}

B, Q, T, S, P = 2, 256, 8, 32, 256
coords = torch.stack(torch.meshgrid(torch.arange(8), torch.arange(8), torch.arange(4)), dim=-1)
coords = coords.reshape(P, 3).float() * .15
inputs = {'point_clouds': torch.cat([coords.unsqueeze(0).expand(B, -1, -1), torch.rand(B, P, 3)], dim=-1),
          'superpoint': [torch.arange(P) // 8 for _ in range(B)]}
for index, name in enumerate(['positive_map', 'modify_positive_map', 'pron_positive_map',
                              'rel_positive_map', 'other_entity_map']):
    inputs[name] = torch.zeros(B, 1, T)
    inputs[name][:, 0, index] = 1.
endpoints = {
    'last_center': torch.rand(B, Q, 3),
    'last_pred_size': torch.rand(B, Q, 3) + .25,
    'last_sem_cls_scores': torch.randn(B, Q, T),
    'last_proj_queries': torch.randn(B, Q, 64),
    'proj_tokens': torch.randn(B, T, 64),
    'seeds_obj_cls_logits': torch.randn(B, 1, 1024),
    'query_points_sample_inds': torch.arange(Q).unsqueeze(0).expand(B, -1),
    'last_pred_masks': [torch.randn(1, Q, S) - 1. for _ in range(B)],
    'sp_last_pred_masks': [torch.randn(Q, S) - 1. for _ in range(B)],
    'adaptive_weights': [torch.tensor(.05) for _ in range(B)],
}
leaves = {}
for key, value in endpoints.items():
    if torch.is_tensor(value) and value.is_floating_point():
        leaves[key] = value.requires_grad_(True)
    elif isinstance(value, list):
        for index, tensor in enumerate(value):
            leaves[key + '_' + str(index)] = tensor.requires_grad_(True)


from scripts.scanrefer_joint_readout import JointRecReadout, joint_rec_readout_loss
import io

def same_outputs(first, second):
    return all(torch.equal(value, second[key]) if torch.is_tensor(value) else value == second[key]
               for key, value in first.items())

parent_reference = build_rec_reranker_outputs(endpoints, inputs, models['parent'], artifacts['parent'])
reference = build_rec_geometry_runtime_outputs(
    endpoints, inputs, parent_reference, models['geometry'], artifacts['geometry'],
    hierarchical_model=models['v99'], hierarchical_artifact=artifacts['v99'])
module = JointRecReadout(artifacts).eval()
initial = {name: value.clone() for name, value in module.state_dict().items()}
results = {}
for control in [True, False]:
    module.zero_grad(set_to_none=True)
    for value in leaves.values():
        value.grad = None
    outputs = module(endpoints, inputs, detach_visual=control)
    assert same_outputs(outputs['runtime'], reference)
    target = outputs['parent']['candidate_batch']['boxes'][:, :1].detach()
    loss, stats = joint_rec_readout_loss(outputs, target, torch.ones(B, 1, dtype=torch.bool))
    assert torch.isfinite(loss)
    loss.backward()
    parameter_grads = {name: {'present': value.grad is not None,
                             'finite': bool(torch.isfinite(value.grad).all()) if value.grad is not None else None,
                             'l2': float(value.grad.norm()) if value.grad is not None else None}
                       for name, value in module.named_parameters()}
    endpoint_grads = {name: None if value.grad is None else float(value.grad.norm())
                      for name, value in leaves.items()}
    assert len(parameter_grads) == 42
    assert all(value['present'] and value['finite'] for value in parameter_grads.values())
    if control:
        assert all(value is None for value in endpoint_grads.values())
    else:
        assert all(value is not None and value > 0 for value in endpoint_grads.values())
        assert all(value['l2'] > 0 for value in parameter_grads.values())
    results['detached' if control else 'joint'] = {
        'parity': True, 'loss': float(loss), 'stats': stats,
        'parameter_grads': parameter_grads, 'endpoint_grads': endpoint_grads}
assert results['detached']['loss'] == results['joint']['loss']

module.zero_grad(set_to_none=True)
for value in leaves.values():
    value.grad = None
miss_output = module(endpoints, inputs)
miss_target = torch.tensor([1000., 1000., 1000., 1., 1., 1.]).reshape(1, 1, 6).expand(B, -1, -1)
miss_loss, miss_stats = joint_rec_readout_loss(miss_output, miss_target, torch.ones(B, 1, dtype=torch.bool))
assert miss_stats['parent_covered_rows'] == miss_stats['geometry_covered_rows'] == 0
assert miss_stats['hierarchy_covered_queries'] == 0 and miss_stats['hierarchy_loss'] == 0
assert miss_stats['parent_loss'] > 0 and miss_stats['geometry_loss'] > 0
assert torch.isfinite(miss_loss)
miss_loss.backward()
assert all(torch.isfinite(value.grad).all() for value in module.parameters() if value.grad is not None)

payload = io.BytesIO()
torch.save({'kind': 'joint-rec-readout-state-v1', 'readout': module.export_artifacts()}, payload)
payload.seek(0)
restored = JointRecReadout(torch.load(payload, map_location='cpu')['readout']).eval()
with torch.no_grad():
    restored_output = restored(endpoints, inputs)
assert same_outputs(restored_output['runtime'], reference)
assert all(torch.equal(value, initial[name]) for name, value in module.state_dict().items())
assert all(torch.equal(value, restored.state_dict()[name]) for name, value in module.state_dict().items())

receipt = {'schema': 'mcln-scanrefer-joint-readout-cpu-v1', 'status': 'pass',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'synthetic_rows': B, 'real_data_rows': 0, 'gpu_forwards': 0, 'backbone_forwards': 0,
    'optimizer_steps': 0, 'checkpoint_file_writes': 0, 'protected_state_unchanged': True,
    'detached_and_joint_forward_parity': True, 'matched_loss': True,
    'detached_gradient_boundary_correct': True, 'joint_endpoint_and_parameter_gradients_connected': True,
    'no_covered_candidate_rank_positive_rows': 0, 'all_miss_quality_loss_retained': True,
    'all_miss_stats': miss_stats, 'in_memory_checkpoint_round_trip': True,
    'readout_tensor_count': len(initial), 'readout_parameter_count': sum(value.numel() for value in module.parameters()),
    'arms': results, 'torch': torch.__version__, 'python': sys.version,
    'source_sha256': hashlib.sha256((directory / 'scripts/scanrefer_joint_readout.py').read_bytes()).hexdigest()}
with (directory / 'receipt.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({key: value for key, value in receipt.items() if key != 'arms'}), flush=True)
