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

parent_reference = build_rec_reranker_outputs(endpoints, inputs, models['parent'], artifacts['parent'])
reference = build_rec_geometry_runtime_outputs(
    endpoints, inputs, parent_reference, models['geometry'], artifacts['geometry'],
    hierarchical_model=models['v99'], hierarchical_artifact=artifacts['v99'])

captured = {}
def capture(name):
    def hook(module, arguments, output):
        captured[name] = output
        if name in ['parent', 'geometry']:
            captured[name + '_valid'] = arguments[1]
    return hook
handles = []
for name, model in models.items():
    model.requires_grad_(True)
    handles.append(model.register_forward_hook(capture(name)))
parent = _build_rec_reranker_outputs_float32(endpoints, inputs, models['parent'], artifacts['parent'])
output = _build_rec_geometry_runtime_outputs_float32(
    endpoints, inputs, parent, models['geometry'], artifacts['geometry'],
    hierarchical_model=models['v99'], hierarchical_artifact=artifacts['v99'])
for handle in handles:
    handle.remove()
parity = {key: (torch.equal(value, reference[key]) if torch.is_tensor(value)
                else value == reference[key]) for key, value in output.items()}
assert all(parity.values())

# Synthetic targets exist only to exercise loss gradients. There is no training data or optimizer.
ground_truth = parent['candidate_batch']['boxes'][:, :1].detach()
gt_valid = torch.ones(B, 1, dtype=torch.bool)
parent_iou = compute_query_ious(parent['candidate_batch']['boxes'].detach(), ground_truth, gt_valid)
geometry_iou = compute_query_ious(output['rec_geometry_boxes'].detach(), ground_truth, gt_valid)
parent_loss, _ = compute_rec_reranker_loss(captured['parent'], parent_iou, captured['parent_valid'])
geometry_loss, _ = compute_rec_reranker_loss(captured['geometry'], geometry_iou, captured['geometry_valid'])
variants = output['rec_geometry_valid_mask'].reshape(B, 16, 7)
hierarchy_loss, _ = graded_listwise_loss(captured['v99'], geometry_iou.reshape(B, 16, 7),
                                        variants.any(dim=2), variants)
loss = parent_loss + geometry_loss + hierarchy_loss
assert torch.isfinite(loss)
loss.backward()

def grad_info(parameters):
    result = {}
    for name, value in parameters:
        gradient = value.grad
        result[name] = {'present': gradient is not None,
                        'finite': bool(torch.isfinite(gradient).all()) if gradient is not None else None,
                        'l2': float(gradient.norm()) if gradient is not None else None}
    return result

gradients = {name: grad_info(model.named_parameters()) for name, model in models.items()}
leaf_gradients = grad_info(leaves.items())
for name, model in models.items():
    assert all(torch.equal(value, before[name][key]) for key, value in model.state_dict().items())
    assert all(value['present'] and value['finite'] for value in gradients[name].values())
assert all(value['finite'] for value in leaf_gradients.values() if value['present'])
receipt = {'schema': 'mcln-scanrefer-readout-gradient-probe-v1', 'status': 'complete',
           'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
           'device': 'cpu', 'synthetic_rows': B, 'real_data_rows': 0, 'gpu_forwards': 0,
           'readout_forwards': 2, 'optimizer_steps': 0, 'backbone_forwards': 0, 'checkpoint_writes': 0,
           'protected_state_unchanged': True, 'frozen_vs_grad_forward_parity': parity,
           'final_scores_requires_grad': output['rec_geometry_scores'].requires_grad,
           'parent_compact_scores_requires_grad': parent['compact_scores'].requires_grad,
           'query_features_requires_grad': parent['candidate_batch']['features'].requires_grad,
           'geometry_boxes_requires_grad': output['rec_geometry_boxes'].requires_grad,
           'losses': {'parent': float(parent_loss), 'geometry': float(geometry_loss),
                      'hierarchy': float(hierarchy_loss), 'total': float(loss)},
           'parameter_gradients': gradients, 'endpoint_gradients': leaf_gradients,
           'source_files': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                            for name in ['train_dist_mod.py', 'models/rec_reranker.py',
                                         'models/rec_mask_geometry.py', 'models/rec_pareto_contextual_hierarchy.py']},
           'torch': torch.__version__, 'python': sys.version}
with (directory / 'receipt.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(receipt), flush=True)
