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


from scripts.scanrefer_rec_evaluation import rec_evaluation_view
from models.rec_candidate_adapter import build_full_rec_query_state
with torch.no_grad():
    endpoints['last_pred_size'][0, 0, 0] = -.25
    endpoints['last_pred_size'][1, -1, 1] = -.1
raw_sizes = endpoints['last_pred_size'].clone()
view = rec_evaluation_view(endpoints)
full = build_full_rec_query_state(endpoints, inputs)
assert torch.equal(view['last_pred_size'], full['boxes'][..., 3:])
assert torch.equal(endpoints['last_pred_size'], raw_sizes)
assert view['last_sem_cls_scores'] is endpoints['last_sem_cls_scores']
assert (raw_sizes < 0).sum() == 2
results = []
for value in [endpoints, view]:
    parent = build_rec_reranker_outputs(value, inputs, models['parent'], artifacts['parent'])
    result = build_rec_geometry_runtime_outputs(value, inputs, parent, models['geometry'], artifacts['geometry'],
        hierarchical_model=models['v99'], hierarchical_artifact=artifacts['v99'])
    results.append(result)
assert all(torch.equal(value, results[1][key]) if torch.is_tensor(value) else value == results[1][key]
           for key, value in results[0].items())
receipt = {'schema': 'mcln-scanrefer-evaluation-view-cpu-v1', 'status': 'pass',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'synthetic_rows': 2, 'negative_size_entries': 2, 'real_data_rows': 0,
    'optimizer_steps': 0, 'gpu_forwards': 0, 'raw_sizes_unchanged': True,
    'same_extent_representation_as_existing_candidate_adapter': True,
    'protected_v99_runtime_outputs_unchanged_by_evaluation_view': True,
    'source_sha256': hashlib.sha256((directory / 'scripts/scanrefer_rec_evaluation.py').read_bytes()).hexdigest()}
with (directory / 'receipt.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(receipt), flush=True)
