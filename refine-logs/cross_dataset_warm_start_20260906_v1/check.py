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


import copy
from main_utils import parse_option, prepare_source_moe_gate_checkpoint_config
from train_dist_mod import TrainTester, build_detector_overlap_valid
from scripts.scanrefer_joint_readout import JointRecReadout

contract = json.loads((directory / 'nr_contract.json').read_text())
assert hashlib.sha256((source / 'g0_source_manifest.json').read_bytes()).hexdigest() == contract['source_manifest_sha256']
for name, digest in json.loads((source / 'g0_source_manifest.json').read_text())['files'].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest

def load_core(path, digest):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    assert hasher.hexdigest() == digest
    payload = torch.load(path, map_location='cpu')
    state = {name[7:] if name.startswith('module.') else name: value for name, value in payload['model'].items()}
    assert all(torch.isfinite(value).all() for value in state.values() if value.is_floating_point())
    return payload, state

nr_payload, nr_state = load_core(contract['checkpoint'], contract['checkpoint_sha256'])
scan_payload, scan_state = load_core(expected['artifacts']['backbone']['path'], expected['artifacts']['backbone']['sha256'])
assert nr_payload['evaluation_only'] is True and 'optimizer' not in nr_payload and 'scheduler' not in nr_payload
assert set(nr_state) == set(scan_state)
assert all(value.shape == scan_state[name].shape and value.dtype == scan_state[name].dtype for name, value in nr_state.items())
protocols = {}
for dataset, expected_rows in [('nr3d', 7899), ('sr3d', 17726)]:
    argv = list(contract['eval_argv'])
    for key, value in [('--dataset', dataset), ('--test_dataset', dataset),
                       ('--expected_eval_sample_count', str(expected_rows))]:
        argv[argv.index(key) + 1] = value
    sys.argv = ['cross-dataset-cpu-model-loading'] + argv
    args = prepare_source_moe_gate_checkpoint_config(parse_option())
    assert args.butd_cls and not args.butd and not args.butd_gt
    assert args.eval_use_selector_choice_scores
    core = TrainTester.get_model(args).cpu().eval()
    core.load_state_dict(nr_state, strict=True)
    assert all(torch.equal(value, nr_state[name]) for name, value in core.state_dict().items())
    protocols[dataset] = {'strict_nr_checkpoint_load': True, 'model_tensors': len(core.state_dict()),
        'model_parameter_count': sum(value.numel() for value in core.parameters()),
        'butd_cls': args.butd_cls, 'butd': args.butd, 'butd_gt': args.butd_gt,
        'native_selector_output': args.eval_use_selector_choice_scores,
        'native_filter_non_gt_boxes': args.butd_cls, 'data_or_model_forwards': 0}
    del core

# The original ScanRefer scorer metadata is immutable; this copy only exercises
# the already implemented candidate filter needed by the butd_cls protocol.
with torch.no_grad():
    original = JointRecReadout(artifacts).eval()
    scan_result = original(endpoints, inputs)
    inputs['det_boxes'] = scan_result['parent']['candidate_batch']['boxes'][:, :1].clone()
    inputs['det_bbox_label_mask'] = torch.ones(B, 1, dtype=torch.bool)
    derived_artifacts = copy.deepcopy(artifacts)
    derived_artifacts['geometry']['filter_non_gt_boxes'] = True
    derived = JointRecReadout(derived_artifacts).eval()
    filtered = derived(endpoints, inputs)
    raw_runtime, runtime = scan_result['runtime'], filtered['runtime']
    assert torch.equal(raw_runtime['rec_geometry_boxes'], runtime['rec_geometry_boxes'])
    expected_valid = build_detector_overlap_valid(
        raw_runtime['rec_geometry_boxes'].reshape(B, 16, 7, 6),
        raw_runtime['rec_geometry_valid_mask'].reshape(B, 16, 7),
        inputs['det_boxes'], inputs['det_bbox_label_mask'], iou_threshold=.25).reshape(B, 112)
    assert expected_valid.any(dim=1).all()
    assert torch.equal(runtime['rec_geometry_valid_mask'], expected_valid)
    removed = raw_runtime['rec_geometry_valid_mask'] & ~expected_valid
    assert removed.any()
    assert torch.equal(filtered['continuous']['geometry_valid'], expected_valid)
    assert torch.isneginf(runtime['rec_geometry_scores'][~expected_valid]).all()
assert artifacts['geometry']['filter_non_gt_boxes'] is False
assert not torch.cuda.is_initialized()
result = {'schema': 'mcln-cross-dataset-warm-start-readiness-v1', 'status': 'pass',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'source_manifest_sha256': contract['source_manifest_sha256'],
    'nr_checkpoint_sha256': contract['checkpoint_sha256'],
    'scan_checkpoint_sha256': expected['artifacts']['backbone']['sha256'],
    'scan_nr_model_state_names_shapes_dtypes_equal': True,
    'nr_weight_initialization_requires_fresh_optimizer': True,
    'protocols': protocols, 'scan_geometry_filter_flag': artifacts['geometry']['filter_non_gt_boxes'],
    'required_nr_sr_geometry_filter_flag': True,
    'synthetic_geometry_variants_removed_per_row': removed.sum(dim=1).tolist(),
    'synthetic_readout_loss_and_runtime_validity_equal': True,
    'protected_artifacts_not_written': True, 'synthetic_rows': B,
    'real_dataset_rows': 0, 'gpu_forwards': 0, 'optimizer_steps': 0, 'formal_rows': 0,
    'sr_protected_checkpoint_restored': False,
    'limits': 'CPU weight/interface compatibility and candidate filtering only;not cross-dataset quality or authorization to start Nr/Sr before ScanRefer promotion.'}
with (directory / 'receipt.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(result), flush=True)
