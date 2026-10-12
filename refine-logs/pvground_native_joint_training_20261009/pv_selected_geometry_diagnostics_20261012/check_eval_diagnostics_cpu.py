"""CPU math/source fixtures, not a model, dataset or accuracy evaluation."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import types

import numpy as np
import torch

root = Path(__file__).resolve().parent
base = root.parent
directory = root / 'diagnostic_cpu_fixture_r3_model_evaluator'
assert not directory.exists()
directory.mkdir()
loss_tree = ast.parse((root / 'source_overlay_v2/models/losses.py').read_text(encoding='utf-8'))
names = {'box_cxcyczwhd_to_xyzxyz', '_volume_par', '_intersect_par', '_iou3d_par'}
scope = dict(torch=torch)
functions = [node for node in loss_tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(functions) == 4
exec(compile(ast.fix_missing_locations(ast.Module(body=functions)), '<saved native box kernel>', 'exec'), scope)
loss_module = types.ModuleType('models.losses')
for name in names:
    setattr(loss_module, name, scope[name])
sys.modules['models'] = types.ModuleType('models')
sys.modules['models.losses'] = loss_module
score_spec = importlib.util.spec_from_file_location('native_root_bbs', str(base / 'source/native_root_bbs.py'))
score_module = importlib.util.module_from_spec(score_spec)
sys.modules['native_root_bbs'] = score_module
score_spec.loader.exec_module(score_module)
spec = importlib.util.spec_from_file_location('diagnostic_actual', str(root / 'selected_geometry_eval_diagnostics.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
evaluator_path = base.parent / 'runtime_binding/model_source/src/grounding_evaluator.py'
stage = json.loads((root / 'SOURCE_STAGE_R3_RECEIPT.json').read_bytes())
assert hashlib.sha256(evaluator_path.read_bytes()).hexdigest() == stage['source_files']['src/grounding_evaluator.py']['sha256']
tree = ast.parse(evaluator_path.read_text(encoding='utf-8'))
original_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GroundingEvaluator')
selected_methods = {'_parse_gt', 'evaluate_bbox_by_pos_align'}
original_class.body = [node for node in original_class.body if isinstance(node, ast.FunctionDef) and node.name in selected_methods]
assert len(original_class.body) == 2
exec(compile(ast.fix_missing_locations(ast.Module(body=[original_class])), str(evaluator_path), 'exec'), scope)
evaluator = scope['GroundingEvaluator'].__new__(scope['GroundingEvaluator'])
evaluator.only_root = True
evaluator.thresholds = [.25, .5]
evaluator.topks = [1, 5, 10]
evaluator.filter_non_gt_boxes = False
evaluator.dets = {('last_', threshold, topk, 'bbs'): 0 for threshold in (.25, .5) for topk in (1, 5, 10)}
evaluator.gts = dict(evaluator.dets)

batch = 2
centers = torch.full((batch, 256, 3), 20.)
centers[0, 5] = torch.tensor([.25, 0., 0.])
centers[1, 17] = torch.tensor([2.1, 0., 0.])
centers[1, 200] = 0
sizes = torch.full_like(centers, 2.)
logits = torch.zeros(batch, 256, 4)
logits[0, 5, 0] = 5
logits[1, 17, 0] = 6
native = centers.clone()
native[0, 5] = torch.tensor([1., 0., 0.])
mask_centers = centers.clone()
mask_centers[0, 5] = mask_centers[1, 17] = 0
map_root = torch.tensor([[[1., 0., 0., 0.]]]).repeat(batch, 1, 1)
maps = {name: torch.zeros_like(map_root) for name in
    ('modify_positive_map', 'pron_positive_map', 'rel_positive_map', 'other_entity_map', 'auxi_entity_positive_map')}
own = [torch.full((256, 4), -4.) for unused in range(batch)]
own[0][5, :2] = own[1][17, :2] = 4
points = dict(last_sem_cls_scores=logits, last_center=centers, last_pred_size=sizes,
    native_coarse_center=native, native_coarse_size=sizes.clone(),
    span_mask_center=mask_centers, span_mask_size=sizes.clone(),
    center_label=torch.zeros(batch, 1, 3), size_gts=torch.full((batch, 1, 3), 2.),
    box_label_mask=torch.ones(batch, 1), language_dataset=['scanrefer'] * batch,
    superpoints=[torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])] * batch,
    sp_last_pred_masks=own, last_pred_masks=[[value.clone()] for value in own],
    adaptive_weights=[torch.tensor(.5)] * batch,
    gt_masks=torch.tensor([[[1, 1, 1, 1, 0, 0, 0, 0]]]).repeat(batch, 1, 1),
    utterances=['synthetic fixture . not mentioned'] * batch,
    scan_ids=['engineering_scene'] * batch, target_id=torch.tensor([0, 1]),
    positive_map=map_root, point_clouds=torch.arange(batch * 8 * 6).reshape(batch, 8, 6).float(), **maps)
before = copy.deepcopy(points)
torch_rng = torch.get_rng_state().clone()
python_rng = random.getstate()
numpy_rng = np.random.get_state()
diagnostic = module.SelectedGeometryEvalDiagnostics(directory, 0)
diagnostic.append(points)
assert [row['selected_query'] for row in diagnostic.rows] == [5, 17]
evaluator.evaluate_bbox_by_pos_align(points, 'last_')
assert evaluator.dets[('last_', .25, 1, 'bbs')] == evaluator.dets[('last_', .5, 1, 'bbs')] == 1
assert sum(row['final_iou'] > .5 for row in diagnostic.rows) == 1
assert diagnostic.rows[0]['first_qualified_rank050'] == 1
assert diagnostic.rows[1]['first_qualified_rank050'] > 1
assert all(row['best256_iou'] > .5 for row in diagnostic.rows)
assert all(2 * row['own_mask_intersection'] > row['own_mask_union'] for row in diagnostic.rows)
assert torch.equal(torch.get_rng_state(), torch_rng) and random.getstate() == python_rng
after_numpy = np.random.get_state()
assert numpy_rng[0] == after_numpy[0] and numpy_rng[2:] == after_numpy[2:]
assert np.array_equal(numpy_rng[1], after_numpy[1])
def identical(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(identical(left[name], right[name]) for name in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(identical(a, b) for a, b in zip(left, right))
    return left == right
assert identical(before, points)
two_rows = copy.deepcopy(diagnostic.rows)
diagnostic.rows = []
for index in range(9508):
    row = copy.deepcopy(two_rows[index % 2])
    row['row_index'] = index
    diagnostic.rows.append(row)
metrics = dict(rows=9508, primary_score='last/bbs', hits025=4754, hits050=4754)
initial = diagnostic.finish(metrics)
assert initial['050']['full256_covered'] == 9508
assert initial['050']['errors_with_qualified_candidate'] == 4754
assert initial['050']['final_vs_native_repairs'] == 4754
assert initial['050']['final_vs_mask_endpoint_breaks'] == 4754
second = module.SelectedGeometryEvalDiagnostics(directory, 1)
second.rows = copy.deepcopy(diagnostic.rows)
second.rows[1]['final_iou'] = .75  # One explicit engineering repair.
second.rows[1]['point_cloud_sha256'] = 'changed_input_fixture'
final = second.finish(dict(metrics, hits025=4755, hits050=4755))
assert final['versus_E0']['050']['repairs'] == 1 and final['versus_E0']['050']['breaks'] == 0
assert final['versus_E0']['exact_same_input_rows'] == 9507 and not final['versus_E0']['all_inputs_identical']
assert final['versus_E0']['050']['exact_input_repairs'] == 0
# Remove only the read-only additions and prove the complete training entry is unchanged.
source = (root / 'source_overlay_v4/train_dist_mod.py').read_text(encoding='utf-8')
source = source.replace('from selected_geometry_eval_diagnostics import SelectedGeometryEvalDiagnostics\n', '')
source = source.replace('        diagnostics = SelectedGeometryEvalDiagnostics(args.log_dir, epoch)\n', '')
source = source.replace('\n                    diagnostics.append(end_points)', '')
source = source.replace('        diagnostics.finish(metrics)\n        return metrics', '')
source = source.replace("        metrics = dict(rows=9508, hits025=int(evaluator.dets[keys[0]]),", "        return dict(rows=9508, hits025=int(evaluator.dets[keys[0]]),")
assert ast.dump(ast.parse(source)) == ast.dump(ast.parse((base / 'source/train_dist_mod.py').read_text(encoding='utf-8')))
checks = ['selected_geometry_eval_diagnostics.py', 'source_overlay_v4/train_dist_mod.py',
    'observe_study_authorized.py', 'stage_eval_diagnostics_authorized.py',
    'launch_gpu_preflight_pair_authorized_v4.py', 'launch_normal_arm_authorized_v4.py']
embedded = 0
for name in checks:
    raw = (root / name).read_bytes()
    assert b'\r\r\n' not in raw
    tree = ast.parse(raw.decode())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'code' for target in node.targets):
            code = ast.literal_eval(node.value)
            ast.parse(code)
            embedded += 1
result = dict(status='CPU_NATIVE_EVALUATOR_DIAGNOSTIC_MATH_AND_SOURCE_EQUIVALENCE_PASS',
    actual_staged_PV_evaluator_source_path=str(evaluator_path),
    actual_staged_PV_evaluator_sha256=hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
    actual_native_box_kernel_and_root_ranking_exercised=True, fixture_batch_rows=2,
    synthetic_aggregation_rows=9508, actual_dataset_rows=0, actual_model_forwards=0,
    GPU_calls=0, optimizer_updates=0, RNG_and_inputs_unchanged=True,
    complete_original_train_entry_AST_after_removing_diagnostics_equal=True,
    saved_syntax_files=checks, embedded_fragments=embedded,
    two_epoch_synthetic_row_file_bytes=initial['row_file_bytes'] + final['row_file_bytes'],
    formal_accuracy=None, source_only=True)
(root / 'EVAL_DIAGNOSTIC_CPU_CHECK.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
print(json.dumps(result))
