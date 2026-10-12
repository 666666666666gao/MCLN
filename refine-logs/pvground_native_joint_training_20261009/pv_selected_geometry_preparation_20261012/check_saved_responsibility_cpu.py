"""Actual saved inputs plus native box terms; not a network/accuracy replay."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

root = Path(__file__).resolve().parent
source = root.parent / 'source'
saved = Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit')
torch.set_num_threads(1)
native_path = source / 'models/losses.py'
tree = ast.parse(native_path.read_bytes().decode('utf-8'))
names = ('box_cxcyczwhd_to_xyzxyz', '_volume_par', '_intersect_par', '_iou3d_par', 'generalized_box_iou3d')
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
criterion_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SetCriterion')
methods = [node for node in criterion_node.body if isinstance(node, ast.FunctionDef)
           and node.name in ('loss_boxes', '_get_src_permutation_idx')]
assert len(nodes) == 5 and len(methods) == 2
nodes.append(ast.ClassDef(name='NativeBoxTerms', bases=[], keywords=[], body=methods, decorator_list=[]))
namespace = dict(torch=torch, F=F)
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes)), str(native_path), 'exec'), namespace)
criterion = namespace['NativeBoxTerms']()
objective_path = root / 'selected_support_geometry_objective.py'
module_spec = importlib.util.spec_from_file_location('prepared_objective', str(objective_path))
objective = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(objective)
bindings, reports = [], []

for step in (1, 2):
    directory = saved / ('preflight_support_step_' + str(step))
    receipt = json.loads((directory / 'receipt.json').read_bytes())
    mask_boxes, scores, own_rows, text_rows, alphas, mappings, targets, indices = [], [], [], [], [], [], [], []
    selected_queries = []
    for witness in receipt['rows']:
        path = directory / Path(witness['path']).name
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == witness['sha256'] and len(raw) == witness['bytes']
        bindings.append(dict(path=str(path), sha256=witness['sha256'], bytes=len(raw)))
        with np.load(str(path), allow_pickle=False) as a:
            q = witness['selected_query']
            current_scores = torch.from_numpy(a['scores'].copy())
            assert int(current_scores.argsort(descending=True)[0]) == q
            obs = int(np.flatnonzero(a['observed_queries'] == q)[0])
            scores.append(current_scores)
            mask_boxes.append(torch.from_numpy(np.concatenate([a['mask_center'], a['mask_size']], -1)))
            # Only the observed selected rows are needed. Dict indexing stands
            # in for the cached slice, not for an executed full model tensor.
            own_rows.append({q: torch.from_numpy(a['observed_query_logits'][obs].copy()).requires_grad_()})
            text_rows.append([{q: torch.from_numpy(a['observed_text_logits'][obs].copy()).requires_grad_()}])
            alphas.append(torch.from_numpy(a['alpha'].copy()).requires_grad_())
            mappings.append(torch.from_numpy(a['superpoints'].copy()).long())
            targets.append(dict(boxes=torch.from_numpy(a['actual_gt_boxes'].copy()),
                                masks=torch.from_numpy(a['actual_gt_masks'].copy())))
            indices.append((torch.from_numpy(a['matched_queries'].copy()).long(),
                            torch.from_numpy(a['matched_gt_ids'].copy()).long()))
            selected_queries.append(q)
    # Final B outputs were not saved. Geometry here is the actual saved Mask
    # endpoint, a kernel-gradient probe only, never a claimed B prediction.
    boxes = torch.stack(mask_boxes).requires_grad_()
    native_scores = torch.stack(scores).requires_grad_()
    predictions = dict(last_center=boxes[..., :3], last_pred_size=boxes[..., 3:],
        sp_last_pred_masks=own_rows, last_pred_masks=text_rows,
        adaptive_weights=alphas, superpoints=mappings)
    batch = dict(language_dataset=['scanrefer'] * 8, box_label_mask=torch.ones(8, 1))
    before_indices = [(a.clone(), b.clone()) for a, b in indices]
    extra, record = objective.selected_support_geometry_loss(predictions, batch, indices, targets, criterion, native_scores)
    original = criterion.loss_boxes(dict(pred_boxes=boxes), targets, indices, 8., None)
    original_gradient = torch.autograd.grad((10 * original['loss_bbox'] + 2 * original['loss_giou']) / 7,
                                          boxes, retain_graph=True)[0]
    extra_gradient = torch.autograd.grad(extra, boxes, retain_graph=True)[0]
    expected = torch.zeros(8, 256, dtype=torch.bool)
    for bid, row in enumerate(record['rows']):
        if row['extra_geometry']:
            expected[bid, row['query']] = True
    observed = extra_gradient.abs().sum(-1) > 0
    assert torch.equal(observed, expected)
    assert original_gradient[expected].abs().sum() == 0
    assert all(torch.equal(a, c) and torch.equal(b, d) for (a, b), (c, d) in zip(indices, before_indices))
    for tensor in [native_scores] + alphas + [t for row in own_rows for t in row.values()] + [t for row in text_rows for t in row[0].values()]:
        gradient = torch.autograd.grad(extra, tensor, allow_unused=True, retain_graph=True)[0]
        assert gradient is None or torch.count_nonzero(gradient) == 0
    assert record['extra_rows'] == int(expected.sum())
    reports.append(dict(step=step, rows=8, original_matched_output_gradients_only=True,
        extra_output_gradient_queries=int(expected.sum()), extra_loss=float(extra),
        extra_gradient_l1=float(extra_gradient.abs().sum()),
        original_and_auxiliary_recipients_disjoint=True, original_indices_unchanged=True,
        qualification_and_selection_gradients_absent=True, record=record))

    # Existing protection rule includes non-root original matches. This is an
    # explicit synthetic engineering check; no dataset accuracy is measured.
    protected = dict(predictions)
    synthetic_boxes = torch.stack(mask_boxes[:1]).requires_grad_()
    protected.update(last_center=synthetic_boxes[..., :3], last_pred_size=synthetic_boxes[..., 3:],
        sp_last_pred_masks=own_rows[:1], last_pred_masks=text_rows[:1],
        adaptive_weights=alphas[:1], superpoints=mappings[:1])
    chosen = selected_queries[0]
    root_query = int(indices[0][0][0])
    assert chosen != root_query
    synthetic_indices = [(torch.tensor([root_query, chosen]), torch.tensor([0, 1]))]
    synthetic_targets = [dict(boxes=targets[0]['boxes'].expand(2, -1).clone(),
                              masks=targets[0]['masks'].expand(2, -1).clone())]
    synthetic_batch = dict(language_dataset=['scanrefer'], box_label_mask=torch.ones(1, 2))
    zero, protection = objective.selected_support_geometry_loss(protected, synthetic_batch, synthetic_indices,
                                                                synthetic_targets, criterion, native_scores[:1])
    assert protection['rows'][0]['role'] == 'matched_other' and protection['extra_rows'] == 0
    assert float(zero) == 0 and torch.count_nonzero(torch.autograd.grad(zero, synthetic_boxes)[0]) == 0
    reports[-1]['synthetic_matched_other_and_empty_extra_protection_pass'] = True

result = dict(status='CPU_NATIVE_BOX_KERNEL_AND_SAVED_SUPPORT_RESPONSIBILITY_PASS',
    torch_version=torch.__version__, input_bindings=bindings,
    native_loss_source=str(native_path), native_loss_source_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
    objective_source_sha256=hashlib.sha256(objective_path.read_bytes()).hexdigest(),
    reports=reports, observations=16, unique_training_rows=8,
    final_geometry_is_saved_Mask_endpoint_not_unsaved_B_output=True,
    native_semantic_score_forward_tested=False, complete_native_get_loss_hook_executed=False,
    full_model_factory_tested=False, new_neural_forwards=0, new_optimizer_updates=0,
    GPU_calls=0, remote_queries=0, active_training_changes=0, formal_accuracy=None)
(root / 'CPU_RESPONSIBILITY_CHECK.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=result['status'], extra_rows_per_step=[r['record']['extra_rows'] for r in reports],
                     observations=16, unique_training_rows=8, new_neural_forwards=0)))
