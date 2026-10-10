"""Synthetic CPU checks for the isolated native matcher and box criterion.

No PV forward, real dataset, Mask loss, optimizer, GPU or accuracy evaluation.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch

root = Path(__file__).resolve().parent
spec = json.loads((root / 'CPU_CHECK_SPEC.json').read_bytes())
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
torch.set_num_threads(1)
torch.manual_seed(2027)
warm = Path(spec['warm_source'])
assert hashlib.sha256((warm / 'models/losses.py').read_bytes()).hexdigest() == spec['original_losses_sha256']
assert hashlib.sha256((root / 'source/models/losses.py').read_bytes()).hexdigest() == spec['query_losses_sha256']
sys.path.insert(0, str(warm))


def load_losses(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module.__file__ == str(path)
    return module


original = load_losses('unchanged_warm_losses', warm / 'models/losses.py')
modified = load_losses('isolated_query_losses', root / 'source/models/losses.py')
bindings = {}
for name, filename in (('utils', 'utils/__init__.py'), ('utils.scatter_util', 'utils/scatter_util.py')):
    path = Path(sys.modules[name].__file__)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert path == warm / filename and digest == spec['import_binding_sha256'][filename]
    bindings[name] = dict(path=str(path), sha256=digest)
matchers = dict(original=original.HungarianMatcher(1, 0, 2, True),
    default=modified.HungarianMatcher(1, 0, 2, True),
    text=modified.HungarianMatcher(1, 0, 2, True, mask_source='text'),
    query=modified.HungarianMatcher(1, 0, 2, True, mask_source='query'))
for matcher in matchers.values():
    assert (matcher.cost_class, matcher.cost_bbox, matcher.cost_giou, matcher.cost_masks) == (1, 0, 2, .0002)

captured = []
native_assignment = modified.linear_sum_assignment


def capture(matrix):
    captured.append(matrix.detach().cpu().numpy().copy())
    return native_assignment(matrix)


original.linear_sum_assignment = capture
modified.linear_sum_assignment = capture


def fixture(counts, mutation='none'):
    boxes = torch.zeros(len(counts), 256, 6)
    boxes[..., :3] = 30
    boxes[..., 3:] = 1
    logits = torch.zeros(len(counts), 256, 256)
    mapping = torch.tensor([0, 0, 1, 2, 3, 3], dtype=torch.long)
    text_masks, query_masks, targets = [], [], []
    for batch, count in enumerate(counts):
        target_boxes = torch.tensor([[float(4 * gt), 0., 0., 1., 1., 1.] for gt in range(count)])
        own = torch.full((256, 4), -2.)
        for gt in range(count):
            low, high = 3 + 28 * gt, 17 + 28 * gt
            boxes[batch, low] = boxes[batch, high] = target_boxes[gt] + torch.tensor([.05, 0., 0., 0., 0., 0.])
            correct = torch.tensor([2., 2., -2., -2.]) if gt == 0 else torch.tensor([-2., -2., 2., 2.])
            own[low], own[high] = -correct, correct
            if mutation == 'swap_own':
                own[low], own[high] = own[high].clone(), own[low].clone()
        text = torch.tensor([2., 2., -2., -2.]).view(1, 1, 4).expand(1, 256, 4)
        if mutation == 'change_text':
            text = torch.full_like(text, -2.)
        text_masks.append(text)
        query_masks.append(own)
        pmap = torch.zeros(count, 256)
        pmap[:, 0] = 1.
        masks = torch.stack([torch.tensor([1., 1., 0., 0.]) if gt == 0 else torch.tensor([0., 0., 1., 1.]) for gt in range(count)])
        targets.append(dict(labels=torch.zeros(count, dtype=torch.int64), boxes=target_boxes,
            masks=masks[:, mapping], positive_map=pmap,
            auxi_box=target_boxes, auxi_entity_positive_map=pmap))
    outputs = dict(pred_logits=logits, pred_boxes=boxes, pred_masks=text_masks,
        sp_pred_masks=query_masks, superpoints=[mapping.clone() for _ in counts])
    if mutation == 'no_masks':
        del outputs['pred_masks']
        del outputs['sp_pred_masks']
    return outputs, targets


records = []


def assignment_as_gt(result):
    return [dict(sorted(zip(gt.tolist(), query.tolist()))) for query, gt in result]


def run(name, mode, counts, mutation='none', gradient=False):
    outputs, targets = fixture(counts, mutation)
    captured.clear()
    if gradient:
        outputs['pred_boxes'].requires_grad_()
        criterion = modified.SetCriterion(matchers[mode], losses=['boxes'])
        box_losses, result = criterion(outputs, targets)
        objective = 10 * box_losses['loss_bbox'] + 2 * box_losses['loss_giou']
        assert torch.isfinite(objective).all()
        objective.backward()
        grad = outputs['pred_boxes'].grad
        active = [torch.nonzero(grad[b].abs().sum(-1) > 0).flatten().tolist() for b in range(len(counts))]
        expected = [sorted(pair[0].tolist()) for pair in result]
        assert active == expected
        gradient_info = dict(loss_bbox=float(box_losses['loss_bbox']), loss_giou=float(box_losses['loss_giou']),
            directly_supervised_queries=active, unmatched_direct_gradient_max=0.,
            criterion_scope='Native SetCriterion.forward with boxes-only requested losses; no full training loss.')
    else:
        result = matchers[mode](outputs, targets)
        gradient_info = None
    assert len(captured) == len(counts)
    assignments = assignment_as_gt(result)
    for batch, count in enumerate(counts):
        assert len(result[batch][0].unique()) == len(result[batch][1].unique()) == count
        assert list(assignments[batch]) == list(range(count))
        high = mode == 'query' and mutation not in ('swap_own', 'no_masks')
        assert assignments[batch] == {gt: (17 if high else 3) + 28 * gt for gt in range(count)}
    records.append(dict(name=name, mode=mode, gt_counts=counts, mutation=mutation,
        assignments=[dict(query=pair[0].tolist(), gt=pair[1].tolist()) for pair in result],
        gradient=gradient_info))
    return [matrix.copy() for matrix in captured]


checks = []
for label, counts in (('root_only', [1]), ('root_anchor', [2]), ('mixed_gt_batch', [1, 2])):
    old = run(label + '_original', 'original', counts)
    default = run(label + '_default', 'default', counts)
    text = run(label + '_text', 'text', counts)
    query = run(label + '_query', 'query', counts)
    swapped = run(label + '_query_swapped_own', 'query', counts, 'swap_own')
    text_swapped = run(label + '_text_swapped_own', 'text', counts, 'swap_own')
    query_text = run(label + '_query_changed_text', 'query', counts, 'change_text')
    early_text = run(label + '_no_mask_text', 'text', counts, 'no_masks')
    early_query = run(label + '_no_mask_query', 'query', counts, 'no_masks')
    for name, first, second in (('default_matches_warm', old, default), ('explicit_text_matches_warm', old, text),
            ('text_ignores_own_mutation', text, text_swapped), ('query_ignores_shared_text_mutation', query, query_text),
            ('no_mask_prefix_unchanged', early_text, early_query)):
        differences = [float(abs(a - b).max()) for a, b in zip(first, second)]
        assert all(value == 0 for value in differences)
        checks.append(dict(fixture=label, check=name, maximum_cost_difference=differences))
    differences = [float(abs(a - b).max()) for a, b in zip(query, swapped)]
    assert all(value > 0 for value in differences)
    checks.append(dict(fixture=label, check='query_own_mutation_changes_cost_and_assignment',
        maximum_cost_difference=differences))
    run(label + '_text_box_gradient', 'text', counts, gradient=True)
    run(label + '_query_box_gradient', 'query', counts, gradient=True)

assert not torch.cuda.is_initialized()
receipt = dict(status='ACTUAL_ISOLATED_QUERY_MASK_ASSIGNMENT_CPU_CHECK_COMPLETE',
    evaluation_type='simulation_only', constructed_case_count=3, matcher_calls=len(records),
    assignment_problems=sum(len(row['gt_counts']) for row in records),
    assigned_gt_pairs=sum(sum(row['gt_counts']) for row in records),
    native_boxes_only_criterion_calls=sum(row['gradient'] is not None for row in records),
    neural_model_forward_calls=0, full_native_criterion_calls=0, Mask_loss_calls=0,
    dataset_rows=0, optimizer_updates=0, gpu_calls=0, current_training_queries=0,
    active_training_source_mutations=0, formal_accuracy=None,
    source_losses_path=modified.__file__, original_losses_path=original.__file__,
    actual_import_bindings=bindings, torch_version=torch.__version__, spec=spec,
    records=records, checks=checks,
    limitation='Deliberately tied geometry isolates Mask influence; this does not estimate real assignment changes, training benefit or new contributions.')
(root / 'QUERY_MASK_ASSIGNMENT_CPU_RESULT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key: receipt[key] for key in ('status', 'matcher_calls', 'native_boxes_only_criterion_calls',
    'assignment_problems', 'formal_accuracy', 'gpu_calls')}), flush=True)
