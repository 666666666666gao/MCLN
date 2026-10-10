"""Controlled CPU fixtures for the unchanged deployed Hungarian matcher.

Synthetic inputs are an engineering check, not dataset evaluation.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch

root = Path(__file__).resolve().parent
spec = json.loads((root / 'CHECK_SPEC.json').read_bytes())
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
torch.set_num_threads(1)
source = Path(spec['warm_source'])
for name, digest in spec['source_sha256'].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
sys.path.insert(0, str(source))
module_spec = importlib.util.spec_from_file_location('actual_native_losses', source / 'models/losses.py')
losses = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(losses)
assert losses.__file__ == str(source / 'models/losses.py')
import_bindings = {}
for module_name, name in (('utils', 'utils/__init__.py'), ('utils.scatter_util', 'utils/scatter_util.py')):
    path = Path(sys.modules[module_name].__file__)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert path == source / name and digest == spec['import_binding_sha256'][name]
    import_bindings[module_name] = dict(path=str(path), sha256=digest)
matcher = losses.HungarianMatcher(1, 0, 2, True)
assert (matcher.cost_class, matcher.cost_bbox, matcher.cost_giou, matcher.cost_masks) == (1, 0, 2, 0.0002)
original_assignment = losses.linear_sum_assignment
captured = []


def capture_assignment(matrix):
    captured.append(matrix.detach().cpu().numpy().copy())
    return original_assignment(matrix)


losses.linear_sum_assignment = capture_assignment


def fixture(gt_counts):
    batch = len(gt_counts)
    boxes = torch.zeros(batch, 256, 6)
    boxes[..., :3] = 30
    boxes[..., 3:] = 1
    logits = torch.zeros(batch, 256, 256)
    text_masks, own_masks, targets = [], [], []
    for b, count in enumerate(gt_counts):
        target_boxes = torch.tensor([[float(4 * k), 0., 0., 1., 1., 1.] for k in range(count)])
        for k in range(count):
            boxes[b, 3 + 14 * k] = target_boxes[k]
        text = torch.tensor([2., 2., -2., -2.])
        text_masks.append(text.view(1, 1, 4).expand(1, 256, 4))
        own = torch.full((256, 4), -2.)
        own[200, :2] = 2.
        own[201, 2:] = 2.
        own_masks.append(own)
        pmap = torch.zeros(count, 256)
        pmap[:, 0] = 1.
        target_mask = torch.stack([torch.tensor([1., 1., 0., 0.]) if k == 0 else torch.tensor([0., 0., 1., 1.]) for k in range(count)])
        targets.append(dict(labels=torch.zeros(count, dtype=torch.int64), boxes=target_boxes, masks=target_mask, positive_map=pmap))
    output = dict(pred_logits=logits, pred_boxes=boxes, pred_masks=text_masks,
                  sp_pred_masks=own_masks, adaptive_weights=[torch.tensor(.5) for _ in gt_counts],
                  superpoints=[torch.arange(4) for _ in gt_counts])
    return output, targets


records = []


def run(name, gt_counts, mutation):
    output, target = fixture(gt_counts)
    if mutation == 'swap_own':
        output['sp_pred_masks'] = [value.flip(0) for value in output['sp_pred_masks']]
    elif mutation == 'change_shared_text':
        output['pred_masks'] = [torch.full_like(value, -2.) for value in output['pred_masks']]
    elif mutation == 'remove_mask':
        del output['pred_masks']
    captured.clear()
    result = matcher(output, target)
    assert len(captured) == len(gt_counts)
    assigned = [dict(query=pair[0].tolist(), gt=pair[1].tolist()) for pair in result]
    for b, count in enumerate(gt_counts):
        assert assigned[b] == dict(query=[3 + 14 * k for k in range(count)], gt=list(range(count)))
    records.append(dict(name=name, gt_counts=gt_counts, mutation=mutation, assignments=assigned))
    return [matrix.copy() for matrix in captured]


single_base = run('single_base', [1], 'none')
single_own = run('single_swapped_own', [1], 'swap_own')
single_text = run('single_changed_shared_text', [1], 'change_shared_text')
single_without = run('single_no_mask_key', [1], 'remove_mask')
multi_base = run('two_gt_base', [2], 'none')
multi_own = run('two_gt_swapped_own', [2], 'swap_own')
multi_text = run('two_gt_changed_shared_text', [2], 'change_shared_text')
batch_base = run('mixed_batch_base', [1, 2], 'none')
batch_own = run('mixed_batch_swapped_own', [1, 2], 'swap_own')
checks = []
for name, baseline, changed in (
        ('single_own', single_base, single_own), ('multi_own', multi_base, multi_own),
        ('mixed_batch_own', batch_base, batch_own)):
    differences = [float(abs(a - b).max()) for a, b in zip(baseline, changed)]
    assert all(value == 0 for value in differences)
    checks.append(dict(name=name, maximum_native_cost_difference=differences))
for name, baseline, changed in (
        ('single_shared_text', single_base, single_text),
        ('single_mask_component', single_without, single_base),
        ('two_gt_shared_text', multi_base, multi_text)):
    for b, (a, changed_matrix) in enumerate(zip(baseline, changed)):
        difference = changed_matrix - a
        spreads = difference.max(axis=0) - difference.min(axis=0)
        # Float32 addition can round the same column constant differently.
        assert float(abs(spreads).max()) < 1e-6
        checks.append(dict(name=name, batch_index=b,
                           first_row_column_shift=difference[0].tolist(),
                           maximum_column_shift_spread=float(abs(spreads).max())))
losses.linear_sum_assignment = original_assignment
assert not torch.cuda.is_initialized()
receipt = dict(status='ACTUAL_NATIVE_MATCHER_CONTROLLED_CPU_CHECK_COMPLETE',
    evaluation_type='simulation_only', fixture_count=9, matcher_calls=9,
    neural_model_forward_calls=0, criterion_calls=0, optimizer_updates=0,
    dataset_rows=0, formal_accuracy=None, gpu_calls=0,
    active_training_source_mutations=0, current_training_queries=0,
    source_sha256=spec['source_sha256'], actual_losses_path=losses.__file__,
    actual_import_bindings=import_bindings,
    torch_version=torch.__version__, records=records, checks=checks,
    scope='Native matcher behavior with deliberately constructed CPU inputs; no causal training or benchmark claim.')
(root / 'MATCHER_CPU_RESULT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(status=receipt['status'], fixture_count=9, matcher_calls=9,
                     formal_accuracy=None, gpu_calls=0)), flush=True)
