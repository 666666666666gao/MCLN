"""Check the CE replacement against the repository's actual native method.

Uses synthetic tensors; this is not a real-data training or accuracy result.
AST extraction avoids importing CUDA extensions for a loss-only CPU check.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    module_path = args.source / 'models/nr3d_semantic_assignment.py'
    spec = importlib.util.spec_from_file_location('assignment', str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse((args.source / 'models/losses.py').read_text())
    native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SetCriterion')
    methods = [n for n in native.body if isinstance(n, ast.FunctionDef)
               and n.name in ('loss_pos_align', '_get_src_permutation_idx')]
    assert len(methods) == 2
    shell = ast.parse('class NativeCriterion:\n    pass\n')
    shell.body[0].body = methods
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(shell), 'actual_native_ce', 'exec'), namespace)
    criterion = namespace['NativeCriterion']()
    criterion.eos_coef = .1
    torch.manual_seed(2027)
    logits = torch.randn(3, 6, 8, dtype=torch.float64, requires_grad=True)
    boxes = torch.tensor([[[0., 0., 0., 2., 2., 2.]] * 6] * 3,
                         dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        boxes[:, 3, 3] = 4.  # IoU exactly .5: not eligible.
        boxes[:, 4:, 0] = 10.
        boxes[2, 2, 0] = 10.  # Nr3D row with no eligible unmatched box.
    targets = []
    for _ in range(3):
        target = {'boxes': torch.tensor([[0., 0., 0., 2., 2., 2.],
                                         [8., 0., 0., 2., 2., 2.]], dtype=torch.float64)}
        for index, key in enumerate(('positive_map', 'modify_positive_map',
                                     'pron_positive_map', 'rel_positive_map', 'other_entity_map')):
            target[key] = torch.zeros(2, 8, dtype=torch.float64)
            target[key][:, index] = 1.
        targets.append(target)
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1])) for _ in range(3)]
    outputs = {'pred_logits': logits, 'pred_boxes': boxes, 'language_dataset': ['nr3d'] * 3}
    datasets = ['nr3d', 'scannet', 'nr3d']
    denominator = torch.tensor([6.], dtype=logits.dtype)
    delta, selected = module.semantic_assignment_delta(outputs, targets, indices, denominator, datasets, .1)
    assert selected.nonzero().tolist() == [[0, 2]]
    native_loss = criterion.loss_pos_align(outputs, targets, indices, denominator, None)['loss_ce']
    labels = torch.zeros_like(logits)
    labels[..., -1] = 1
    weights = torch.full(logits.shape[:2], .1, dtype=torch.float32)
    for bid, (queries, tids) in enumerate(indices):
        root = sum(targets[bid][key][tids] * weight for key, weight in
                   [('positive_map', .6), ('modify_positive_map', .2),
                    ('pron_positive_map', .2), ('rel_positive_map', .1)])
        labels[bid, queries] = root
        weights[bid, queries] = 1
    labels[0, 2] = labels[0, 0]
    expected = (((labels * torch.log(labels + 1e-6) - labels * logits.log_softmax(-1)).sum(-1)) * weights).sum() / denominator
    corrected = native_loss + delta
    assert torch.allclose(expected, corrected, atol=1e-8, rtol=1e-8), {
        'expected': float(expected), 'corrected': float(corrected),
        'native': float(native_loss), 'delta': float(delta)}
    grad = torch.autograd.grad(corrected, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    assert torch.allclose(grad, expected_grad, atol=1e-8, rtol=1e-8)
    change = torch.autograd.grad(delta, logits, retain_graph=True)[0]
    assert bool((change[~selected] == 0).all())
    assert torch.autograd.grad(delta, boxes, allow_unused=True, retain_graph=True)[0] is None
    hook = module.LastLayerSemanticAssignment(criterion)
    hook.bind({'last_sem_cls_scores': logits}, datasets)
    hooked = criterion.loss_pos_align(outputs, targets, indices, denominator, None)['loss_ce']
    assert torch.equal(hooked, corrected) and len(hook.records) == 1
    earlier = dict(outputs, pred_logits=logits.clone())
    unchanged = criterion.loss_pos_align(earlier, targets, indices, denominator, None)['loss_ce']
    assert torch.equal(unchanged, native_loss) and len(hook.records) == 1
    hook.remove()
    assert torch.equal(criterion.loss_pos_align(outputs, targets, indices, denominator, None)['loss_ce'], native_loss)
    empty_delta, empty_selected = module.semantic_assignment_delta(outputs, targets, indices, denominator, ['scannet'] * 3, .1)
    assert not bool(empty_selected.any()) and float(empty_delta) == 0
    report = {'status': 'pass', 'synthetic_only': True, 'optimizer_steps': 0,
              'native_ce_method_extracted_from': str(args.source / 'models/losses.py'),
              'qualified_indices': selected.nonzero().tolist(),
              'explicit_relabel_loss_error': float((expected - corrected).abs()),
              'explicit_relabel_gradient_error': float((grad - expected_grad).abs().max()),
              'other_gt_and_detection_prompts_unchanged': True,
              'iou_threshold_strict': True, 'geometry_detached': True,
              'earlier_layers_unchanged': True, 'hook_removed_restores_native': True}
    args.receipt.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
