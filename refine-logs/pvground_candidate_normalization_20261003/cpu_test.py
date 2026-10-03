"""Exercise semantic replacement using the actual native criterion methods."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import torch

from pvground_candidate_consistency import candidate_consistency_correction, verify_consistency_replacement

parser = argparse.ArgumentParser()
parser.add_argument('--native-loss', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
tree = ast.parse(args.native_loss.read_text())
owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
             and any(isinstance(member, ast.FunctionDef) and member.name == 'loss_sem_align'
                     for member in node.body))
methods = {member.name: member for member in owner.body if isinstance(member, ast.FunctionDef)}
scope = {'torch': torch}
for name in ('loss_sem_align', '_get_src_permutation_idx'):
    module = ast.Module(body=[methods[name]])
    exec(compile(module, str(args.native_loss), 'exec'), scope)
native = SimpleNamespace(temperature=.07, eos_coef=.1)
native._get_src_permutation_idx = lambda indices: scope['_get_src_permutation_idx'](native, indices)
native.loss_sem_align = lambda *values: scope['loss_sem_align'](native, *values)
torch.manual_seed(2027)
records = []
for include_qualified in (False, True):
    batch = {'language_dataset': ['scanrefer', 'scanrefer'],
             'box_label_mask': torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
             'center_label': torch.tensor([[[0., 0., 0.], [3., 0., 0.], [0., 0., 0.]],
                                           [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]),
             'size_gts': torch.ones(2, 3, 3)}
    names = ('positive_map', 'modify_positive_map', 'pron_positive_map', 'other_entity_map', 'rel_positive_map')
    for name in names:
        batch[name] = torch.zeros(2, 3, 8)
    for name, token in zip(names, (1, 2, 3, 4, 5)):
        batch[name][:, 0, token] = 1
    batch['positive_map'][0, 1, 6] = 1
    centers = torch.full((2, 6, 3), 5.)
    centers[:, 0] = 0
    centers[0, 3] = 0  # High overlap but already matched to another GT: protected.
    if include_qualified:
        centers[:, 1] = 0
        centers[:, 2] = .01
    predictions = {'last_center': centers.requires_grad_(),
                   'last_pred_size': torch.ones(2, 6, 3, requires_grad=True),
                   'last_proj_queries': torch.randn(2, 6, 4, requires_grad=True),
                   'proj_tokens': torch.randn(2, 8, 4, requires_grad=True),
                   'tokenized': {'attention_mask': torch.ones(2, 8, dtype=torch.long)}}
    indices = [(torch.tensor([0, 3]), torch.tensor([0, 1])),
               (torch.tensor([0]), torch.tensor([0]))]
    targets = [{name: batch[name][bid][batch['box_label_mask'][bid]] for name in names}
               for bid in range(2)]
    outputs = {'proj_queries': predictions['last_proj_queries'],
               'proj_tokens': predictions['proj_tokens'], 'tokenized': predictions['tokenized']}
    old = native.loss_sem_align(outputs, targets, indices, 3, None)['loss_sem_align']
    predictions['last__loss_sem_align'] = old
    preserved = [(queries.clone(), targets.clone()) for queries, targets in indices]
    correction, statistics, selected = candidate_consistency_correction(predictions, batch, indices, native)
    assert not bool(selected[0, 3])
    assert all(torch.equal(q, pq) and torch.equal(t, pt)
               for (q, t), (pq, pt) in zip(indices, preserved))
    assert statistics['contrastive_native_denominator'] == 3
    assert statistics['contrastive_expanded_denominator'] == 3 + int(selected.sum())
    corrected = old * (.5 / 7) + correction
    if not include_qualified:
        assert not bool(selected.any()) and float(correction) == 0
        before_gradient = torch.autograd.grad(old * (.5 / 7), predictions['last_proj_queries'], retain_graph=True)[0]
        after_gradient = torch.autograd.grad(corrected, predictions['last_proj_queries'], retain_graph=True)[0]
        assert torch.equal(before_gradient, after_gradient)
    else:
        assert int(selected.sum()) == 4
        expanded = [(torch.tensor([0, 3, 1, 2]), torch.tensor([0, 1, 0, 0])),
                    (torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0]))]
        expected = native.loss_sem_align(outputs, targets, expanded, 7, None)['loss_sem_align'] * (.5 / 7)
        assert torch.allclose(corrected, expected, rtol=1e-6, atol=1e-6)
        actual_gradient = torch.autograd.grad(corrected, predictions['last_proj_queries'], retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, predictions['last_proj_queries'], retain_graph=True)[0]
        assert torch.allclose(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-7)
    witness = verify_consistency_replacement(predictions, batch, indices, native, correction, selected)
    records.append(dict(qualified_case=include_qualified, **statistics, **witness))
receipt = dict(status='pass', native_method_source=str(args.native_loss), fixture_count=2,
               no_qualified_loss_and_gradient_exact=True, protected_other_match=True,
               regression_matching_unchanged=True, expanded_native_loss_and_gradient_allclose=True,
               CPU_only=True, model_forward=False, expanded_count_normalization=True, fixtures=records)
args.output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
