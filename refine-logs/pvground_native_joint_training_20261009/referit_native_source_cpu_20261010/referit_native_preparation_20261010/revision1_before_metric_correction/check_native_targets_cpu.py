"""Synthetic CPU CE/mask target checks, not real dataset or model evaluation."""
import importlib.util
import json
from pathlib import Path
import sys

import torch

root = Path(__file__).resolve().parent
bundle = json.loads((root / 'CPU_INPUTS.json').read_bytes())
sys.path.insert(0, bundle['native_model_source'])
from models.losses import SetCriterion
sys.path.insert(0, str(root / 'source'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old_g = load('old_scan_g', root / 'original/pvground_semantic_assignment.py')
old_c = load('old_scan_c', root / 'original/selected_query_mask_objective.py')
new_g = load('new_referit_g', root / 'source/pvground_semantic_assignment.py')
new_c = load('new_referit_c', root / 'source/selected_query_mask_objective.py')
torch.set_num_threads(1)
torch.manual_seed(2027)
criterion = SetCriterion(matcher=None, losses={}, eos_coef=.1)
results = []


def panel(language, count):
    logits = torch.randn(count, 5, 8)
    logits[:, 4, 0] = 12
    logits[:, 4, 1:] = -12
    logits.requires_grad_()
    centers = torch.zeros(count, 5, 3)
    centers[:, 1] = 4
    centers[:, 3:] = 8
    centers.requires_grad_()
    sizes = torch.full((count, 5, 3), 2.0, requires_grad=True)
    batch = dict(language_dataset=[language] * count,
        sample_dataset=[language] * 2 + (['scannet'] if count == 3 else []),
        box_label_mask=torch.ones(count, 2, dtype=torch.bool),
        center_label=torch.zeros(count, 2, 3), size_gts=torch.full((count, 2, 3), 2.0))
    batch['center_label'][:, 1] = 4
    for name, token in [('positive_map', 0), ('modify_positive_map', 2),
                        ('pron_positive_map', 3), ('rel_positive_map', 4)]:
        value = torch.zeros(count, 2, 8)
        value[:, 0, token] = .5
        value[:, 1, 1] = .5
        batch[name] = value
    batch['other_entity_map'] = torch.zeros(count, 2, 8)
    masks = torch.zeros(count, 2, 8)
    masks[:, 0, :2] = 1
    masks[:, 1, 2:4] = 1
    targets = [dict(boxes=torch.cat([batch['center_label'][bid], batch['size_gts'][bid]], -1),
        masks=masks[bid], **{name:batch[name][bid] for name in
        ('positive_map','modify_positive_map','pron_positive_map','rel_positive_map','other_entity_map')})
        for bid in range(count)]
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1])),
               (torch.tensor([0, 4]), torch.tensor([0, 1]))]
    if count == 3:
        indices.append((torch.tensor([0, 1]), torch.tensor([0, 1])))
    own = [torch.randn(5, 4, requires_grad=True) for _ in range(count)]
    text = [torch.randn(5, 4, requires_grad=True) for _ in range(count)]
    predictions = dict(last_sem_cls_scores=logits, last_center=centers, last_pred_size=sizes,
        sp_last_pred_masks=own, last_pred_masks=[[value] for value in text],
        adaptive_weights=[torch.tensor(.3, requires_grad=True) for _ in range(count)],
        superpoints=[torch.tensor([0,0,1,1,2,2,3,3]) for _ in range(count)])
    output = dict(pred_logits=logits, language_dataset=batch['language_dataset'])
    ce = criterion.loss_pos_align(output, targets, indices, 2 * count, None)['loss_ce']
    predictions['last__loss_ce'] = ce
    return predictions, batch, targets, indices


p, b, targets, indices = panel('scanrefer', 2)
old_g_loss, _ = old_g.semantic_assignment_correction(p, b, indices, .1)
new_g_loss, _ = new_g.semantic_assignment_correction(p, b, indices, .1, 6)
assert torch.equal(old_g_loss, new_g_loss)
old_g_gradient = torch.autograd.grad(old_g_loss, p['last_sem_cls_scores'], retain_graph=True)[0]
new_g_gradient = torch.autograd.grad(new_g_loss, p['last_sem_cls_scores'], retain_graph=True)[0]
assert torch.equal(old_g_gradient, new_g_gradient)
old_c_loss, old_c_record = old_c.selected_query_mask_loss(p, b, indices, targets)
new_c_loss, new_c_record = new_c.selected_query_mask_loss(p, b, indices, targets)
assert torch.equal(old_c_loss, new_c_loss) and old_c_record['extra_rows'] == new_c_record['extra_rows'] == 1
for language in ('scanrefer', 'nr3d', 'sr3d'):
    p, b, targets, indices = panel(language, 3)
    correction, record = new_g.semantic_assignment_correction(p, b, indices, .1, 6)
    native_check = new_g.verify_native_replacement(p, b, indices, .1, correction, 6)
    qualified = new_g.qualified_unmatched(p, b, indices)
    assert bool(qualified[:2, 2].all()) and not bool(qualified[2].any())
    change_gradient = torch.autograd.grad(correction, p['last_sem_cls_scores'], retain_graph=True)[0]
    assert bool((change_gradient[2] == 0).all())
    for bid, (queries, _) in enumerate(indices):
        assert bool((change_gradient[bid, queries] == 0).all())
    extra, mask_record = new_c.selected_query_mask_loss(p, b, indices, targets)
    assert mask_record['extra_rows'] == 1
    assert [row['role'] for row in mask_record['rows']] == ['unmatched', 'matched_other', 'detection']
    gradients = torch.autograd.grad(extra, p['sp_last_pred_masks'], retain_graph=True, allow_unused=True)
    assert bool((gradients[0][4].abs().sum() > 0))
    assert bool((gradients[0][:4] == 0).all())
    assert gradients[1] is None and gradients[2] is None
    text_and_alpha = torch.autograd.grad(extra, p['last_pred_masks'][0] + p['adaptive_weights'][:1], retain_graph=True)
    assert all(bool(value.abs().sum() > 0) for value in text_and_alpha)
    expected_root = torch.tensor([.3125,0,.0625,.0625,.0625,0,0,0]) if language == 'sr3d' else torch.tensor([.3,0,.1,.1,.05,0,0,0])
    assert torch.equal(new_g.root_token_target(b)[0], expected_root)
    results.append(dict(language=language, CE_correction=float(correction), extra_mask_loss=float(extra),
        native_CE_and_gradient_check=native_check, protected_detection_and_all_matches=True,
        selected_unmatched_mask_has_gradient=True, extra_rows=1))

print(json.dumps(dict(status='CPU_SYNTHETIC_NATIVE_TARGET_CHECKS_PASS',
    torch_version=torch.__version__, CUDA_visible_devices=__import__('os').environ['CUDA_VISIBLE_DEVICES'],
    seed=2027, rows_per_language=3, candidate_count=5, token_count=8, point_count=8,
    evaluation_type='synthetic_target_and_gradient_engineering_not_accuracy',
    criterion_source=str(Path(bundle['native_model_source']) / 'models/losses.py'),
    ScanRefer_original_loss_and_gradient_values_identical=True, results=results,
    full_PV_factory_checked=False, author_checkpoints_loaded=False,
    real_dataset_rows=0, full_goal_complete=False)), flush=True)
