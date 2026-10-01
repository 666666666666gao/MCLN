"""CPU fixtures for fixed-prediction replay, not ScanRefer performance."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.cs_mcln_modules import GeometryEvidenceReadback
from models.modules import ClsAgnosticPredictHead


diagnostic = Path('/root/autodl-tmp/cs_mcln_readback_diagnostic_20261001_v1/diagnose_cs_mcln_readback.py')
spec = importlib.util.spec_from_file_location('paired_diagnostic', diagnostic)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.manual_seed(2027)
torch.set_num_threads(2)
readback = GeometryEvidenceReadback().eval().requires_grad_(False)
head = ClsAgnosticPredictHead(256, 1, 256, seed_feat_dim=288).eval().requires_grad_(False)
model = SimpleNamespace(prediction_heads=[head])
query = torch.randn(2, 256, 288)
evidence = torch.randn(2, 256, 306)
evidence[..., -9:-6] = evidence[..., -9:-6].abs() + 0.1
buffers = {name: value.clone() for name, value in head.named_buffers()}

with torch.no_grad():
    outputs = {
        'last_center': torch.randn(2, 256, 3),
        'last_pred_size': torch.rand(2, 256, 3) + 0.1,
        'last_query_masks': [torch.randn(256, 40), torch.randn(256, 51)],
    }
    after = readback(query, evidence)
    assert torch.equal(after, query)
    head.write_semantic_scores(after.transpose(1, 2).contiguous(), outputs, 'last_')
    bypass = module.bypass_readback(model, {'before': query, 'after': after}, outputs)
    assert torch.equal(outputs['last_sem_cls_scores'], bypass['last_sem_cls_scores'])

    readback.output.weight.normal_(std=0.02)
    after = readback(query, evidence)
    head.write_semantic_scores(after.transpose(1, 2).contiguous(), outputs, 'last_')
    bypass = module.bypass_readback(model, {'before': query, 'after': after}, outputs)
    assert not torch.equal(outputs['last_sem_cls_scores'], bypass['last_sem_cls_scores'])
    expected = {}
    head.write_semantic_scores(query.transpose(1, 2).contiguous(), expected, 'last_')
    assert torch.equal(bypass['last_sem_cls_scores'], expected['last_sem_cls_scores'])
    assert all(bypass[name] is value for name, value in outputs.items()
               if name != 'last_sem_cls_scores')

    # Execute the real native evaluator; capture its own score tensors.
    for key in ('positive_map', 'modify_positive_map', 'pron_positive_map',
                'other_entity_map', 'auxi_entity_positive_map', 'rel_positive_map'):
        outputs[key] = torch.rand(2, 1, 256)
    outputs['center_label'] = torch.zeros(2, 1, 3)
    outputs['size_gts'] = torch.ones(2, 1, 3)
    outputs['box_label_mask'] = torch.ones(2, 1)
    evaluator = module.ScoreRecordingEvaluator(
        only_root=True, thresholds=[0.25, 0.5], topks=[1],
        prefixes=['last_'], filter_non_gt_boxes=False, model='MCLN',
        eval_use_selector_choice_scores=False,
    )
    evaluator.batch_scores, evaluator.batch_rankings = [], []
    evaluator.evaluate_bbox_by_pos_align(outputs, 'last_')
    assert len(evaluator.batch_scores) == len(evaluator.batch_rankings) == 2
    probabilities = outputs['last_sem_cls_scores'].softmax(-1)
    for bid in range(2):
        terms = []
        for key in ('positive_map', 'modify_positive_map', 'pron_positive_map',
                    'rel_positive_map', 'other_entity_map'):
            mapping = outputs[key][bid]
            if key == 'positive_map':
                mapping = (mapping > 0).to(mapping.dtype)
            terms.append((probabilities[bid].unsqueeze(0) * mapping.unsqueeze(1)).sum(-1))
        expected_scores = terms[0] + terms[1] + terms[2] + terms[3] - terms[4]
        assert torch.equal(evaluator.batch_scores[bid], expected_scores)
        assert torch.equal(evaluator.batch_rankings[bid], expected_scores.argsort(1, True))

assert all(torch.equal(value, buffers[name]) for name, value in head.named_buffers())
assert all(parameter.grad is None for parameter in head.parameters())
assert all(parameter.grad is None for parameter in readback.parameters())
assert not torch.cuda.is_initialized()
print(json.dumps({
    'fixture': 'CPU synthetic fixed-prediction semantic replay and native score capture',
    'zero_readback_equal': True, 'nonzero_readback_changes_semantic_only': True,
    'native_per_row_scores_and_full256_rank_exact': True,
    'head_running_buffers_unchanged': True, 'optimizer_steps': 0,
    'cuda_initialized': False, 'real_dataset_forward': False,
}), flush=True)
