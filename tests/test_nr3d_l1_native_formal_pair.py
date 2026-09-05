"""CPU parity with the actual evaluator; all examples are synthetic."""

import pytest
import torch

from scripts.nr3d_candidate_contract import diagnose_root_candidates
from scripts.run_nr3d_l1_native_formal_pair import require_native_metric_parity, selected_row
from src.grounding_evaluator import GroundingEvaluator


def predictions():
    positive = torch.zeros(1, 1, 256)
    positive[0, 0, 0] = 1
    logits = torch.tensor([[5., 5., -5., -5.], [-5., -5., 5., 5.]])
    scores = torch.tensor([[.9, .8]])
    outputs = {
        'positive_map': positive,
        'center_label': torch.zeros(1, 1, 3), 'size_gts': torch.ones(1, 1, 3),
        'box_label_mask': torch.ones(1, 1, dtype=torch.bool),
        'last_center': torch.tensor([[[5., 0., 0.], [0., 0., 0.]]]),
        'last_pred_size': torch.ones(1, 2, 3),
        'last_sem_cls_scores': torch.tensor([[[4., 0.], [3., 0.]]]),
        'selected_source_scores': scores,
        'source_choice_source_scores': {'default': scores},
        'selector_choice_source_names': ['default'],
        'proj_tokens': torch.tensor([[[1., 0.], [0., 1.]]]),
        'last_proj_queries': torch.tensor([[[1., 0.], [0., 1.]]]),
        'all_detected_boxes': torch.tensor([[[0., 0., 0., 1., 1., 1.]]]),
        'all_detected_bbox_label_mask': torch.ones(1, 1, dtype=torch.bool),
        'last_pred_masks': [logits.unsqueeze(0)], 'sp_last_pred_masks': [logits],
        'adaptive_weights': [torch.tensor(.5)],
        'superpoints': torch.tensor([[0, 1, 2, 3]]),
        'gt_masks': torch.tensor([[[1, 1, 0, 0]]]),
        'is_unique': torch.tensor([True]), 'is_hard': torch.tensor([False]),
        'is_view_dep': torch.tensor([False]),
    }
    for name in ['modify_positive_map', 'pron_positive_map', 'other_entity_map',
                 'auxi_entity_positive_map', 'rel_positive_map']:
        outputs[name] = torch.zeros_like(positive)
    return outputs


def observe(outputs):
    evaluator = GroundingEvaluator(only_root=True, prefixes=['last_'], topks=[1],
                                  filter_non_gt_boxes=True, eval_use_selector_choice_scores=True)
    evaluator.evaluate(outputs, 'last_')
    row = selected_row(diagnose_root_candidates(outputs, evaluator)[0])
    return row, evaluator.export_retrain_metrics(expected_sample_count=1)


def test_native_parity_preserves_distinct_rec_filter_and_mask_query():
    row, native = observe(predictions())
    assert row['rec_query'] == 1 and row['mask_query'] == 0
    assert row['rec_box_iou'] == row['mask_iou'] == 1.0
    assert row['rec_query_mask_iou'] == row['legal_box_oracle_query_mask_iou'] == 0.0
    assert row['candidate_profiles']['before_filter']['top_query'] == 0
    assert row['candidate_profiles']['after_filter']['top_query'] == 1
    summary = require_native_metric_parity([{'position': row}], 'position', native)
    assert summary == {'sample_count': 1, 'rec_hits025': 1, 'rec_hits050': 1,
                       'mask_hits025': 1, 'mask_hits050': 1, 'mask_iou_sum': 1.0}


def test_native_parity_counts_a_row_with_no_legal_rec_candidate():
    outputs = predictions()
    outputs['last_center'][0, 1, 0] = 10
    row, native = observe(outputs)
    assert row['rec_query'] is None and row['candidate_count'] == 0
    summary = require_native_metric_parity([{'position': row}], 'position', native)
    assert summary['rec_hits025'] == summary['rec_hits050'] == 0
    assert summary['mask_hits025'] == summary['mask_hits050'] == 1


def test_native_parity_rejects_a_metric_from_the_other_arm():
    outputs = predictions()
    old, old_native = observe(outputs)
    outputs['selected_source_scores'] = torch.tensor([[.8, .9]])
    new, new_native = observe(outputs)
    rows = [{'protected': old, 'position': new}]
    require_native_metric_parity(rows, 'protected', old_native)
    require_native_metric_parity(rows, 'position', new_native)
    assert new['mask_query'] == 1 and new['mask_iou'] == 0
    with pytest.raises(AssertionError):
        require_native_metric_parity(rows, 'position', old_native)
