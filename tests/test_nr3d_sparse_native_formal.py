"""Synthetic rows through the actual native evaluator for three-arm isolation."""

import copy

import pytest
import torch

from scripts.nr3d_candidate_contract import diagnose_root_candidates
from scripts.run_nr3d_l1_native_formal_pair import selected_row, require_native_metric_parity
from scripts.run_nr3d_sparse_native_formal import require_same_selection
from src.grounding_evaluator import GroundingEvaluator
from test_nr3d_l1_native_formal_pair import predictions


def observe(outputs):
    evaluator = GroundingEvaluator(only_root=True, prefixes=['last_'], topks=[1],
        filter_non_gt_boxes=True, eval_use_selector_choice_scores=True)
    evaluator.evaluate(outputs, 'last_')
    observed = diagnose_root_candidates(outputs, evaluator)[0]
    row = selected_row(observed)
    oracle = observed['box_oracle_after_filter']
    row['legal_box_oracle_query'] = None if oracle is None else oracle['query']
    return row, evaluator.export_retrain_metrics(expected_sample_count=1)


def test_three_evaluators_keep_mask_change_separate_from_identical_grounding():
    outputs = predictions()
    protected, protected_metrics = observe(outputs)
    logits = torch.tensor([[-5., -5., 5., 5.], [5., 5., -5., -5.]])
    outputs['last_pred_masks'] = [logits.unsqueeze(0)]
    outputs['sp_last_pred_masks'] = [logits]
    sparse, sparse_metrics = observe(outputs)
    row = {'protected': protected, 'native': copy.deepcopy(protected), 'sparse': sparse}
    require_same_selection(row)
    assert protected['rec_query'] == sparse['rec_query'] == 1
    assert protected['mask_query'] == sparse['mask_query'] == 0
    assert protected['mask_iou'] == 1 and sparse['mask_iou'] == 0
    assert protected['rec_query_mask_iou'] == 0 and sparse['rec_query_mask_iou'] == 1
    for arm, metric in [('protected', protected_metrics), ('native', protected_metrics), ('sparse', sparse_metrics)]:
        require_native_metric_parity([row], arm, metric)
    with pytest.raises(AssertionError):
        require_native_metric_parity([row], 'sparse', protected_metrics)


def test_no_legal_rec_row_still_counts_mask_and_cannot_change_filtering():
    outputs = predictions()
    outputs['last_center'][0, 1, 0] = 10
    observed, metric = observe(outputs)
    row = {arm: copy.deepcopy(observed) for arm in ['protected', 'native', 'sparse']}
    require_same_selection(row)
    assert observed['rec_query'] is None and observed['candidate_count'] == 0
    result = require_native_metric_parity([row], 'sparse', metric)
    assert result['sample_count'] == 1 and result['rec_hits025'] == 0 and result['mask_hits025'] == 1
    row['sparse']['candidate_count'] = 1
    with pytest.raises(AssertionError):
        require_same_selection(row)
