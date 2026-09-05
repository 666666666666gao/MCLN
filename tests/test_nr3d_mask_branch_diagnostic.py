import torch

from scripts.nr3d_mask_branch_diagnostic import summarize_branches, superpoint_mask_ious


def test_superpoint_counts_match_native_point_iou_with_gaps_and_zero_logits():
    labels = torch.tensor([0, 0, 2, 2, 2, 4])
    target = torch.tensor([True, False, True, True, False, False])
    logits = torch.tensor([[1., 0., -1., 0., 0.], [0., 4., 1., -2., -1.]])
    point_masks = (logits > 0)[:, labels]
    native = ((point_masks & target).sum(1).double()
              / (point_masks | target).sum(1).double())
    assert torch.equal(superpoint_mask_ious(logits, labels, target), native)


def test_raw_mask_recovery_and_box_alignment_remain_distinct():
    ious = {'text': torch.tensor([.2, .2, .2]),
            'query': torch.tensor([.4, .8, .1]),
            'fused': torch.tensor([.3, .4, .2])}
    boxes = torch.tensor([.8, .1, .9])
    result = summarize_branches(ious, boxes, torch.tensor([True, True, False]), .75,
                               {'native_rec': 0, 'native_mask': 1, 'best_legal_box': 0})
    assert result['any_raw_query_mask_gt050'] and not result['any_fused_mask_gt050']
    assert result['all_query_oracles']['query']['query'] == 1
    assert result['good_legal_box_mask_oracles']['query']['query'] == 0
    assert result['selected_queries']['native_rec']['query'] == 0
    assert result['selected_queries']['native_mask']['query'] == 1
    empty = summarize_branches(ious, boxes, torch.zeros(3, dtype=torch.bool), .75,
                               {'native_rec': None})
    assert empty['selected_queries']['native_rec'] is None
    assert all(value is None for value in empty['good_legal_box_mask_oracles'].values())
