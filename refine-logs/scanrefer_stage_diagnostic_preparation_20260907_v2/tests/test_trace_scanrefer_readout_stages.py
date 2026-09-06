import copy

import pytest
import torch

from models.rec_geometry_reranker import build_deployed_parent_state, blend_rec_geometry_scores
from models.rec_pareto_contextual_hierarchy import apply_pareto_contextual_policy
from scripts.trace_scanrefer_readout_stages import STAGES, trace_readout_stages


def fixture(proposal=(3., 3.), tie=False, remove_first_geometry=False):
    boxes = torch.zeros(1, 256, 6)
    boxes[0, :, 0] = torch.arange(256)
    boxes[..., 3:] = 1.
    indices = torch.full((1, 16), -1, dtype=torch.long)
    indices[0, :2] = torch.tensor([12, 5])
    valid = torch.zeros(1, 16, dtype=torch.bool)
    valid[0, :2] = True
    scores = torch.zeros(1, 16)
    scores[0, 0] = 1.
    scores[0, 1] = float(tie)
    parent_state = build_deployed_parent_state(scores, indices, valid, 256)
    geometry_valid = torch.zeros(1, 16, 7, dtype=torch.bool)
    geometry_valid[0, :2, :2] = True
    if remove_first_geometry:
        geometry_valid[0, 0] = False
    geometry_parent = build_deployed_parent_state(scores, indices, geometry_valid.any(dim=2), 256)
    geometry_logits = torch.zeros(1, 112)
    geometry_logits[0, [0, 1, 7, 8]] = torch.tensor([3., 2., 1., 0.])
    geometry = blend_rec_geometry_scores(geometry_parent, geometry_logits, geometry_valid, .7, 0)
    query_logits = torch.full((1, 16, 2), -20.)
    query_logits[0, 0] = 8.
    variant_logits = torch.full((1, 16, 7, 2), -20.)
    variant_logits[0, 0, 0] = 0.
    variant_logits[0, 0, 1] = torch.tensor(proposal)
    policy = apply_pareto_contextual_policy(
        geometry['flat_scores'], query_logits, variant_logits,
        geometry_valid.any(dim=2), geometry_valid, aggregate_margin=.1)
    variant_boxes = torch.zeros(1, 112, 6)
    variant_boxes[..., 3:] = 1.
    variant_boxes[0, :7, 0] = 12. + torch.arange(7) * .01
    variant_boxes[0, 7:14, 0] = 5. + torch.arange(7) * .01
    candidate = {'num_queries': 256, 'query_indices': indices, 'valid_mask': valid}
    readout = {
        'parent': {'candidate_batch': candidate, 'compact_scores': scores,
                   'query_scores': parent_state['query_scores']},
        'continuous': {'geometry_valid': geometry_valid.reshape(1, 112),
                       'geometry': {'ranking_logits': geometry_logits},
                       'v99': {'query_logits': query_logits, 'variant_logits': variant_logits}},
        'runtime': {'rec_geometry_runtime_mode': 'flat_geometry_axis',
                    'rec_geometry_boxes': variant_boxes,
                    'rec_geometry_valid_mask': geometry_valid.reshape(1, 112),
                    'rec_geometry_scores': policy['scores']},
    }
    metadata = {'geometry': {'geometry_weight': .7, 'regressed_variant_index': 0},
                'v99': {'schema': 'rec-pareto-contextual-hierarchical-v1',
                        'policy': {'aggregate_margin': .1}}}
    return boxes, torch.tensor([5]), readout, metadata


@pytest.mark.parametrize('proposal, final_variant, accepted', [
    ((3., 3.), 1, True), ((3., -12.), 0, False),
])
def test_keeps_ungated_proposal_separate_from_deployed_selection(proposal, final_variant, accepted):
    result = trace_readout_stages(*fixture(proposal))
    assert result['stage_names'] == STAGES
    assert result['query_indices'].tolist() == [[5, 12, 12, 12, 12, 12]]
    assert result['variant_indices'].tolist() == [[-1, -1, -1, 0, 1, final_variant]]
    assert result['pareto_pass'].tolist() == [accepted]
    assert result['boxes'][0, 0, 0] == 5.
    assert result['boxes'][0, -1, 0] == pytest.approx(12. + final_variant * .01)


def test_parent_tie_uses_global_query_id_instead_of_compact_position():
    result = trace_readout_stages(*fixture(tie=True))
    assert result['query_indices'][0, 1:3].tolist() == [5, 5]
    assert result['boxes'][0, 1, 0] == 5.


def test_geometry_validity_change_has_its_own_parent_stage():
    result = trace_readout_stages(*fixture(remove_first_geometry=True))
    assert result['query_indices'][0, 1:3].tolist() == [12, 5]
    assert result['query_indices'][0, 3:].tolist() == [5, 5, 5]
    assert result['final_flat_indices'].tolist() == [7]
    assert result['variant_indices'][0, -1] == 0


def test_rejects_score_drift_even_if_winning_index_is_unchanged():
    values = fixture()
    scores = values[2]['runtime']['rec_geometry_scores']
    original_choice = scores.argmax(dim=1).clone()
    scores[0, original_choice.item()] += .01
    assert torch.equal(original_choice, scores.argmax(dim=1))
    with pytest.raises(AssertionError):
        trace_readout_stages(*values)


def test_trace_does_not_modify_inputs_or_create_a_gradient_path():
    values = fixture()
    values[0].requires_grad_(True)
    before = copy.deepcopy(values)
    result = trace_readout_stages(*values)
    assert not result['boxes'].requires_grad
    assert torch.equal(values[0], before[0])
    for group in ['parent', 'continuous', 'runtime']:
        for key, value in values[2][group].items():
            if torch.is_tensor(value):
                assert torch.equal(value, before[2][group][key])
    assert not result['persistent_instance_identity_available']
