"""Read stage selections from one completed V99 forward; no GT enters this trace."""

import torch

from models.rec_geometry_reranker import build_deployed_parent_state, blend_rec_geometry_scores
from models.rec_pareto_contextual_hierarchy import apply_pareto_contextual_policy


STAGES = ('native', 'parent', 'parent_after_geometry_validity',
          'geometry', 'v99_proposal', 'v99_final')


@torch.no_grad()
def trace_readout_stages(native_boxes, native_query_indices, readout, metadata):
    """Recover existing intermediate choices, then verify the deployed final scores.

    ``native_query_indices`` must come from the actual native evaluator; this
    function does not substitute another token-score or filtering path.
    V99 proposal is the ungated proposal, not a separate deployed system.
    Query indices identify slots within this forward, not persistent instances.
    """
    parent = readout['parent']
    candidate = parent['candidate_batch']
    runtime = readout['runtime']
    continuous = readout['continuous']
    batch, queries, _ = native_boxes.shape
    assert queries == candidate['num_queries'] == 256
    assert tuple(candidate['query_indices'].shape) == (batch, 16)
    assert tuple(native_query_indices.shape) == (batch,)
    assert native_query_indices.dtype == torch.long
    assert bool(((native_query_indices >= 0) & (native_query_indices < queries)).all())
    assert tuple(runtime['rec_geometry_boxes'].shape) == (batch, 112, 6)
    assert runtime['rec_geometry_runtime_mode'] == 'flat_geometry_axis'
    assert metadata['v99']['schema'] == 'rec-pareto-contextual-hierarchical-v1'
    assert metadata['geometry']['regressed_variant_index'] == 0

    before_geometry = build_deployed_parent_state(
        parent['compact_scores'], candidate['query_indices'], candidate['valid_mask'], queries)
    assert torch.equal(before_geometry['query_scores'], parent['query_scores'])
    effective_valid = continuous['geometry_valid'].reshape(batch, 16, 7)
    geometry_parent = build_deployed_parent_state(
        parent['compact_scores'], candidate['query_indices'], effective_valid.any(dim=2), queries)
    geometry = blend_rec_geometry_scores(
        geometry_parent, continuous['geometry']['ranking_logits'].float(), effective_valid,
        float(metadata['geometry']['geometry_weight']), 0)
    assert geometry['use_parent_query_axis'] is False
    settings = metadata['v99']['policy']
    hierarchy = continuous['v99']
    policy = apply_pareto_contextual_policy(
        geometry['flat_scores'], hierarchy['query_logits'], hierarchy['variant_logits'],
        effective_valid.any(dim=2), effective_valid, float(settings['aggregate_margin']),
        min_head_gain025=settings.get('min_head_gain025', 0.0),
        min_head_gain050=settings.get('min_head_gain050', 0.0))
    deployed_valid = runtime['rec_geometry_valid_mask']
    assert bool(deployed_valid.any(dim=1).all())
    expected_scores = policy['scores'].masked_fill(~deployed_valid, -float('inf'))
    assert torch.equal(expected_scores, runtime['rec_geometry_scores'])

    row_indices = torch.arange(batch, device=native_boxes.device)
    selections = {
        'native': native_query_indices,
        'parent': before_geometry['top1_query_index'],
        'parent_after_geometry_validity': geometry_parent['top1_query_index'],
    }
    stage_boxes = [native_boxes[row_indices, selections[name]] for name in STAGES[:3]]
    stage_queries = [selections[name] for name in STAGES[:3]]
    stage_variants = [torch.full_like(native_query_indices, -1) for _ in STAGES[:3]]
    flat_selections = {
        'geometry': geometry['flat_scores'].masked_fill(~deployed_valid, -float('inf')).argmax(dim=1),
        'v99_proposal': policy['proposal_indices'],
        'v99_final': runtime['rec_geometry_scores'].argmax(dim=1),
    }
    for name in STAGES[3:]:
        flat = flat_selections[name]
        assert bool(deployed_valid[row_indices, flat].all())
        compact = torch.div(flat, 7, rounding_mode='floor')
        stage_queries.append(candidate['query_indices'][row_indices, compact])
        stage_variants.append(flat.remainder(7))
        stage_boxes.append(runtime['rec_geometry_boxes'][row_indices, flat])
    return {
        'stage_names': STAGES,
        'boxes': torch.stack(stage_boxes, dim=1),
        'query_indices': torch.stack(stage_queries, dim=1),
        'variant_indices': torch.stack(stage_variants, dim=1),
        'top16_query_indices': candidate['query_indices'].detach().clone(),
        'top16_valid': candidate['valid_mask'].detach().clone(),
        'effective_variant_valid': effective_valid.detach().clone(),
        'deployed_variant_valid': deployed_valid.detach().clone(),
        'geometry_flat_indices': flat_selections['geometry'],
        'proposal_flat_indices': flat_selections['v99_proposal'],
        'final_flat_indices': flat_selections['v99_final'],
        'pareto_pass': policy['pareto_pass'],
        'predicted_head_gain': policy['head_gain'],
        'predicted_aggregate_gain': policy['aggregate_gain'],
        'final_scores_verified_equal': True,
        'persistent_instance_identity_available': False,
    }
