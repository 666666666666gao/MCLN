"""Reuse the protected REC readout weights in an online training graph.

Candidate construction and deployment decisions retain the existing V99 rules.
Training supervises the continuous outputs before rank conversion and selection.
This is joint training of the existing system, not removal of its geometry rules.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from models.rec_hierarchical_reranker import monotone_hit_probabilities
from models.rec_pareto_contextual_hierarchy import ParetoContextualHierarchicalReranker
from models.rec_reranker import (
    QueryReranker, compute_query_ious, compute_rec_reranker_loss, select_listwise_targets,
)
from scripts.run_v95_threshold_aligned_listwise_hierarchical import (
    graded_quality, masked_soft_listwise_cross_entropy,
)


def detach_prediction_tree(value):
    """The matched control blocks only readout-to-backbone gradients."""
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {name: detach_prediction_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [detach_prediction_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_prediction_tree(item) for item in value)
    return value


class JointRecReadout(nn.Module):
    """One module holding all three pretrained scorers and their normalization."""

    def __init__(self, artifacts):
        super().__init__()
        self.metadata = {
            name: copy.deepcopy({key: value for key, value in artifact.items()
                                 if key != 'model_state_dict'})
            for name, artifact in artifacts.items()
        }
        self.scorers = nn.ModuleDict({
            'parent': QueryReranker(**artifacts['parent']['model_config']),
            'geometry': QueryReranker(**artifacts['geometry']['model_config']),
            'v99': ParetoContextualHierarchicalReranker(),
        })
        for name, model in self.scorers.items():
            model.load_state_dict(artifacts[name]['model_state_dict'], strict=True)

    def export_artifacts(self):
        """Embed current readout weights and metadata in the training checkpoint."""
        artifacts = copy.deepcopy(self.metadata)
        for name, model in self.scorers.items():
            artifacts[name]['model_state_dict'] = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        return artifacts

    def forward(self, end_points, inputs, detach_visual=False):
        from train_dist_mod import (
            _build_rec_reranker_outputs_float32,
            _build_rec_geometry_runtime_outputs_float32,
        )
        if detach_visual:
            end_points = detach_prediction_tree(end_points)
        captured = {}

        def capture(name):
            def hook(module, arguments, output):
                captured[name] = output
                if name in ('parent', 'geometry'):
                    captured[name + '_valid'] = arguments[1]
            return hook

        handles = [model.register_forward_hook(capture(name))
                   for name, model in self.scorers.items()]
        parent = _build_rec_reranker_outputs_float32(
            end_points, inputs, self.scorers['parent'], self.metadata['parent'])
        runtime = _build_rec_geometry_runtime_outputs_float32(
            end_points, inputs, parent, self.scorers['geometry'], self.metadata['geometry'],
            hierarchical_model=self.scorers['v99'], hierarchical_artifact=self.metadata['v99'])
        for handle in handles:
            handle.remove()
        return {'runtime': runtime, 'parent': parent, 'continuous': captured}


def _reranker_loss(outputs, ious, valid):
    # A row without a passing box still supplies quality labels, but no positive rank target.
    quality_loss, _ = compute_rec_reranker_loss(outputs, ious, valid, listwise_weight=0.)
    covered = ((ious > .25) & valid).any(dim=1)
    rank_loss = outputs['ranking_logits'].sum() * 0.
    if bool(covered.any()):
        targets = select_listwise_targets(ious[covered], valid[covered])
        rank_loss = F.cross_entropy(
            outputs['ranking_logits'][covered].masked_fill(~valid[covered], -1e4), targets)
        rank_loss = rank_loss * covered.float().mean()
    return quality_loss + rank_loss, int(covered.sum())


def joint_rec_readout_loss(readout, root_boxes, root_valid):
    """GT is read here only; root boxes and targets never enter the readout forward."""
    parent = readout['parent']
    runtime = readout['runtime']
    continuous = readout['continuous']
    parent_ious = compute_query_ious(
        parent['candidate_batch']['boxes'].detach(), root_boxes, root_valid).detach()
    geometry_ious = compute_query_ious(
        runtime['rec_geometry_boxes'].detach(), root_boxes, root_valid).detach()
    parent_loss, parent_covered = _reranker_loss(
        continuous['parent'], parent_ious, continuous['parent_valid'])
    geometry_loss, geometry_covered = _reranker_loss(
        continuous['geometry'], geometry_ious, continuous['geometry_valid'])

    variant_valid = runtime['rec_geometry_valid_mask'].reshape(-1, 16, 7)
    query_valid = variant_valid.any(dim=2)
    ious = geometry_ious.reshape(-1, 16, 7)
    quality = graded_quality(ious)
    query_quality = quality.masked_fill(~variant_valid, -float('inf')).max(dim=2).values
    hierarchy = continuous['v99']
    query_probability = monotone_hit_probabilities(hierarchy['query_logits'])
    variant_probability = monotone_hit_probabilities(hierarchy['variant_logits'])
    query_utility = 2. * query_probability[..., 0] + query_probability[..., 1]
    variant_utility = 2. * variant_probability[..., 0] + variant_probability[..., 1]
    covered_queries = ((ious > .25) & variant_valid).any(dim=2)
    covered_rows = covered_queries.any(dim=1)
    query_loss = query_utility.sum() * 0.
    variant_loss = variant_utility.sum() * 0.
    if bool(covered_rows.any()):
        query_loss = masked_soft_listwise_cross_entropy(
            query_utility[covered_rows], query_quality[covered_rows], query_valid[covered_rows], dim=1)
        query_loss = query_loss * covered_rows.float().mean()
        variant_loss = masked_soft_listwise_cross_entropy(
            variant_utility[covered_queries], quality[covered_queries], variant_valid[covered_queries], dim=1)
        variant_loss = variant_loss * covered_queries.sum().float() / query_valid.sum()
    hierarchy_loss = query_loss + variant_loss
    loss = parent_loss + geometry_loss + hierarchy_loss
    return loss, {
        'parent_loss': float(parent_loss.detach()),
        'geometry_loss': float(geometry_loss.detach()),
        'hierarchy_loss': float(hierarchy_loss.detach()),
        'parent_covered_rows': parent_covered,
        'geometry_covered_rows': geometry_covered,
        'hierarchy_covered_queries': int(covered_queries.sum()),
    }
