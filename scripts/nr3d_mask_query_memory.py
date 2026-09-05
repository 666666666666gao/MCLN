"""Candidate-specific Mask Query reading from native superpoint memory.

This is an isolated C1 prototype. It does not alter identity scoring, Box
regression, Text Mask decoding, source selection, or the base state dict.
"""

import math

import torch
from torch import nn


class MaskQueryMemoryReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(288)
        self.memory_norm = nn.LayerNorm(288)
        self.query_projection = nn.Linear(288, 64, bias=False)
        self.key_projection = nn.Linear(288, 64, bias=False)
        self.value_projection = nn.Linear(288, 64, bias=False)
        self.output = nn.Linear(64, 288, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, query, memory, superpoint_xyz, boxes):
        """One scene: (Q,288), (S,288), (S,3), (Q,6) -> (Q,288)."""
        normalized_memory = self.memory_norm(memory)
        keys = self.key_projection(normalized_memory)
        values = self.value_projection(normalized_memory)
        queries = self.query_projection(self.query_norm(query))
        # Same positive-size convention as existing native Mask geometry.
        half_size = boxes[:, 3:].clamp(min=1e-4).unsqueeze(1) * .5
        relative = (superpoint_xyz.unsqueeze(0) - boxes[:, :3].unsqueeze(1)) / half_size
        spatial_bias = -.5 * relative.square().sum(dim=-1)
        weights = (queries @ keys.T / math.sqrt(64) + spatial_bias).softmax(dim=-1)
        context = weights @ values
        return query + self.output(context)


class MaskQueryMemoryIntervention:
    """Wrap only native Query Mask prediction; retain its original writer."""

    def __init__(self, model, addon):
        self.model = model
        self.addon = addon
        self.original_predict = model._seg_seeds_prediction
        self.superpoint_xyz = []
        self.scene_index = 0
        self.batch_size = 0
        self.handles = [
            model.register_forward_pre_hook(self.begin_batch),
            model.super_grouper.register_forward_pre_hook(self.capture_centers),
            model.register_forward_hook(self.end_batch),
        ]
        model._seg_seeds_prediction = self.predict

    def begin_batch(self, module, inputs):
        self.batch_size = inputs[0]['point_clouds'].shape[0]
        self.superpoint_xyz = []
        self.scene_index = 0

    def capture_centers(self, module, inputs):
        self.superpoint_xyz.append(inputs[1].squeeze(0))

    def predict(self, query, mask_feats, end_points, prefix=''):
        scene = self.scene_index
        boxes = torch.cat((end_points[prefix + 'center'][scene],
                           end_points[prefix + 'pred_size'][scene]), dim=-1)
        updated = self.addon(query.squeeze(0), mask_feats.squeeze(0).T,
                             self.superpoint_xyz[scene], boxes)
        self.scene_index += 1
        return self.original_predict(updated.unsqueeze(0), mask_feats, end_points, prefix)

    def end_batch(self, module, inputs, output):
        assert self.scene_index == len(self.superpoint_xyz) == self.batch_size
        self.superpoint_xyz = []

    def remove(self):
        self.model._seg_seeds_prediction = self.original_predict
        for handle in self.handles:
            handle.remove()
        self.superpoint_xyz = []
