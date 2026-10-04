"""Unintegrated source draft: final boundary evidence conditions native scoring.

No native factory, GPU probe, optimizer update or accuracy result exists for
this draft. It emits a Query residual, not an independent deployed quality rank.
"""
import torch
from torch import nn

from pvground_boundary_box_refiner import expected_offset, SIZE_FLOOR


class BoundaryEvidenceReadback(nn.Module):
    """Read six final distributions in the condition of the full expression."""

    def __init__(self, use_geometry_evidence=True):
        super().__init__()
        self.use_geometry_evidence = use_geometry_evidence
        self.query = nn.Linear(288, 64)
        self.text = nn.Linear(288, 64)
        self.face_embedding = nn.Parameter(torch.empty(6, 64))
        nn.init.normal_(self.face_embedding, std=.02)
        self.evidence = nn.Sequential(nn.Linear(44, 64), nn.ReLU(), nn.Linear(64, 64))
        self.language_read = nn.MultiheadAttention(64, 4, batch_first=True)
        self.role_norm = nn.LayerNorm(64)
        self.face_read = nn.MultiheadAttention(64, 4, batch_first=True)
        self.output_norm = nn.LayerNorm(64)
        self.output = nn.Linear(64, 288)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('axes', torch.tensor([0, 1, 2, 0, 1, 2]), persistent=False)
        self.register_buffer('signs', torch.tensor([-1., -1., -1., 1., 1., 1.]), persistent=False)

    def forward(self, semantic_query, text, text_padding_mask, raw_points, end_points):
        batch, candidates, width = semantic_query.shape
        assert candidates == 256 and width == 288
        assert raw_points.shape == (batch, 50000, 6)
        logits = end_points['boundary_logits']
        whole = end_points['whole_mask_range_evidence']
        assert logits.shape == (batch, candidates, 6, 33)
        assert whole.shape == (batch, candidates, 109)
        probability = logits.softmax(-1)
        entropy = -(probability * logits.log_softmax(-1)).sum(-1)
        offset = expected_offset(logits)
        scene_low = raw_points[..., :3].amin(1)
        scene_span = (raw_points[..., :3].amax(1) - scene_low).clamp(min=SIZE_FLOOR)
        axis_low = scene_low.index_select(1, self.axes)[:, None]
        axis_span = scene_span.index_select(1, self.axes)[:, None]
        sign = self.signs[None, None]
        coarse_center = end_points['p3_coarse_center'].index_select(2, self.axes)
        coarse_size = end_points['p3_coarse_size'].clamp(min=SIZE_FLOOR).index_select(2, self.axes)
        final_center = end_points['last_center'].index_select(2, self.axes)
        final_size = end_points['last_pred_size'].index_select(2, self.axes)
        coarse_face = (coarse_center + .5 * sign * coarse_size - axis_low) / axis_span
        final_face = (final_center + .5 * sign * final_size - axis_low) / axis_span
        moments = torch.stack([whole[..., start:start+3].index_select(2, self.axes)
            for start in (96, 99, 102, 105)], dim=-1)
        mass = whole[..., 108:109].expand(batch, candidates, 6)
        floored = end_points['boundary_size_floored'].index_select(2, self.axes).to(logits.dtype)
        evidence = torch.cat([probability, offset[..., None], entropy[..., None],
            coarse_face[..., None], final_face[..., None], moments, mass[..., None],
            sign.expand(batch, candidates, 6)[..., None], floored[..., None]], dim=-1)
        assert evidence.shape == (batch, candidates, 6, 44)
        if not self.use_geometry_evidence:
            evidence = torch.zeros_like(evidence)
        query = self.query(semantic_query)
        roles = self.evidence(evidence) + query[:, :, None] + self.face_embedding[None, None]
        language = self.text(text)
        requested = roles.reshape(batch, candidates * 6, 64)
        contextual = self.language_read(requested, language, language,
            key_padding_mask=text_padding_mask, need_weights=False)[0]
        roles = self.role_norm(requested + contextual).reshape(batch * candidates, 6, 64)
        requested = query.reshape(batch * candidates, 1, 64)
        observed = self.face_read(requested, roles, roles, need_weights=False)[0][:, 0]
        residual = self.output(self.output_norm(requested[:, 0] + observed))
        return semantic_query + residual.reshape(batch, candidates, 288)
