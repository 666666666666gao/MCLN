"""Face-specific training residual on a retained axis-wise geometry prior.

Source preparation only. Install as the PV model's candidate_span_mixer;
native last_center/last_pred_size and their existing loss remain the outputs.
No ground truth, candidate scoring, pruning or evaluation-time model switch.
"""
import torch
from torch import nn


class FaceResidualSpanMixer(nn.Module):
    def __init__(self, axis_prior, use_source_evidence):
        super().__init__()
        assert axis_prior.source_mode == 'extremal_support'
        assert isinstance(use_source_evidence, bool)
        self.axis_prior = axis_prior
        self.use_source_evidence = use_source_evidence
        self.face_residual = nn.Sequential(nn.Linear(332, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.face_residual[-1].weight)
        nn.init.zeros_(self.face_residual[-1].bias)

    def refine_one(self, query, native_center, native_size, mask_center,
                   mask_size, prior_center, prior_size, evidence):
        assert query.shape == (256, 288)
        assert evidence['face_tokens'].shape == (256, 6, 32)
        assert prior_center.shape == prior_size.shape == native_center.shape == (256, 3)
        native_size = native_size.clamp_min(1e-6)
        native_faces = torch.stack([native_center-native_size/2,
                                   native_center+native_size/2], -1)
        mask_faces = torch.stack([mask_center-mask_size/2,
                                 mask_center+mask_size/2], -1)
        scale = (native_size+mask_size)[:, :, None].expand(-1, -1, 2)
        prior_gate = evidence['axis_gate'][:, :, None].expand(-1, -1, 2).reshape(256, 6)
        fraction = evidence['source_fraction'].reshape(256, 6)
        tokens = evidence['face_tokens']
        if not self.use_source_evidence:
            # Same-capacity control: the extra face correction cannot see the
            # extreme-source tokens/count fraction; the retained prior is common.
            tokens = torch.zeros_like(tokens)
            fraction = torch.zeros_like(fraction)
        geometry = torch.stack([
            ((native_faces-mask_faces)/scale).reshape(256, 6),
            (native_size[:, :, None].expand(-1, -1, 2)/scale).reshape(256, 6),
            (mask_size[:, :, None].expand(-1, -1, 2)/scale).reshape(256, 6),
            prior_gate, fraction], -1)
        inputs = torch.cat([tokens, query[:, None].expand(-1, 6, -1), geometry,
            evidence['reference_valid'].to(query.dtype)[:, None, None].expand(-1, 6, -1),
            torch.eye(6, device=query.device, dtype=query.dtype)[None].expand(256, -1, -1)], -1)
        assert inputs.shape == (256, 6, 332)
        residual = self.face_residual(inputs).squeeze(-1)
        face_gate = (prior_gate+residual).clamp(0, 1)
        movement = ((face_gate-prior_gate).reshape(256, 3, 2)
                    * (native_faces-mask_faces))
        # Add movements to the existing center/size rather than reconstructing
        # their faces: zero residual preserves the prior's float32 outputs.
        center = prior_center + (movement[:, :, 0]+movement[:, :, 1])/2
        raw_size = prior_size + movement[:, :, 1]-movement[:, :, 0]
        # Closed data contains31 disjoint source-axis intervals/30 expressions.
        # Independent face choices can invert them. Absolute size orders the
        # two endpoints, with the same1e-6 minimum as the native layout.
        size = raw_size.abs().clamp_min(1e-6)
        current = dict(evidence)
        current.update(face_gate=face_gate,face_gate_residual=residual,
            raw_face_size=raw_size,face_movement=movement)
        return center, size, current

    def forward(self, query, supports, predictions, geometries):
        prior_center, prior_size, prior_evidence = self.axis_prior(
            query, supports, predictions, geometries)
        centers, sizes, evidence = [], [], []
        for bid in range(len(query)):
            center, size, current = self.refine_one(query[bid],
                predictions['native_coarse_center'][bid], predictions['native_coarse_size'][bid],
                predictions['last_center'][bid], predictions['last_pred_size'][bid],
                prior_center[bid], prior_size[bid], prior_evidence[bid])
            centers.append(center)
            sizes.append(size)
            evidence.append(current)
        return torch.stack(centers), torch.stack(sizes), evidence
