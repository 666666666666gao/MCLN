"""Decode each face from its axis profile and observed local members.

This is an unintegrated source implementation. It has no native preflight or
accuracy result yet. The fixed-G plain distribution head remains its control.
"""
import torch
from torch import nn

from pvground_whole_mask_box_refiner import WholeMaskSupportBoxRefiner
from pvground_boundary_box_refiner import REG_MAX, SIZE_FLOOR, expected_offset, decode_faces


class FaceConditionedBoxRefiner(WholeMaskSupportBoxRefiner):
    """Preserve directional evidence until a shared six-face decoding step."""

    def __init__(self):
        super().__init__(use_whole_range=True)
        # The flat 1302-D aggregate is the completed control, not this decoder.
        del self.aggregate
        self.face_embedding = nn.Parameter(torch.empty(6, 64))
        nn.init.normal_(self.face_embedding, std=.02)
        self.face_condition = nn.Linear(11, 64)
        self.range_member = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64))
        self.support_attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.support_norm = nn.LayerNorm(64)
        self.face_update = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        self.output = nn.Linear(64, REG_MAX + 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # Native local positions are center,x+,x-,y+,y-,z+,z-.
        # Distribution target/decode order is x-,y-,z-,x+,y+,z+.
        self.register_buffer('face_local_ids', torch.tensor([2, 4, 6, 1, 3, 5]), persistent=False)
        self.register_buffer('face_axes', torch.tensor([0, 1, 2, 0, 1, 2]), persistent=False)
        self.register_buffer('face_signs', torch.tensor([-1., -1., -1., 1., 1., 1.]), persistent=False)

    def forward(self, query, raw_points, coarse_center, coarse_size, end_points):
        batch, count, _ = query.shape
        assert raw_points.shape == (batch, 50000, 6) and count == 256
        center = coarse_center.detach()
        layout_size = coarse_size.detach().clamp(min=SIZE_FLOOR)
        locations = center[:, :, None] + .5 * layout_size[:, :, None] * self.locations
        indices, distances = self.nearest_members(raw_points[..., :3], locations.reshape(batch, count * 7, 3))
        batch_index = torch.arange(batch, device=query.device)[:, None, None]
        members = raw_points[batch_index, indices].reshape(batch, count, 7, self.neighbors, 6)
        distances = distances.reshape(batch, count, 7, self.neighbors, 1)
        support = self.member_support(indices, end_points, batch, count)
        member_input = torch.cat([members[..., 3:], members[..., :3] - center[:, :, None, None],
            members[..., :3] - locations[:, :, :, None], distances, support], dim=-1)
        condition = self.condition(query)
        local = torch.relu(self.member(member_input) + condition[:, :, None, None])
        whole, coarse = self.whole_range(raw_points, coarse_center, coarse_size, end_points)

        # Each face reads its own axis's 32 scene-normalized support bins.
        # Relative position retains the full scene and does not crop to the box.
        profile = whole[..., :96].reshape(batch, count, 3, 32).index_select(2, self.face_axes)
        positions = (torch.arange(32, device=query.device, dtype=query.dtype) + .5) / 32
        positions = positions[None, None, None].expand_as(profile)
        axis_center = coarse[..., :3].index_select(2, self.face_axes)
        signs = self.face_signs[None, None, :, None].expand_as(profile)
        range_input = torch.stack([profile, positions, positions - axis_center[..., None], signs], dim=-1)
        global_tokens = self.range_member(range_input)
        axis_statistics = torch.stack([
            whole[..., start:start + 3].index_select(2, self.face_axes)
            for start in (96, 99, 102, 105)
        ] + [whole[..., 108:109].expand(batch, count, 6)], dim=-1)
        frame = coarse[:, :, None].expand(batch, count, 6, 6)
        face_query = (condition[:, :, None] + self.face_embedding[None, None]
                      + self.face_condition(torch.cat([frame, axis_statistics], dim=-1)))

        # Six faces keep their own 16 observed members plus the center members.
        # The four native Mask channels remain soft evidence, not hard gates.
        face_local = local.index_select(2, self.face_local_ids)
        center_local = local[:, :, 0:1].expand(batch, count, 6, self.neighbors, 64)
        memory = torch.cat([global_tokens, face_local, center_local], dim=-2)
        rows = batch * count * 6
        requested = face_query.reshape(rows, 1, 64)
        memory = memory.reshape(rows, 32 + 2 * self.neighbors, 64)
        observed = self.support_attention(requested, memory, memory, need_weights=False)[0][:, 0]
        face_state = self.support_norm(face_query.reshape(rows, 64) + observed)
        face_state = face_state + self.face_update(face_state)
        logits = self.output(face_state).reshape(batch, count, 6, REG_MAX + 1)
        final_center, final_size, raw_size = decode_faces(coarse_center, coarse_size, expected_offset(logits))

        end_points['whole_mask_range_evidence'] = whole
        end_points['p3_coarse_center'] = coarse_center
        end_points['p3_coarse_size'] = coarse_size
        end_points['p3_neighbor_distances'] = distances.detach()
        end_points['boundary_logits'] = logits
        end_points['boundary_face_state'] = face_state.reshape(batch, count, 6, 64)
        end_points['boundary_size_floor_count'] = (raw_size <= SIZE_FLOOR).sum().detach()
        end_points['boundary_size_floored'] = (raw_size <= SIZE_FLOOR).detach()
        return final_center, final_size


def install_face_conditioned_refinement(model):
    assert model.candidate_box_refiner is None
    model.candidate_box_refiner = FaceConditionedBoxRefiner()
