"""Final-layer candidate-aligned raw-point box refinement for PV-Ground."""
import torch
from torch import nn


class CandidateAlignedBoxRefiner(nn.Module):
    """Read 16 input indices at the center/six faces; produce one box residual.

    Input XYZ/RGB are the same augmented 50000 points used by the native model.
    Nearest members are observations, not instance labels. Distances are encoded
    without a radius filter or duplicated index padding. Only the final box
    changes directly; native semantic, contrastive and Mask heads are untouched.
    """

    def __init__(self, d_model=288, neighbors=16, query_chunk=128):
        super().__init__()
        self.neighbors = neighbors
        self.query_chunk = query_chunk
        self.register_buffer('locations', torch.tensor([
            [0., 0., 0.], [1., 0., 0.], [-1., 0., 0.],
            [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]
        ]), persistent=False)
        self.member = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 64))
        self.condition = nn.Linear(d_model, 64)
        self.aggregate = nn.Sequential(nn.Linear(d_model + 7 * 128 + 3, 288), nn.ReLU())
        self.output = nn.Linear(288, 6)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @torch.no_grad()
    def nearest_members(self, xyz, locations):
        """Chunk locations to avoid a B x 1792 x 50000 distance allocation."""
        indices = []
        distances = []
        for begin in range(0, locations.shape[1], self.query_chunk):
            distance = torch.cdist(locations[:, begin:begin+self.query_chunk], xyz)
            values, members = distance.topk(self.neighbors, dim=-1, largest=False, sorted=True)
            indices.append(members)
            distances.append(values)
        return torch.cat(indices, dim=1), torch.cat(distances, dim=1)

    def forward(self, query, raw_points, coarse_center, coarse_size, end_points):
        batch, count, _ = query.shape
        assert raw_points.shape == (batch, 50000, 6)
        center = coarse_center.detach()
        # The native evaluator uses the same size floor. Use it for the sampling
        # layout only; the zero residual preserves even raw negative-size outputs.
        layout_size = coarse_size.detach().clamp(min=1e-6)
        locations = center[:, :, None] + .5 * layout_size[:, :, None] * self.locations
        flat_locations = locations.reshape(batch, count * 7, 3)
        indices, distances = self.nearest_members(raw_points[..., :3], flat_locations)
        batch_index = torch.arange(batch, device=query.device)[:, None, None]
        members = raw_points[batch_index, indices].reshape(batch, count, 7, self.neighbors, 6)
        distances = distances.reshape(batch, count, 7, self.neighbors, 1)
        member_input = torch.cat([
            members[..., 3:],
            members[..., :3] - center[:, :, None, None],
            members[..., :3] - locations[:, :, :, None],
            distances
        ], dim=-1)
        feature = torch.relu(self.member(member_input) + self.condition(query)[:, :, None, None])
        pooled = torch.cat([feature.mean(dim=-2), feature.max(dim=-2).values], dim=-1)
        evidence = torch.cat([query, pooled.reshape(batch, count, 7 * 128), layout_size], dim=-1)
        residual = self.output(self.aggregate(evidence))
        end_points['p3_coarse_center'] = coarse_center
        end_points['p3_coarse_size'] = coarse_size
        end_points['p3_neighbor_distances'] = distances.detach()
        return coarse_center + residual[..., :3], coarse_size + residual[..., 3:]


def install_candidate_box_refinement(model):
    assert model.candidate_box_refiner is None
    model.candidate_box_refiner = CandidateAlignedBoxRefiner()
