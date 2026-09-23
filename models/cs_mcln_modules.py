"""Three native, single-answer modules for context and support aware MCLN."""

import math

import torch
from torch import nn


def _member_mean(values, member_ids, count):
    """Aggregate only sampled points that actually belong to each superpoint."""
    sums = values.new_zeros(count, values.shape[-1])
    sums.index_add_(0, member_ids, values)
    numbers = values.new_zeros(count, 1)
    numbers.index_add_(0, member_ids, values.new_ones(values.shape[0], 1))
    return sums / numbers.clamp_min(1), numbers


class ObservationCrossScaleEnhancer(nn.Module):
    """Use actual raw, SA1, SA2 and propagated members to enrich scene memory."""

    def __init__(self, d_model=288):
        super().__init__()
        self.seed_delta = nn.Sequential(
            nn.Conv1d(128 + 256 + d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )
        self.source_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(7, 128), nn.ReLU(), nn.Linear(128, d_model)),
            nn.Sequential(nn.Linear(128 + 4, 128), nn.ReLU(), nn.Linear(128, d_model)),
            nn.Sequential(nn.Linear(256 + 4, 128), nn.ReLU(), nn.Linear(128, d_model)),
            nn.Sequential(nn.Linear(d_model + 4, 128), nn.ReLU(), nn.Linear(128, d_model)),
        ])
        self.source_score = nn.Linear(d_model + 1, 1)
        self.super_delta = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.seed_delta[-1].weight)
        nn.init.zeros_(self.seed_delta[-1].bias)
        nn.init.zeros_(self.super_delta.weight)
        nn.init.zeros_(self.super_delta.bias)

    def enhance_seeds(self, end_points):
        sa2_inds = end_points['sa2_inds'].long()
        sa1 = end_points['sa1_features'].gather(
            2, sa2_inds.unsqueeze(1).expand(-1, 128, -1)
        )
        sources = torch.cat((
            sa1,
            end_points['sa2_features'],
            end_points['fp2_features'],
        ), dim=1)
        return end_points['fp2_features'] + self.seed_delta(sources)

    def enhance_superpoints(self, super_features, super_xyz_list, point_clouds,
                            superpoint_ids, end_points, encoded_seeds):
        result = []
        sa1_global = end_points['sa1_inds'].long()
        sa2_global = sa1_global.gather(1, end_points['sa2_inds'].long())
        for batch_index, base in enumerate(super_features):
            member = superpoint_ids[batch_index].long()
            centers = super_xyz_list[batch_index].squeeze(0)
            n_super = base.shape[1]
            raw_xyz = point_clouds[batch_index, :, :3]
            raw_rgb = point_clouds[batch_index, :, 3:6]
            centered = raw_xyz - centers.index_select(0, member)
            raw_values = torch.cat((raw_rgb, centered.square()), dim=-1)
            raw_mean, raw_count = _member_mean(raw_values, member, n_super)

            source_inputs = [
                torch.cat((
                    raw_mean,
                    torch.log1p(raw_count) / math.log1p(raw_xyz.shape[0]),
                ), dim=-1),
            ]
            source_counts = [raw_count]
            for source_xyz, source_features, source_inds in (
                (end_points['sa1_xyz'][batch_index],
                 end_points['sa1_features'][batch_index].transpose(0, 1),
                 sa1_global[batch_index]),
                (end_points['sa2_xyz'][batch_index],
                 end_points['sa2_features'][batch_index].transpose(0, 1),
                 sa2_global[batch_index]),
                (end_points['fp2_xyz'][batch_index],
                 encoded_seeds[batch_index].transpose(0, 1),
                 sa2_global[batch_index]),
            ):
                ids = member.index_select(0, source_inds)
                rel = source_xyz - centers.index_select(0, ids)
                mean, count = _member_mean(
                    torch.cat((source_features, rel), dim=-1), ids, n_super
                )
                source_inputs.append(torch.cat((
                    mean,
                    torch.log1p(count) / math.log1p(source_xyz.shape[0]),
                ), dim=-1))
                source_counts.append(count)

            encoded = [layer(value) for layer, value in zip(
                self.source_encoders, source_inputs
            )]
            scores = torch.cat([
                self.source_score(torch.cat((feature,
                    torch.log1p(count)), dim=-1))
                for feature, count in zip(encoded, source_counts)
            ], dim=-1)
            valid = torch.cat([count > 0 for count in source_counts], dim=-1)
            weights = scores.masked_fill(~valid, -1e4).softmax(dim=-1)
            fused = sum(
                weights[:, source:source + 1] * feature
                for source, feature in enumerate(encoded)
            )
            result.append(base + self.super_delta(fused).transpose(0, 1))
        return result


class ContextSupportReader(nn.Module):
    """Read full-scene context and nearby superpoints with one candidate Query."""

    def __init__(self, d_model=288, heads=8, support_count=16):
        super().__init__()
        self.support_count = support_count
        self.global_attention = nn.MultiheadAttention(
            d_model, heads, batch_first=True
        )
        self.local_attention = nn.MultiheadAttention(
            d_model, heads, batch_first=True
        )
        self.relative_position = nn.Linear(3, d_model)
        self.delta = nn.Linear(3 * d_model, d_model)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, query, global_features, super_features,
                super_xyz_list, candidate_centers):
        context = self.global_attention(
            query, global_features, global_features, need_weights=False
        )[0]
        local = []
        for batch_index, memory in enumerate(super_features):
            centers = super_xyz_list[batch_index].squeeze(0)
            query_centers = candidate_centers[batch_index]
            nearest = torch.cdist(query_centers, centers).topk(
                min(self.support_count, centers.shape[0]),
                largest=False,
            ).indices
            locations = centers[nearest]
            values = memory.transpose(0, 1)[nearest]
            values = values + self.relative_position(
                locations - query_centers.unsqueeze(1)
            )
            selected = self.local_attention(
                query[batch_index].unsqueeze(1),
                values,
                values,
                need_weights=False,
            )[0].squeeze(1)
            local.append(selected)
        return query + self.delta(torch.cat((
            query, context, torch.stack(local, dim=0)
        ), dim=-1))


class MaskSupportBoxRefiner(nn.Module):
    """Feed differentiable predicted-mask support into the native box output."""

    def __init__(self, d_model=288):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(2 * d_model + 9, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 6),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, query, mask_query, super_features, super_xyz_list,
                centers, sizes):
        result_centers = []
        result_sizes = []
        for batch_index, memory in enumerate(super_features):
            xyz = super_xyz_list[batch_index].squeeze(0)
            feature = memory.transpose(0, 1)
            logits = mask_query[batch_index] @ feature.transpose(0, 1)
            mass = logits.sigmoid()
            weights = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            observed_center = weights @ xyz
            offset = xyz.unsqueeze(0) - observed_center.unsqueeze(1)
            observed_extent = (
                (weights.unsqueeze(-1) * offset.square()).sum(dim=1) + 1e-6
            ).sqrt()
            observed_feature = weights @ feature
            change = self.delta(torch.cat((
                query[batch_index], observed_feature,
                observed_center - centers[batch_index],
                observed_extent, sizes[batch_index],
            ), dim=-1))
            result_centers.append(centers[batch_index] + change[:, :3])
            result_sizes.append(sizes[batch_index] * change[:, 3:].exp())
        return torch.stack(result_centers), torch.stack(result_sizes)
