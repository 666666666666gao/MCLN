"""Single-output CS modules adapted to PV-Ground's native point/voxel memory."""

import math

import torch
from torch import nn

from utils.scatter_util import deterministic_scatter_mean_dim0


SOURCE_WIDTHS = (128, 128, 128, 128, 256, 256)
SOURCE_NAMES = ('bev', 'raw_points', 'x_conv1', 'x_conv2', 'x_conv3', 'x_conv4')


def _member_mean(values, member_ids, count):
    mean = deterministic_scatter_mean_dim0(values, member_ids, dim_size=count)
    numbers = torch.bincount(member_ids, minlength=count).to(values.dtype).unsqueeze(-1)
    return mean, numbers


class PointVoxelObservationEnhancer(nn.Module):
    """Enrich seeds and actual superpoint members with native six-source features.

    Member counts describe raw/FPS membership, not VSA neighborhood confidence.
    """

    def __init__(self, d_model=288):
        super().__init__()
        self.seed_delta = nn.Sequential(
            nn.Conv1d(sum(SOURCE_WIDTHS) + d_model, d_model, 1),
            nn.ReLU(), nn.Conv1d(d_model, d_model, 1),
        )
        widths = (7,) + tuple(width + 4 for width in SOURCE_WIDTHS) + (d_model + 4,)
        self.source_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, d_model))
            for width in widths
        ])
        self.source_score = nn.Linear(d_model + 1, 1)
        self.super_delta = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.seed_delta[-1].weight)
        nn.init.zeros_(self.seed_delta[-1].bias)
        nn.init.zeros_(self.super_delta.weight)
        nn.init.zeros_(self.super_delta.bias)

    def enhance_seeds(self, source_features, fused_features):
        inputs = torch.cat((source_features, fused_features), dim=-1).transpose(1, 2)
        return fused_features + self.seed_delta(inputs).transpose(1, 2)

    def enhance_superpoints(self, super_features, super_xyz_list, point_clouds,
                            superpoint_ids, seed_inds, seed_xyz, source_features,
                            encoded_seeds):
        result = []
        for bs, base in enumerate(super_features):
            member = superpoint_ids[bs].long()
            centers = super_xyz_list[bs].squeeze(0)
            n_super = base.shape[1]
            raw_xyz = point_clouds[bs, :, :3]
            relative = raw_xyz - centers.index_select(0, member)
            raw, raw_count = _member_mean(torch.cat((
                point_clouds[bs, :, 3:6], relative.square(),
            ), dim=-1), member, n_super)
            inputs = [torch.cat((raw, torch.log1p(raw_count)
                                 / math.log1p(raw_xyz.shape[0])), dim=-1)]
            counts = [raw_count]
            sampled_member = member.index_select(0, seed_inds[bs].long())
            sampled_relative = seed_xyz[bs] - centers.index_select(0, sampled_member)
            features = source_features[bs].split(SOURCE_WIDTHS, dim=-1)
            features += (encoded_seeds[bs].transpose(0, 1),)
            for feature in features:
                mean, count = _member_mean(torch.cat((feature, sampled_relative), dim=-1),
                                           sampled_member, n_super)
                inputs.append(torch.cat((mean, torch.log1p(count)
                                         / math.log1p(seed_xyz.shape[1])), dim=-1))
                counts.append(count)
            encoded = [layer(value) for layer, value in zip(self.source_encoders, inputs)]
            scores = torch.cat([self.source_score(torch.cat((feature, torch.log1p(count)),
                                                           dim=-1))
                                for feature, count in zip(encoded, counts)], dim=-1)
            valid = torch.cat([count > 0 for count in counts], dim=-1)
            weights = scores.masked_fill(~valid, -1e4).softmax(dim=-1)
            fused = sum(weights[:, source:source + 1] * feature
                        for source, feature in enumerate(encoded))
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
                centers, sizes, return_evidence=False):
        result_centers = []
        result_sizes = []
        evidence = []
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
            support = torch.cat((
                observed_feature,
                observed_center - centers[batch_index],
                observed_extent, sizes[batch_index],
            ), dim=-1)
            change = self.delta(torch.cat((query[batch_index], support), dim=-1))
            result_centers.append(centers[batch_index] + change[:, :3])
            result_sizes.append(sizes[batch_index] * change[:, 3:].exp())
            if return_evidence:
                evidence.append(torch.cat((
                    centers[batch_index], support, change,
                ), dim=-1))
        if return_evidence:
            return (
                torch.stack(result_centers),
                torch.stack(result_sizes),
                torch.stack(evidence),
            )
        return torch.stack(result_centers), torch.stack(result_sizes)


class GeometryEvidenceReadback(nn.Module):
    """Aggregate each query's geometry roles before cross-query comparison."""

    def __init__(self, d_model=288, context_dim=64):
        super().__init__()
        self.d_model = d_model
        self.coarse_encoder = nn.Linear(d_model + 6, context_dim)
        self.support_encoder = nn.Linear(2 * d_model + 9, context_dim)
        self.refined_encoder = nn.Linear(d_model + 12, context_dim)
        self.encode = nn.Sequential(
            nn.Linear(d_model + 2 * context_dim, context_dim), nn.ReLU(),
        )
        self.set_attention = nn.MultiheadAttention(
            context_dim, num_heads=4, dropout=0.0, batch_first=True,
        )
        self.output = nn.Linear(context_dim, d_model)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query, evidence):
        center, observed_feature, center_offset, extent, size, change = evidence.split(
            (3, self.d_model, 3, 3, 3, 6), dim=-1,
        )
        role_tokens = torch.stack((
            self.coarse_encoder(torch.cat((query, center, size), dim=-1)),
            self.support_encoder(torch.cat((
                query, observed_feature, center + center_offset,
                center_offset, extent,
            ), dim=-1)),
            self.refined_encoder(torch.cat((
                query, center + change[..., :3],
                size * change[..., 3:].exp(), change,
            ), dim=-1)),
        ), dim=2)
        geometry = torch.cat((
            role_tokens.mean(dim=2), role_tokens.amax(dim=2),
        ), dim=-1)
        tokens = self.encode(torch.cat((query, geometry), dim=-1))
        context, _ = self.set_attention(
            tokens, tokens, tokens, need_weights=False,
        )
        return query + self.output(context)
