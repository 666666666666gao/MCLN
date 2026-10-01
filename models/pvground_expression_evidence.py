"""P2: expression-conditioned reading on the existing G observation memories.

The learned D/G reader and its task routing remain intact. This adds one shared
instance residual before the existing last-layer task normalization/FFN. It
does not add a scorer, alter matching, or query ground-truth geometry.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from pvground_source_query import SOURCE_WIDTHS
from pvground_observation_query import OBSERVATION_WIDTHS


class ExpressionEvidenceRead(nn.Module):
    def __init__(self):
        super().__init__()
        width = 64
        self.text_query = nn.Linear(288, width)
        self.text_key = nn.Linear(288, width)
        self.text_value = nn.Linear(288, width)
        self.condition = nn.Linear(288 + width, width)
        self.projections = nn.ModuleList(nn.Linear(n, width) for n in SOURCE_WIDTHS)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in SOURCE_WIDTHS)
        self.observation_keys = nn.ParameterList(
            nn.Parameter(torch.zeros(width, n)) for n in OBSERVATION_WIDTHS)
        self.observation_values = nn.ParameterList(
            nn.Parameter(torch.zeros(width, n)) for n in OBSERVATION_WIDTHS)
        # Local shape uses normalized coordinates; room relations retain metres.
        self.positions = nn.ModuleList(nn.Sequential(
            nn.Linear(6 + n, width), nn.ReLU(), nn.Linear(width, 1))
            for n in OBSERVATION_WIDTHS)
        self.output = nn.Linear(width, 288)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query, text, text_padding_mask, source_features,
                observations, source_xyz, centers, sizes):
        # Query [Q,B,288]; all memory/coordinates batch first.
        query = query.transpose(0, 1)
        text_logits = torch.matmul(self.text_query(query), self.text_key(text).transpose(-1, -2)) / 8.0
        text_logits = text_logits.masked_fill(text_padding_mask[:, None], float('-inf'))
        text_read = torch.matmul(text_logits.softmax(-1), self.text_value(text))
        condition = self.condition(torch.cat([query, text_read], -1))
        relative = source_xyz[:, None] - centers[:, :, None]
        local = relative / sizes[:, :, None].clamp_min(1e-6)
        logits, memories = [], []
        for feature, state, projection, norm, key_weight, value_weight, position in zip(
                torch.split(source_features, SOURCE_WIDTHS, -1), observations,
                self.projections, self.norms, self.observation_keys,
                self.observation_values, self.positions):
            content = norm(projection(feature))
            key = content + F.linear(state, key_weight)
            value = content + F.linear(state, value_weight)
            spatial = torch.cat([local, relative,
                                 state[:, None].expand(-1, query.shape[1], -1, -1)], -1)
            bias = position(spatial).squeeze(-1)
            logits.append(torch.matmul(condition, key.transpose(-1, -2)) / math.sqrt(key.shape[-1]) + bias)
            memories.append(value)
        weight = torch.cat(logits, -1).softmax(-1)
        evidence = torch.matmul(weight, torch.cat(memories, 1))
        return self.output(evidence).transpose(0, 1)


def install_expression_evidence_read(model):
    last = model.decoder[-1]
    assert last.task_read and last.candidate_evidence_read is None
    last.candidate_evidence_read = ExpressionEvidenceRead()
    last.candidate_evidence_read.train(last.training)
