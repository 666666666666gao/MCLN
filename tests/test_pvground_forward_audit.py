import io
import json

import pytest
import torch
from torch import nn

from tools.pvground_forward_audit import PVGroundForwardAudit


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(2.0))

    def forward(self, query):
        return query + self.bias


class Support(nn.Module):
    def forward(self, query, supports, points, centers, sizes, predictions):
        return [predictions["sp_last_pred_masks"][0] + 3], ["geometry"]


class Span(nn.Module):
    def forward(self, query, supports, predictions, geometries):
        raw = torch.tensor([[-0.2, 0.5, 1.2], [0.0, 0.0, 1.0]])
        return (
            predictions["native_coarse_center"] + 1,
            predictions["native_coarse_size"] + 2,
            [{
                "raw_axis_gate": raw, "axis_gate": raw.clamp(0, 1),
                "reference_valid": torch.tensor([True, False]),
            }],
        )


@pytest.mark.parametrize("bypass", ["none", "reader", "support", "span"])
def test_hooks_bypass_only_requested_output_and_preserve_parameter_keys(bypass):
    model = nn.Module()
    decoder = nn.Module()
    decoder.local_visual = Reader()
    model.decoder = nn.ModuleList([decoder])
    model.pvground_support_corrector = Support()
    model.pvground_span_mixer = Span()
    keys = list(model.state_dict())
    stream = io.StringIO()
    audit = PVGroundForwardAudit(model, stream, bypass)
    predictions = {
        "sp_last_pred_masks": [torch.zeros(2, 4)],
        "native_coarse_center": torch.zeros(1, 2, 3),
        "native_coarse_size": torch.ones(1, 2, 3),
    }
    reader = model.decoder[-1].local_visual(torch.ones(1))
    support, geometry = model.pvground_support_corrector(
        None, None, None, None, None, predictions
    )
    center, size, _ = model.pvground_span_mixer(
        None, None, predictions, geometry
    )
    assert reader.item() == (0 if bypass == "reader" else 3)
    assert torch.equal(support[0], torch.full((2, 4), 0 if bypass == "support" else 3))
    assert torch.equal(center, torch.full((1, 2, 3), 0 if bypass == "span" else 1))
    assert torch.equal(size, torch.full((1, 2, 3), 1 if bypass == "span" else 3))
    row = json.loads(stream.getvalue())
    assert row["candidate_count"] == 2
    assert row["reference_valid_count"] == 1
    assert row["raw_gate_below_zero_by_axis"] == [1, 0, 0]
    assert row["raw_gate_above_one_by_axis"] == [0, 0, 1]
    assert row["clamped_gate_zero_by_axis"] == [2, 1, 0]
    assert row["clamped_gate_one_by_axis"] == [0, 0, 2]
    assert row["center_abs_change_mean_by_axis"] == (
        [0, 0, 0] if bypass == "span" else [1, 1, 1]
    )
    assert list(model.state_dict()) == keys
    audit.close()
    assert model.decoder[-1].local_visual(torch.ones(1)).item() == 3
    assert torch.equal(
        model.pvground_span_mixer(None, None, predictions, geometry)[0],
        torch.ones(1, 2, 3),
    )
