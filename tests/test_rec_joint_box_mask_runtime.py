import sys
from types import SimpleNamespace

import torch

import train_dist_mod
from main_utils import parse_option
from models.rec_joint_box_mask import (
    LEGACY_MASK_POLICY_INDEX,
    MASK_POLICY_COUNT,
)


class FixedAdapter(torch.nn.Module):
    def __init__(self, mask_values, box_values, policy_index=None):
        super().__init__()
        self.register_buffer("mask_values", torch.as_tensor(mask_values).float())
        self.register_buffer("box_values", torch.as_tensor(box_values).float())
        policy_index = (
            LEGACY_MASK_POLICY_INDEX
            if policy_index is None else int(policy_index)
        )
        policy = torch.zeros(16, MASK_POLICY_COUNT)
        policy[:, policy_index] = 1.0
        self.register_buffer("policy_values", policy)

    def forward(self, features, valid_mask):
        batch = features.shape[0]
        mask = self.mask_values.view(1, 16, 7).expand(batch, -1, -1)
        box = self.box_values.view(1, 16, 7, 2).expand(batch, -1, -1, -1)
        policy = self.policy_values.view(
            1, 16, MASK_POLICY_COUNT
        ).expand(batch, -1, -1)
        return {
            "mask_iou": mask,
            "box_logits": box,
            "mask_policy_logits": policy,
        }


def _runtime_inputs():
    features = torch.zeros(1, 112, 179)
    valid = torch.zeros(1, 112, dtype=torch.bool)
    valid[0, :3] = True
    baseline = torch.full((1, 112), -float("inf"))
    baseline[0, :3] = torch.tensor([1.0, 0.9, 0.8])
    mask = torch.zeros(112)
    mask[:3] = torch.tensor([0.50, 0.99, 0.90])
    box = torch.zeros(112, 2)
    box[:3] = torch.tensor([[5.0, 5.0], [-5.0, -5.0], [5.0, 5.0]])
    return features, valid, baseline, mask, box


def test_parser_exposes_joint_adapter_checkpoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_dist_mod.py"])
    args = parse_option()
    assert args.rec_joint_box_mask_checkpoint is None
    monkeypatch.setattr(sys, "argv", [
        "train_dist_mod.py", "--rec_joint_box_mask_checkpoint", "adapter.pth",
    ])
    assert parse_option().rec_joint_box_mask_checkpoint == "adapter.pth"


def test_joint_runtime_policy_promotes_only_safe_mask_candidate():
    features, valid, baseline, mask, box = _runtime_inputs()
    adapter = FixedAdapter(mask, box)
    adapter.eval()
    artifact = {
        "feature_mean": torch.zeros(179),
        "feature_std": torch.ones(179),
        "switch_margin": 0.01,
        "box_margin": 0.05,
    }
    out = train_dist_mod.apply_rec_joint_box_mask_runtime_policy(
        features, valid, baseline, adapter, artifact,
    )
    assert out["selected_flat_indices"].tolist() == [2]
    assert out["switched"].tolist() == [True]
    assert out["scores"][0, 2] > out["scores"][0, 0]
    assert torch.isneginf(out["scores"][0, 3:]).all()
    assert out["selected_parent_positions"].tolist() == [0]
    assert out["selected_mask_policy_indices"].tolist() == [
        LEGACY_MASK_POLICY_INDEX
    ]
    assert out["selected_mask_source_indices"].tolist() == [2]
    assert out["selected_mask_threshold_indices"].tolist() == [2]
    assert out["selected_mask_thresholds"].tolist() == [0.0]


def test_joint_runtime_policy_falls_back_bitwise_when_no_safe_switch():
    features, valid, baseline, mask, box = _runtime_inputs()
    box[:3] = torch.tensor([[5.0, 5.0], [-5.0, -5.0], [-5.0, -5.0]])
    adapter = FixedAdapter(mask, box)
    adapter.eval()
    artifact = {
        "feature_mean": torch.zeros(179),
        "feature_std": torch.ones(179),
        "switch_margin": 0.01,
        "box_margin": 0.05,
    }
    out = train_dist_mod.apply_rec_joint_box_mask_runtime_policy(
        features, valid, baseline, adapter, artifact,
    )
    assert out["selected_flat_indices"].tolist() == [0]
    assert out["switched"].tolist() == [False]
    assert torch.equal(out["scores"], baseline)
