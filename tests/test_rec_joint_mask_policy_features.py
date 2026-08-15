import pytest
import torch

from models.rec_joint_box_mask import (
    build_mask_policy_feature_names,
    compute_mask_policy_inference_features,
)


def test_mask_policy_features_have_frozen_shape_and_mask_invalid_queries():
    text = torch.tensor([[[0.0, 1.0, -1.0], [3.0, 2.0, 1.0]]])
    query = torch.tensor([[[1.0, 0.0, -1.0], [1.0, 2.0, 3.0]]])
    valid = torch.tensor([[True, False]])
    features = compute_mask_policy_inference_features(
        text, query, torch.tensor([0.25]), valid
    )
    assert features.shape == (1, 2, 52)
    assert len(build_mask_policy_feature_names()) == 52
    assert len(set(build_mask_policy_feature_names())) == 52
    assert torch.isfinite(features).all()
    assert torch.equal(features[0, 1], torch.zeros(52))


def test_mask_policy_features_do_not_accept_changed_threshold_contract():
    text = torch.zeros(1, 1, 3)
    with pytest.raises(ValueError, match="threshold"):
        compute_mask_policy_inference_features(
            text, text, 0.5, torch.ones(1, 1, dtype=torch.bool), [0.0]
        )
