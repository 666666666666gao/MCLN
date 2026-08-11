import copy

import pytest
import torch
from types import MethodType
from types import SimpleNamespace

from main_utils import BaseTrainTester
from models.losses import (
    build_joint_query_mask_candidate_mask,
    compute_hungarian_loss,
    compute_joint_query_mask_candidate_loss,
    dice_loss,
    lovasz_hinge_loss,
    sigmoid_focal_loss,
)
from models.joint_query_quality import (
    JOINT_QUERY_GATE_EVIDENCE_DIM,
    JOINT_QUERY_GATE_EVIDENCE_NAMES,
    SOURCE_DISTRIBUTION_RELIABILITY_DIM,
    JointQualityAdaptiveSourceMixer,
    JointQueryQualityReranker,
    QuerySuperpointMaskRefiner,
    build_joint_query_gate_evidence,
    build_source_distribution_reliability_features,
    compute_joint_query_quality_loss,
    compute_joint_query_source_mix_alignment_loss,
    joint_query_target_quality,
    ordinal_threshold_logits,
    summarize_joint_query_residual,
)
from models.mask_fusion import (
    QUERY_MASK_SOURCE_EVIDENCE_DIM,
    apply_query_mask_calibration,
    apply_query_superpoint_mask_residual,
    fuse_query_mask_logits,
)


def _inputs(batch=2, queries=5, features=8):
    torch.manual_seed(7)
    values = torch.randn(batch, queries, features)
    baseline = torch.tensor([
        [0.9, 0.2, 0.7, 0.1, 0.4],
        [0.1, 0.8, 0.3, 0.6, 0.2],
    ])[:batch, :queries]
    valid = torch.ones(batch, queries, dtype=torch.bool)
    return values, baseline, valid


def _gate_outputs(batch=2, queries=5):
    torch.manual_seed(71)
    candidate_mask = torch.tensor([
        [False, True, True, False, False],
        [True, False, False, True, False],
    ])[:batch, :queries]
    default = torch.tensor([0, 1])[:batch]
    selected = torch.tensor([2, 3])[:batch].clamp(max=queries - 1)
    anchor = torch.tensor([4, 0])[:batch].clamp(max=queries - 1)
    return {
        "moe_gate_candidate_mask": candidate_mask,
        "moe_gate_default_query": default.clamp(max=queries - 1),
        "moe_gate_selected_query": selected,
        "moe_gate_action_anchor_query": anchor,
        "moe_candidate_scores": torch.randn(batch, queries),
        "moe_gate_expected_utility": torch.randn(batch, queries),
        "moe_gate_direct_utility": torch.randn(batch, queries),
        "moe_gate_action_margin": torch.randn(batch, queries),
        "moe_gate_box_logits": torch.randn(batch, queries, 2, 3),
        "moe_gate_mask_logits": torch.randn(batch, queries, 2, 3),
        "moe_gate_decision_logits": torch.randn(batch, queries, 3),
    }


def test_v46_gate_evidence_contract_uses_only_deployed_gate_outputs():
    valid = torch.ones(2, 5, dtype=torch.bool)
    outputs = _gate_outputs()
    evidence = build_joint_query_gate_evidence(outputs, valid)

    assert len(JOINT_QUERY_GATE_EVIDENCE_NAMES) == 24
    assert JOINT_QUERY_GATE_EVIDENCE_DIM == 24
    assert evidence.shape == (2, 5, 24)
    assert torch.isfinite(evidence).all()
    assert bool(((evidence >= 0.0) & (evidence <= 1.0)).all().item())
    torch.testing.assert_close(
        evidence[..., 9:15].reshape(2, 5, 2, 3).sum(-1),
        torch.ones(2, 5, 2),
    )
    torch.testing.assert_close(
        evidence[..., 15:21].reshape(2, 5, 2, 3).sum(-1),
        torch.ones(2, 5, 2),
    )
    torch.testing.assert_close(
        evidence[..., 21:24].sum(-1), torch.ones(2, 5)
    )


def test_v46_gate_evidence_is_query_permutation_equivariant():
    valid = torch.ones(2, 5, dtype=torch.bool)
    outputs = _gate_outputs()
    direct = build_joint_query_gate_evidence(outputs, valid)
    permutation = torch.tensor([3, 1, 4, 0, 2])
    inverse = permutation.argsort()
    permuted_outputs = {}
    index_keys = {
        "moe_gate_default_query",
        "moe_gate_selected_query",
        "moe_gate_action_anchor_query",
    }
    for key, value in outputs.items():
        if key in index_keys:
            permuted_outputs[key] = inverse[value]
        else:
            permuted_outputs[key] = value[:, permutation]
    permuted = build_joint_query_gate_evidence(
        permuted_outputs, valid[:, permutation]
    )

    torch.testing.assert_close(direct, permuted[:, inverse])


def test_v46_gate_evidence_rejects_missing_or_invalid_contracts():
    valid = torch.ones(2, 5, dtype=torch.bool)
    outputs = _gate_outputs()
    outputs.pop("moe_gate_box_logits")
    with pytest.raises(ValueError, match="outputs are missing"):
        build_joint_query_gate_evidence(outputs, valid)

    outputs = _gate_outputs()
    outputs["moe_gate_action_margin"] = torch.full((2, 5), float("nan"))
    with pytest.raises(ValueError, match="action_margin"):
        build_joint_query_gate_evidence(outputs, valid)


def test_v46_gate_evidence_preserves_identity_and_detaches_gate_inputs():
    features, baseline, valid = _inputs()
    gate_evidence = torch.rand(
        *baseline.shape, JOINT_QUERY_GATE_EVIDENCE_DIM, requires_grad=True
    )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_gate_evidence=True, detach_inputs=True,
    )

    identity = model(features, baseline, valid, gate_evidence=gate_evidence)
    assert torch.equal(identity["scores"], baseline)
    assert torch.count_nonzero(identity["residual"]) == 0

    with torch.no_grad():
        model.residual_head.weight.normal_(mean=0.0, std=0.1)
    outputs = model(features, baseline, valid, gate_evidence=gate_evidence)
    outputs["scores"].sum().backward()
    assert gate_evidence.grad is None
    assert model.input_projection[1].weight.grad is not None

    with pytest.raises(ValueError, match="gate_evidence"):
        model(features, baseline, valid)
    with pytest.raises(ValueError, match="requires use_gate_evidence"):
        JointQueryQualityReranker(8)(
            features, baseline, valid, gate_evidence=gate_evidence
        )


def test_zero_initialization_is_exact_baseline_identity():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    ).eval()
    outputs = model(features, baseline, valid)
    assert torch.equal(outputs["residual"], torch.zeros_like(baseline))
    assert torch.equal(outputs["selected_indices"], baseline.argmax(1))
    assert torch.equal(outputs["scores"], baseline)


def test_v42_zero_initialization_is_exact_box_and_mask_identity():
    features, baseline, valid = _inputs()
    base_mask_weights = torch.tensor([
        [0.2, 0.4, 0.6, 0.8, 0.5],
        [0.7, 0.3, 0.1, 0.9, 0.5],
    ])
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    ).eval()

    outputs = model(features, baseline, valid, base_mask_weights)

    assert torch.equal(outputs["scores"], baseline)
    assert torch.count_nonzero(outputs["residual"]) == 0
    assert torch.equal(outputs["mask_fusion_weights"], base_mask_weights)
    assert torch.count_nonzero(outputs["mask_alpha_residual"]) == 0
    assert torch.count_nonzero(outputs["mask_logit_bias"]) == 0


def test_v43_source_evidence_preserves_exact_step_zero_identity():
    features, baseline, valid = _inputs()
    base_mask_weights = torch.rand_like(baseline)
    source_evidence = torch.rand(
        *baseline.shape, QUERY_MASK_SOURCE_EVIDENCE_DIM
    )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True, use_source_mask_evidence=True,
    ).eval()

    outputs = model(
        features, baseline, valid, base_mask_weights, source_evidence
    )

    assert torch.equal(outputs["scores"], baseline)
    assert torch.equal(outputs["mask_fusion_weights"], base_mask_weights)
    assert torch.count_nonzero(outputs["residual"]) == 0
    assert torch.count_nonzero(outputs["mask_alpha_residual"]) == 0
    assert torch.count_nonzero(outputs["mask_logit_bias"]) == 0


def test_v43_source_evidence_is_required_validated_and_detached():
    features, baseline, valid = _inputs()
    base_mask_weights = torch.rand_like(baseline)
    source_evidence = torch.rand(
        *baseline.shape, QUERY_MASK_SOURCE_EVIDENCE_DIM,
        requires_grad=True,
    )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True, use_source_mask_evidence=True,
        detach_inputs=True,
    )
    with torch.no_grad():
        model.mask_calibration_head.weight.normal_(mean=0.0, std=0.1)

    outputs = model(
        features, baseline, valid, base_mask_weights, source_evidence
    )
    outputs["mask_logit_bias"].sum().backward()

    assert source_evidence.grad is None
    assert model.input_projection[1].weight.grad is not None
    with pytest.raises(ValueError, match="source_mask_evidence"):
        model(features, baseline, valid, base_mask_weights)
    with pytest.raises(ValueError, match="source_mask_evidence"):
        model(
            features, baseline, valid, base_mask_weights,
            source_evidence[..., :-1],
        )
    with pytest.raises(ValueError, match="requires enabled mask calibration"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4,
            use_source_mask_evidence=True,
        )


def test_v42_mask_calibration_head_receives_both_channel_gradients():
    features, baseline, valid = _inputs()
    base_mask_weights = torch.full_like(baseline, 0.4)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    )

    outputs = model(features, baseline, valid, base_mask_weights)
    loss = (
        outputs["mask_fusion_weights"].sum()
        + outputs["mask_logit_bias"].sum()
    )
    loss.backward()

    gradient = model.mask_calibration_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert bool((gradient.abs().sum(dim=1) > 0.0).all().item())


def test_v42_reaches_pure_source_and_unit_threshold_bias_without_saturation():
    features, baseline, valid = _inputs()
    base_mask_weights = torch.full_like(baseline, 0.5)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    ).eval()
    half_logit = torch.atanh(torch.tensor(0.5))
    with torch.no_grad():
        model.mask_calibration_head.bias.fill_(half_logit)

    outputs = model(features, baseline, valid, base_mask_weights)

    torch.testing.assert_close(
        outputs["mask_alpha_residual"], torch.full_like(baseline, 0.5)
    )
    torch.testing.assert_close(
        outputs["mask_fusion_weights"], torch.ones_like(baseline)
    )
    torch.testing.assert_close(
        outputs["mask_logit_bias"], torch.ones_like(baseline)
    )


def test_invalid_queries_are_never_selected():
    features, baseline, valid = _inputs()
    valid[:, 0] = False
    baseline[:, 0] = 1000.0
    outputs = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    ).eval()(features, baseline, valid)
    assert not bool((outputs["selected_indices"] == 0).any().item())
    assert torch.equal(outputs["residual"][:, 0], torch.zeros(2))


def test_query_permutation_equivariance():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    ).eval()
    permutation = torch.tensor([3, 1, 4, 0, 2])
    direct = model(features, baseline, valid)
    permuted = model(
        features[:, permutation], baseline[:, permutation], valid[:, permutation]
    )
    inverse = permutation.argsort()
    for key in (
            "scores", "residual", "box_iou", "mask_iou", "quality",
            "centered_quality", "baseline_rank", "baseline_standardized"):
        assert torch.allclose(direct[key], permuted[key][:, inverse], atol=1e-6)
    for key in ("box_logits", "mask_logits"):
        assert torch.allclose(
            direct[key], permuted[key][:, inverse], atol=1e-6
        )


def test_v42_calibration_is_permutation_equivariant_and_masks_invalid_queries():
    features, baseline, valid = _inputs()
    valid[:, 4] = False
    base_mask_weights = torch.tensor([
        [0.2, 0.4, 0.6, 0.8, 0.5],
        [0.7, 0.3, 0.1, 0.9, 0.5],
    ])
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    ).eval()
    with torch.no_grad():
        model.mask_calibration_head.weight.normal_(mean=0.0, std=0.1)
        model.mask_calibration_head.bias.copy_(torch.tensor([0.1, -0.2]))
    permutation = torch.tensor([3, 1, 4, 0, 2])

    direct = model(features, baseline, valid, base_mask_weights)
    permuted = model(
        features[:, permutation], baseline[:, permutation],
        valid[:, permutation], base_mask_weights[:, permutation],
    )
    inverse = permutation.argsort()
    for key in (
            "scores", "mask_fusion_weights", "mask_alpha_residual",
            "mask_logit_bias"):
        torch.testing.assert_close(
            direct[key], permuted[key][:, inverse], atol=1e-6, rtol=1e-6
        )
    assert torch.count_nonzero(direct["mask_alpha_residual"][:, 4]) == 0
    assert torch.count_nonzero(direct["mask_logit_bias"][:, 4]) == 0
    assert not bool((direct["selected_indices"] == 4).any().item())


def test_v43_source_evidence_is_query_permutation_equivariant():
    features, baseline, valid = _inputs()
    base_mask_weights = torch.rand_like(baseline)
    source_evidence = torch.rand(
        *baseline.shape, QUERY_MASK_SOURCE_EVIDENCE_DIM
    )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True, use_source_mask_evidence=True,
    ).eval()
    with torch.no_grad():
        model.quality_head.weight.normal_(mean=0.0, std=0.1)
        model.residual_head.weight.normal_(mean=0.0, std=0.1)
        model.mask_calibration_head.weight.normal_(mean=0.0, std=0.1)
    permutation = torch.tensor([3, 1, 4, 0, 2])

    direct = model(
        features, baseline, valid, base_mask_weights, source_evidence
    )
    permuted = model(
        features[:, permutation], baseline[:, permutation],
        valid[:, permutation], base_mask_weights[:, permutation],
        source_evidence[:, permutation],
    )
    inverse = permutation.argsort()
    for key in (
            "scores", "residual", "mask_fusion_weights",
            "mask_alpha_residual", "mask_logit_bias"):
        torch.testing.assert_close(
            direct[key], permuted[key][:, inverse], atol=1e-6, rtol=1e-6
        )


def test_box_tier_strictly_dominates_mask_quality():
    box = torch.tensor([[0.24, 0.26, 0.49, 0.51]])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    quality = joint_query_target_quality(box, mask, mask_weight=0.25)
    assert quality[0, 1] > quality[0, 0]
    assert quality[0, 3] > quality[0, 2]


def test_box_tier_rejects_mask_weight_that_can_cross_a_tier():
    box = torch.tensor([[0.25, 0.251]])
    mask = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="below 0.8"):
        joint_query_target_quality(box, mask, mask_weight=0.8)


def test_ordinal_threshold_probabilities_are_nested():
    raw = torch.tensor([
        [[-2.0, 4.0], [3.0, -1.0], [0.0, 0.0]],
    ])
    probability = ordinal_threshold_logits(raw).sigmoid()
    assert torch.all(probability[..., 1] <= probability[..., 0])
    torch.testing.assert_close(
        probability[..., 1],
        raw[..., 0].sigmoid() * raw[..., 1].sigmoid(),
    )


def test_listwise_ranking_directly_trains_shared_quality_head():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    )
    outputs = model(features, baseline, valid)
    box = torch.tensor([
        [0.1, 0.8, 0.4, 0.2, 0.6],
        [0.7, 0.1, 0.2, 0.9, 0.3],
    ])
    mask = torch.tensor([
        [0.2, 0.7, 0.5, 0.1, 0.8],
        [0.6, 0.2, 0.4, 0.7, 0.3],
    ])
    loss = compute_joint_query_quality_loss(
        outputs, box, mask,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
    )["loss"]
    loss.backward()
    assert model.quality_head.weight.grad.abs().sum() > 0
    assert model.residual_head.weight.grad.abs().sum() > 0


def test_multitask_loss_has_finite_nonzero_head_gradients():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    )
    outputs = model(features, baseline, valid)
    box = torch.tensor([
        [0.1, 0.8, 0.4, 0.2, 0.6],
        [0.7, 0.1, 0.2, 0.9, 0.3],
    ])
    mask = torch.tensor([
        [0.2, 0.7, 0.5, 0.1, 0.8],
        [0.6, 0.2, 0.4, 0.7, 0.3],
    ])
    losses = compute_joint_query_quality_loss(outputs, box, mask)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert model.residual_head.weight.grad.abs().sum() > 0
    assert model.quality_head.weight.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_reranker_detaches_backbone_inputs_by_default():
    features, baseline, valid = _inputs()
    features.requires_grad_(True)
    baseline.requires_grad_(True)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    )
    outputs = model(features, baseline, valid)
    outputs["scores"].sum().backward()
    assert features.grad is None
    assert baseline.grad is None


def test_residual_summary_ignores_invalid_queries_and_tracks_variation():
    residual = torch.tensor([
        [0.0, 2.0, 100.0],
        [-1.0, 1.0, 0.0],
    ])
    valid = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])
    stats = summarize_joint_query_residual(residual, valid)
    torch.testing.assert_close(
        stats["residual_abs_mean"], torch.tensor(0.8)
    )
    torch.testing.assert_close(
        stats["residual_abs_max"], torch.tensor(2.0)
    )
    expected_std = torch.tensor((1.0 + (2.0 / 3.0) ** 0.5) / 2.0)
    torch.testing.assert_close(stats["residual_query_std"], expected_std)


def test_loss_rejects_out_of_range_targets():
    features, baseline, valid = _inputs()
    outputs = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    )(features, baseline, valid)
    box = torch.zeros(2, 5)
    box[0, 0] = 1.1
    with pytest.raises(ValueError, match="IoU targets"):
        compute_joint_query_quality_loss(outputs, box, torch.zeros_like(box))


def test_empty_supervision_returns_differentiable_zero():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    )
    outputs = model(features, baseline, valid)
    loss = compute_joint_query_quality_loss(
        outputs,
        torch.zeros(2, 5),
        torch.zeros(2, 5),
        sample_mask=torch.zeros(2, dtype=torch.bool),
    )["loss"]
    assert loss.item() == 0.0
    loss.backward()


class _ZeroSetCriterion:
    def __call__(self, outputs, targets):
        del targets
        zero = outputs["pred_boxes"].sum() * 0.0
        return {
            key: zero
            for key in (
                "loss_ce", "loss_bbox", "loss_giou", "loss_mask",
                "loss_dice", "sp_loss_mask", "sp_loss_dice",
                "corresponding_loss_mask", "corresponding_loss_dice",
                "adaptive_weight_loss_mask", "adaptive_weight_loss_dice",
            )
        }, None


def _joint_loss_end_points(model):
    torch.manual_seed(19)
    batch_size, queries, targets, points, superpoint_count = 2, 4, 2, 6, 3
    features = torch.randn(batch_size, queries, 8)
    baseline = torch.tensor([
        [0.8, 0.5, 0.3, 0.1],
        [0.2, 0.7, 0.4, 0.1],
    ])
    valid = torch.ones(batch_size, queries, dtype=torch.bool)
    base_mask_weights = torch.full_like(baseline, 0.4)
    if model.use_adaptive_source_mixing:
        source_scores = torch.stack((
            baseline,
            torch.flip(baseline, dims=(1,)),
            torch.roll(baseline, shifts=1, dims=1),
        ), dim=-1)
        joint = model(
            features, baseline, valid,
            base_mask_weights=(
                base_mask_weights if model.use_mask_calibration else None
            ),
            source_score_stack=source_scores,
            source_validity=torch.ones_like(
                source_scores, dtype=torch.bool
            ),
        )
    elif model.use_mask_calibration:
        joint = model(features, baseline, valid, base_mask_weights)
    else:
        joint = model(features, baseline, valid)
    centers = torch.tensor([
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0],
         [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
        [[1.0, 0.0, 0.0], [0.7, 0.0, 0.0],
         [1.8, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
    sizes = torch.ones(batch_size, queries, 3)
    gt_centers = torch.tensor([
        [[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]],
        [[1.0, 0.0, 0.0], [4.0, 4.0, 4.0]],
    ])
    gt_sizes = torch.ones(batch_size, targets, 3)
    gt_masks = torch.zeros(batch_size, targets, points)
    gt_masks[0, 0, :4] = 1.0
    gt_masks[1, 0, 2:] = 1.0
    superpoints = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
    ], dtype=torch.long)
    text_masks = [
        torch.randn(queries, superpoint_count)
        for _ in range(batch_size)
    ]
    query_masks = [
        torch.randn(queries, superpoint_count)
        for _ in range(batch_size)
    ]
    token_map = torch.zeros(batch_size, targets, 5)
    end_points = {
        "center_label": gt_centers,
        "size_gts": gt_sizes,
        "sem_cls_label": torch.zeros(
            batch_size, targets, dtype=torch.long
        ),
        "gt_masks": gt_masks,
        "positive_map": token_map,
        "modify_positive_map": token_map.clone(),
        "pron_positive_map": token_map.clone(),
        "other_entity_map": token_map.clone(),
        "rel_positive_map": token_map.clone(),
        "box_label_mask": torch.tensor([[1, 0], [1, 0]]),
        "auxi_entity_positive_map": torch.zeros(batch_size, 1, 5),
        "auxi_box": torch.zeros(batch_size, 6),
        "proposal_center": centers,
        "proposal_pred_size": sizes,
        "proposal_sem_cls_scores": torch.zeros(batch_size, queries, 2),
        "last_center": centers,
        "last_pred_size": sizes,
        "last_sem_cls_scores": torch.zeros(batch_size, queries, 2),
        "last_pred_masks": text_masks,
        "sp_last_pred_masks": query_masks,
        "adaptive_weights": [
            torch.tensor(0.4) for _ in range(batch_size)
        ],
        "superpoints": superpoints,
        "super_xyz_list": [torch.zeros(superpoint_count, 3)] * batch_size,
        "language_dataset": ["scanrefer"] * batch_size,
        "sample_dataset": ["scanrefer"] * batch_size,
        "selected_source_scores": joint["scores"],
        "source_choice_source_scores": {"contrastive_text": baseline},
        "moe_shared_source": "contrastive_text",
        "moe_shared_query": baseline.argmax(dim=1),
        "moe_valid_mask": valid,
    }
    if model.use_mask_calibration:
        calibrated = apply_query_mask_calibration(
            text_masks, query_masks,
            joint["mask_fusion_weights"], joint["mask_logit_bias"],
        )
        end_points["last_pred_masks"] = calibrated[0]
        end_points["sp_last_pred_masks"] = calibrated[1]
        end_points["adaptive_weights"] = calibrated[2]
    for key, value in joint.items():
        end_points["joint_query_quality_{}".format(key)] = value
    return end_points


def test_joint_only_fast_loss_matches_full_loss_and_gradients():
    torch.manual_seed(23)
    fast_model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    )
    full_model = copy.deepcopy(fast_model)
    common = {
        "num_decoder_layers": 1,
        "joint_query_quality_loss_weight": 1.0,
        "joint_query_quality_mask_weight": 0.25,
        "joint_query_quality_temperature": 0.25,
        "joint_query_quality_aux_loss_weight": 1.0,
        "joint_query_quality_anchor_loss_weight": 0.5,
        "joint_query_quality_anchor_margin": 0.05,
        "source_moe_balance_loss_weight": 0.0,
        "source_moe_rank_loss_weight": 0.0,
        "source_moe_gate_loss_weight": 0.0,
    }
    fast_loss, fast_end_points = compute_hungarian_loss(
        _joint_loss_end_points(fast_model),
        set_criterion=None,
        joint_query_quality_train_only=True,
        **common
    )
    full_loss, full_end_points = compute_hungarian_loss(
        _joint_loss_end_points(full_model),
        set_criterion=_ZeroSetCriterion(),
        joint_query_quality_train_only=False,
        **common
    )
    torch.testing.assert_close(fast_loss, full_loss)
    for key in (
            "joint_query_quality_loss",
            "joint_query_quality_listwise_loss",
            "joint_query_quality_aux_loss",
            "joint_query_quality_anchor_loss"):
        torch.testing.assert_close(fast_end_points[key], full_end_points[key])

    fast_loss.backward()
    full_loss.backward()
    for (fast_name, fast_parameter), (full_name, full_parameter) in zip(
            fast_model.named_parameters(), full_model.named_parameters()):
        assert fast_name == full_name
        if fast_parameter.grad is None or full_parameter.grad is None:
            assert fast_parameter.grad is None
            assert full_parameter.grad is None
        else:
            torch.testing.assert_close(
                fast_parameter.grad, full_parameter.grad
            )


def test_source_mix_alignment_is_active_in_fast_training_path():
    torch.manual_seed(24)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_adaptive_source_mixing=True,
        source_count=3, shared_source_index=0,
    )
    loss, end_points = compute_hungarian_loss(
        _joint_loss_end_points(model),
        num_decoder_layers=1,
        set_criterion=None,
        joint_query_quality_train_only=True,
        joint_query_quality_loss_weight=1.0,
        joint_query_quality_source_mix_loss_weight=0.25,
        joint_query_quality_source_mix_alignment_temperature=0.25,
        source_moe_balance_loss_weight=0.0,
        source_moe_rank_loss_weight=0.0,
        source_moe_gate_loss_weight=0.0,
    )

    assert end_points[
        "joint_query_quality_source_mix_alignment_loss"
    ].item() > 0.0
    loss.backward()
    gradient = model.adaptive_source_mixer.source_router[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


class _MaskCalibrationCriterion:
    def forward_query_mask_fusion(self, outputs, targets):
        del targets
        loss = outputs["pred_boxes"].sum() * 0.0
        for text, query, alpha in zip(
                outputs["pred_masks"], outputs["sp_pred_masks"],
                outputs["adaptive_weights"]):
            fused = (
                alpha.unsqueeze(-1) * text
                + (1.0 - alpha.unsqueeze(-1)) * query
            )
            loss = loss + fused.square().mean()
        return {
            "adaptive_weight_loss_mask": loss,
            "adaptive_weight_loss_dice": loss * 0.0,
        }, None


def test_v42_fast_mask_loss_backpropagates_to_calibration_head():
    torch.manual_seed(29)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    )
    loss, end_points = compute_hungarian_loss(
        _joint_loss_end_points(model),
        num_decoder_layers=1,
        set_criterion=_MaskCalibrationCriterion(),
        joint_query_quality_train_only=True,
        joint_query_quality_loss_weight=1.0,
        joint_query_quality_mask_weight=0.25,
        joint_query_quality_temperature=0.25,
        joint_query_quality_aux_loss_weight=1.0,
        joint_query_quality_anchor_loss_weight=0.5,
        joint_query_quality_anchor_margin=0.05,
        source_moe_balance_loss_weight=0.0,
        source_moe_rank_loss_weight=0.0,
        source_moe_gate_loss_weight=0.0,
    )
    assert end_points["adaptive_weight_loss_mask"].item() > 0.0

    loss.backward()

    gradient = model.mask_calibration_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert bool((gradient.abs().sum(dim=1) > 0.0).all().item())


def test_candidate_mask_selection_unions_deployed_and_box_oracle_queries():
    scores = torch.tensor([
        [0.9, 0.8, 0.1, 4.0],
        [0.1, 0.9, 0.8, 0.7],
    ])
    box_ious = torch.tensor([
        [0.1, 0.2, 0.95, 1.0],
        [0.9, 0.1, 0.2, 0.3],
    ])
    valid = torch.tensor([
        [True, True, True, False],
        [True, True, True, True],
    ])
    sample_mask = torch.tensor([True, False])

    selected = build_joint_query_mask_candidate_mask(
        scores, box_ious, valid, sample_mask, top_k=1
    )

    assert torch.equal(
        selected,
        torch.tensor([
            [True, False, True, False],
            [False, False, False, False],
        ]),
    )


def test_lovasz_hinge_has_exact_single_pixel_margin_and_gradient():
    logits = torch.tensor([[0.0]], requires_grad=True)
    targets = torch.ones_like(logits)

    loss = lovasz_hinge_loss(logits, targets, num_masks=1)

    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-1.0]]))


def test_lovasz_hinge_is_zero_past_margin_and_permutation_invariant():
    logits = torch.tensor([
        [2.0, -3.0, 4.0, -2.0],
        [0.3, -0.4, -1.2, 1.5],
    ])
    targets = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    permutation = torch.tensor([2, 0, 3, 1])

    perfect = lovasz_hinge_loss(
        logits[:1], targets[:1], num_masks=1
    )
    original = lovasz_hinge_loss(logits, targets, num_masks=2)
    permuted = lovasz_hinge_loss(
        logits.index_select(1, permutation),
        targets.index_select(1, permutation),
        num_masks=2,
    )

    assert perfect.item() == 0.0
    torch.testing.assert_close(original, permuted, rtol=0.0, atol=0.0)


def test_candidate_mask_loss_updates_only_selected_alpha_and_bias_queries():
    text = [torch.tensor([
        [3.0, -2.0, -1.0, 2.0],
        [-1.0, 2.0, 3.0, -2.0],
        [-2.0, 3.0, -2.0, 3.0],
    ])]
    query = [torch.tensor([
        [-2.0, 3.0, 2.0, -1.0],
        [2.0, -1.0, -2.0, 3.0],
        [3.0, -2.0, 3.0, -2.0],
    ])]
    alpha = torch.tensor([[0.4, 0.5, 0.6]], requires_grad=True)
    bias = torch.zeros(1, 3, requires_grad=True)
    calibrated = apply_query_mask_calibration(
        text, query, alpha, bias
    )
    gt_masks = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    superpoints = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    candidate_mask = torch.tensor([[True, False, True]])

    losses = compute_joint_query_mask_candidate_loss(
        calibrated[0], calibrated[1], calibrated[2],
        gt_masks, superpoints, candidate_mask,
    )
    loss = 10.0 * losses["mask_loss"] + 2.0 * losses["dice_loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert bool((alpha.grad[0, [0, 2]].abs() > 0.0).all().item())
    assert alpha.grad[0, 1].item() == 0.0
    assert bool((bias.grad[0, [0, 2]].abs() > 0.0).all().item())
    assert bias.grad[0, 1].item() == 0.0


def test_candidate_gather_before_fusion_matches_dense_reference_exactly():
    text = [torch.tensor([
        [3.0, -2.0, -1.0, 2.0],
        [-1.0, 2.0, 3.0, -2.0],
        [-2.0, 3.0, -2.0, 3.0],
    ])]
    query = [torch.tensor([
        [-2.0, 3.0, 2.0, -1.0],
        [2.0, -1.0, -2.0, 3.0],
        [3.0, -2.0, 3.0, -2.0],
    ])]
    fast_alpha = torch.tensor([[0.4, 0.5, 0.6]], requires_grad=True)
    fast_bias = torch.zeros(1, 3, requires_grad=True)
    dense_alpha = fast_alpha.detach().clone().requires_grad_(True)
    dense_bias = fast_bias.detach().clone().requires_grad_(True)
    gt_masks = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    superpoints = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    candidate_mask = torch.tensor([[True, False, True]])

    fast_calibrated = apply_query_mask_calibration(
        text, query, fast_alpha, fast_bias
    )
    fast_losses = compute_joint_query_mask_candidate_loss(
        fast_calibrated[0], fast_calibrated[1], fast_calibrated[2],
        gt_masks, superpoints, candidate_mask,
    )
    fast_total = (
        10.0 * fast_losses["mask_loss"]
        + 2.0 * fast_losses["dice_loss"]
    )

    dense_calibrated = apply_query_mask_calibration(
        text, query, dense_alpha, dense_bias
    )
    dense_fused = fuse_query_mask_logits(
        dense_calibrated[0][0], dense_calibrated[1][0],
        dense_calibrated[2][0],
    )[candidate_mask[0]]
    dense_target = gt_masks[0, 0].unsqueeze(0).expand_as(dense_fused)
    dense_total = (
        10.0 * sigmoid_focal_loss(dense_fused, dense_target, 2)
        + 2.0 * dice_loss(dense_fused, dense_target, 2)
    )

    torch.testing.assert_close(fast_total, dense_total, rtol=0.0, atol=0.0)
    fast_total.backward()
    dense_total.backward()
    torch.testing.assert_close(
        fast_alpha.grad, dense_alpha.grad, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        fast_bias.grad, dense_bias.grad, rtol=0.0, atol=0.0
    )


def test_v44_candidate_mask_loss_is_active_in_fast_training_path():
    torch.manual_seed(31)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    )
    loss, end_points = compute_hungarian_loss(
        _joint_loss_end_points(model),
        num_decoder_layers=1,
        set_criterion=_MaskCalibrationCriterion(),
        joint_query_quality_train_only=True,
        joint_query_quality_loss_weight=1.0,
        joint_query_quality_candidate_mask_loss_weight=0.25,
        joint_query_quality_candidate_mask_top_k=2,
        source_moe_balance_loss_weight=0.0,
        source_moe_rank_loss_weight=0.0,
        source_moe_gate_loss_weight=0.0,
    )

    assert end_points["joint_query_quality_candidate_mask_loss"].item() > 0.0
    assert end_points["joint_query_quality_candidate_dice_loss"].item() > 0.0
    ratio = end_points[
        "joint_query_quality_candidate_mask_query_ratio"
    ].item()
    assert 0.0 < ratio <= 1.0

    loss.backward()
    gradient = model.mask_calibration_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_v45_candidate_lovasz_loss_is_active_in_fast_training_path():
    torch.manual_seed(37)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    )
    loss, end_points = compute_hungarian_loss(
        _joint_loss_end_points(model),
        num_decoder_layers=1,
        set_criterion=_MaskCalibrationCriterion(),
        joint_query_quality_train_only=True,
        joint_query_quality_loss_weight=1.0,
        joint_query_quality_candidate_lovasz_loss_weight=0.1,
        joint_query_quality_candidate_mask_top_k=2,
        source_moe_balance_loss_weight=0.0,
        source_moe_rank_loss_weight=0.0,
        source_moe_gate_loss_weight=0.0,
    )

    assert end_points[
        "joint_query_quality_candidate_lovasz_loss"
    ].item() > 0.0
    loss.backward()
    gradient = model.mask_calibration_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_v48_spatial_mask_refiner_is_exact_identity_and_detaches_inputs():
    torch.manual_seed(47)
    query = torch.randn(2, 4, 8, requires_grad=True)
    superpoints = [
        torch.randn(8, 7, requires_grad=True),
        torch.randn(8, 5, requires_grad=True),
    ]
    valid = torch.tensor([
        [True, True, True, False],
        [True, False, True, True],
    ])
    refiner = QuerySuperpointMaskRefiner(
        d_model=8, hidden_dim=4, max_delta=2.0, detach_inputs=True
    )

    residuals = refiner(query, superpoints, valid)

    assert [tuple(row.shape) for row in residuals] == [(4, 7), (4, 5)]
    assert all(torch.count_nonzero(row).item() == 0 for row in residuals)
    loss = sum(row.sum() for row in residuals)
    loss.backward()
    gradient = refiner.query_projection[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0
    assert query.grad is None
    assert all(row.grad is None for row in superpoints)


def test_v48_spatial_mask_refiner_is_query_and_superpoint_equivariant():
    torch.manual_seed(48)
    query = torch.randn(1, 4, 8)
    superpoints = [torch.randn(8, 6)]
    refiner = QuerySuperpointMaskRefiner(
        d_model=8, hidden_dim=4, max_delta=2.0
    )
    with torch.no_grad():
        refiner.query_projection[-1].weight.normal_(std=0.1)
    direct = refiner(query, superpoints)[0]
    query_order = torch.tensor([2, 0, 3, 1])
    point_order = torch.tensor([5, 1, 3, 0, 4, 2])
    permuted = refiner(
        query[:, query_order],
        [superpoints[0][:, point_order]],
    )[0]

    torch.testing.assert_close(
        direct[query_order][:, point_order], permuted
    )


def test_v48_spatial_residual_corrects_final_fused_logits_exactly():
    torch.manual_seed(49)
    text = [torch.randn(3, 5)]
    query = [torch.randn(3, 5)]
    alpha = torch.tensor([0.35, 0.6, 0.8])
    residual = [torch.randn(3, 5)]

    refined_text, refined_query = apply_query_superpoint_mask_residual(
        text, query, residual
    )
    refined = fuse_query_mask_logits(
        refined_text[0], refined_query[0], alpha
    )
    expected = fuse_query_mask_logits(text[0], query[0], alpha) + residual[0]

    torch.testing.assert_close(refined, expected, rtol=0.0, atol=1e-6)


def test_v48_candidate_mask_loss_reaches_spatial_refiner():
    torch.manual_seed(50)
    features, baseline, valid = _inputs(batch=1, queries=3, features=8)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
        use_spatial_mask_refiner=True,
        spatial_mask_d_model=8,
        spatial_mask_hidden_dim=4,
    )
    outputs = model(
        features,
        baseline,
        valid,
        base_mask_weights=torch.full((1, 3), 0.5),
        spatial_query_features=torch.randn(1, 3, 8),
        spatial_superpoint_features=[torch.randn(8, 6)],
    )
    text = [torch.randn(3, 6)]
    query = [torch.randn(3, 6)]
    calibrated = apply_query_mask_calibration(
        text,
        query,
        outputs["mask_fusion_weights"],
        outputs["mask_logit_bias"],
    )
    refined = apply_query_superpoint_mask_residual(
        calibrated[0], calibrated[1], outputs["mask_spatial_residuals"]
    )
    losses = compute_joint_query_mask_candidate_loss(
        refined[0],
        refined[1],
        calibrated[2],
        torch.tensor([[[1, 0, 1, 0, 1, 0]]]),
        torch.tensor([[0, 1, 2, 3, 4, 5]]),
        torch.ones(1, 3, dtype=torch.bool),
        compute_lovasz=True,
    )
    total = losses["mask_loss"] + losses["dice_loss"] + losses[
        "lovasz_loss"
    ]
    total.backward()

    gradient = model.spatial_mask_refiner.query_projection[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


class _V48ForwardCrossEncoder(torch.nn.Module):
    def forward(self, vis_feats, pos_feats, padding_mask, text_feats,
                text_padding_mask, end_points, detected_feats,
                detected_mask, spatial_point_xyz):
        del pos_feats, padding_mask, end_points, detected_feats
        del detected_mask, spatial_point_xyz
        return vis_feats, text_feats


class _V48ForwardPosition(torch.nn.Module):
    def forward(self, xyz):
        return xyz.new_zeros(xyz.shape[0], 288, xyz.shape[1])


class _V48ForwardSuperGrouper(torch.nn.Module):
    def forward(self, xyz, super_xyz, features):
        del xyz
        count = super_xyz.shape[1]
        grouped = features[:, :, :count].unsqueeze(-1)
        indices = torch.arange(
            count, device=features.device, dtype=torch.long
        ).reshape(1, count, 1)
        return grouped, indices


class _V48ForwardDecoder(torch.nn.Module):
    def forward(self, query, points, text, query_pos, query_mask,
                text_padding_mask, detected_feats=None, detected_mask=None):
        del points, text, query_pos, query_mask, text_padding_mask
        del detected_feats, detected_mask
        return query


class _V48ForwardBoxHead(torch.nn.Module):
    def forward(self, features, base_xyz, end_points, prefix):
        batch, _, queries = features.shape
        center = base_xyz
        size = features.new_ones(batch, queries, 3)
        end_points[prefix + "center"] = center
        end_points[prefix + "pred_size"] = size
        end_points[prefix + "sem_cls_scores"] = features.new_zeros(
            batch, queries, 2
        )
        return center, size


class _V48ForwardSWA(torch.nn.Module):
    def forward(self, source, query, attn_mask=None, pe=None):
        del attn_mask, pe
        batch_query = query.transpose(0, 1)
        weights = source.new_ones(
            batch_query.shape[0], batch_query.shape[1], source.shape[0]
        )
        return batch_query, weights, weights


class _V48ForwardSelector(torch.nn.Module):
    def forward(self, candidate_feats, candidate_boxes, source_scores,
                valid_mask, text_feats, text_mask):
        del candidate_feats, candidate_boxes, source_scores
        del text_feats, text_mask
        scores = torch.linspace(
            1.0, 0.0, valid_mask.shape[1], device=valid_mask.device
        ).unsqueeze(0).expand(valid_mask.shape[0], -1)
        return {"selected_source_scores": scores}


def _v48_forward_model(monkeypatch):
    from models.mcln import MCLN

    model = object.__new__(MCLN)
    torch.nn.Module.__init__(model)
    model.num_queries = 256
    model.num_decoder_layers = 1
    model.self_position_embedding = "none"
    model.contrastive_align_loss = True
    model.butd = False
    model.source_moe_gate_use_evidence_features = False
    model.source_moe_gate_use_rich_features = True
    model.source_moe_shared_source = "default"
    model.use_joint_query_quality_reranker = True
    model.joint_query_quality_use_mask_calibration = True
    model.joint_query_quality_use_source_mask_evidence = True
    model.joint_query_quality_use_gate_evidence = False
    model.joint_query_quality_use_spatial_mask_refiner = True
    model.source_choice_selector_sources = (
        "default", "default_rank_blend_contrastive010"
    )
    model.cross_encoder = _V48ForwardCrossEncoder()
    model.pos_embed = _V48ForwardPosition()
    model.contrastive_align_projection_text = torch.nn.Linear(288, 64)
    model.contrastive_align_projection_image = torch.nn.Linear(288, 64)
    model.x_mask = torch.nn.Identity()
    model.super_grouper = _V48ForwardSuperGrouper()
    model.rel_encoder = torch.nn.Linear(3, 288, bias=False)
    torch.nn.init.zeros_(model.rel_encoder.weight)
    model.decoder_query_proj = torch.nn.Identity()
    model.decoder = torch.nn.ModuleList([_V48ForwardDecoder()])
    model.proposal_head = _V48ForwardBoxHead()
    model.prediction_heads = torch.nn.ModuleList([_V48ForwardBoxHead()])
    model.x_query = torch.nn.Identity()
    model.swa_layers = torch.nn.ModuleList([
        _V48ForwardSWA(), _V48ForwardSWA(), _V48ForwardSWA()
    ])
    model.swa_ffn_layers = torch.nn.ModuleList([
        torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()
    ])
    model.query_mask_fusion_calibrator = None
    model.source_choice_selector = _V48ForwardSelector()
    model.source_moe = None
    model.joint_query_quality_reranker = JointQueryQualityReranker(
        152, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
        use_source_mask_evidence=True,
        use_spatial_mask_refiner=True,
        spatial_mask_d_model=288,
        spatial_mask_hidden_dim=4,
    )

    def run_backbones(self, inputs):
        batch, points = inputs["point_clouds"].shape[:2]
        channel = torch.linspace(
            -1.0, 1.0, 288, device=inputs["point_clouds"].device
        ).reshape(1, 288, 1)
        point = torch.linspace(
            -0.5, 0.5, points, device=inputs["point_clouds"].device
        ).reshape(1, 1, points)
        point_features = (
            channel + point * channel.square()
        ).expand(batch, -1, -1).contiguous()
        return {
            "fp2_xyz": inputs["point_clouds"][..., :3],
            "fp2_features": point_features,
            "text_feats": inputs["point_clouds"].new_ones(batch, 4, 288),
            "text_attention_mask": torch.zeros(
                batch, 4, dtype=torch.bool,
                device=inputs["point_clouds"].device,
            ),
        }

    def generate_queries(self, xyz, features, end_points):
        end_points["query_points_xyz"] = xyz[:, :self.num_queries]
        end_points["query_points_feature"] = features[:, :, :self.num_queries]
        return end_points

    def prediction_head(self, query, superpoint_feats):
        scores = query.mean(dim=-1, keepdim=True)
        masks = torch.einsum("bqd,bsd->bqs", query, superpoint_feats)
        return scores, masks, torch.zeros_like(masks, dtype=torch.bool)

    model._run_backbones = MethodType(run_backbones, model)
    model._generate_queries = MethodType(generate_queries, model)
    model.prediction_head = MethodType(prediction_head, model)

    def source_batch(end_points, inputs, source_names,
                     include_rich_candidate_feats):
        del inputs
        assert include_rich_candidate_feats is True
        scores = end_points["last_center"].new_zeros(1, 256)
        routed_scores = torch.linspace(
            0.0, 1.0, 256, device=scores.device
        ).unsqueeze(0)
        named_scores = {
            name: scores if name == "default" else routed_scores
            for name in source_names
        }
        return {
            "candidate_feats": end_points["source_choice_candidate_feats"],
            "candidate_boxes": torch.cat((
                end_points["last_center"], end_points["last_pred_size"]
            ), dim=-1),
            "source_scores": named_scores,
            "source_validity": torch.ones(
                1, 256, len(source_names), dtype=torch.bool,
                device=scores.device,
            ),
            "valid_mask": torch.ones_like(scores, dtype=torch.bool),
            "text_feats": end_points["text_feats"],
            "text_mask": end_points["text_attention_mask"],
            "rich_candidate_feats": scores.unsqueeze(-1).expand(-1, -1, 152),
        }

    monkeypatch.setattr(
        "models.mcln.build_mcln_source_choice_batch", source_batch
    )
    return model.eval()


def test_v49_real_mcln_forward_is_identity_and_routes_all_sources(monkeypatch):
    torch.manual_seed(52)
    model = _v48_forward_model(monkeypatch)
    model.joint_query_quality_use_adaptive_source_mixing = True
    model.joint_query_quality_reranker = JointQueryQualityReranker(
        152, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
        use_source_mask_evidence=True,
        use_spatial_mask_refiner=True,
        spatial_mask_d_model=288,
        spatial_mask_hidden_dim=4,
        use_adaptive_source_mixing=True,
        source_count=2,
        shared_source_index=0,
    )
    points = 256
    outputs = model({
        "point_clouds": torch.randn(1, points, 3),
        "superpoint": torch.arange(points).remainder(8).unsqueeze(0),
        "text": ["target"],
    })

    assert torch.equal(
        outputs["selected_source_scores"],
        outputs["joint_query_quality_parent_scores"],
    )
    assert outputs["joint_query_quality_source_mix_weights"].shape == (
        1, 256, 2
    )
    assert outputs[
        "joint_query_quality_source_mix_weight_default"
    ].item() > 0.0
    assert outputs[
        "joint_query_quality_source_mix_weight_default_rank_blend_contrastive010"
    ].item() > 0.0
    assert outputs[
        "joint_query_quality_source_mix_residual_abs_mean"
    ].item() == 0.0
    (-outputs["selected_source_scores"][:, 127].mean()).backward()
    gradient = model.joint_query_quality_reranker.adaptive_source_mixer\
        .strength_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_v48_real_mcln_forward_writes_masks_and_backpropagates_candidate_loss(
        monkeypatch):
    torch.manual_seed(51)
    model = _v48_forward_model(monkeypatch)
    points = 256
    superpoints = torch.arange(points).remainder(8).unsqueeze(0)
    inputs = {
        "point_clouds": torch.randn(1, points, 3),
        "superpoint": superpoints,
        "text": ["target"],
    }

    identity = model(inputs)
    assert identity["moe_shared_source"] == "default"
    assert torch.equal(
        identity["moe_shared_query"], torch.zeros(1, dtype=torch.long)
    )
    assert torch.equal(
        identity["moe_valid_mask"], torch.ones(1, 256, dtype=torch.bool)
    )
    assert torch.count_nonzero(
        identity["joint_query_quality_mask_spatial_residuals"][0]
    ).item() == 0
    identity_masks = [row.detach().clone() for row in identity["last_pred_masks"]]

    with torch.no_grad():
        model.joint_query_quality_reranker.spatial_mask_refiner\
            .query_projection[-1].weight.normal_(std=0.1)
    refined = model(inputs)
    residual = refined["joint_query_quality_mask_spatial_residuals"][0]
    assert residual.abs().sum().item() > 0.0
    assert not torch.equal(refined["last_pred_masks"][0], identity_masks[0])
    assert refined["joint_query_quality_mask_spatial_residual_abs_mean"].item() > 0.0
    assert refined[
        "joint_query_quality_mask_spatial_superpoint_std_mean"
    ].item() > 0.0
    assert refined[
        "joint_query_quality_mask_spatial_query_std_mean"
    ].item() > 0.0

    gt_mask = torch.tensor(
        [[[1, 0] * (points // 2)]], dtype=torch.float32
    )
    token_map = torch.zeros(1, 1, 5)
    refined.update({
        "center_label": torch.zeros(1, 1, 3),
        "size_gts": torch.ones(1, 1, 3),
        "sem_cls_label": torch.zeros(1, 1, dtype=torch.long),
        "gt_masks": gt_mask,
        "positive_map": token_map,
        "modify_positive_map": token_map.clone(),
        "pron_positive_map": token_map.clone(),
        "other_entity_map": token_map.clone(),
        "rel_positive_map": token_map.clone(),
        "box_label_mask": torch.ones(1, 1, dtype=torch.long),
        "auxi_entity_positive_map": torch.zeros(1, 1, 5),
        "auxi_box": torch.zeros(1, 6),
        "language_dataset": ["scanrefer"],
        "sample_dataset": ["scanrefer"],
    })
    total, refined = compute_hungarian_loss(
        refined,
        num_decoder_layers=1,
        set_criterion=_MaskCalibrationCriterion(),
        joint_query_quality_train_only=True,
        joint_query_quality_loss_weight=1.0,
        joint_query_quality_mask_weight=0.25,
        joint_query_quality_temperature=0.25,
        joint_query_quality_aux_loss_weight=1.0,
        joint_query_quality_anchor_loss_weight=0.5,
        joint_query_quality_anchor_margin=0.05,
        joint_query_quality_candidate_mask_loss_weight=0.25,
        joint_query_quality_candidate_lovasz_loss_weight=0.1,
        joint_query_quality_candidate_mask_top_k=16,
        source_moe_balance_loss_weight=0.0,
        source_moe_rank_loss_weight=0.0,
        source_moe_gate_loss_weight=0.0,
    )
    assert refined["joint_query_quality_candidate_mask_loss"].item() > 0.0
    assert refined["joint_query_quality_candidate_dice_loss"].item() > 0.0
    assert refined["joint_query_quality_candidate_lovasz_loss"].item() > 0.0
    total.backward()
    gradient = model.joint_query_quality_reranker.spatial_mask_refiner\
        .query_projection[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_plain_selector_forward_does_not_publish_joint_training_aliases(
        monkeypatch):
    model = _v48_forward_model(monkeypatch)
    model.joint_query_quality_reranker = None
    model.use_joint_query_quality_reranker = False
    model.joint_query_quality_use_mask_calibration = False
    model.joint_query_quality_use_source_mask_evidence = False
    model.joint_query_quality_use_spatial_mask_refiner = False
    points = 256
    outputs = model({
        "point_clouds": torch.randn(1, points, 3),
        "superpoint": torch.arange(points).remainder(8).unsqueeze(0),
        "text": ["target"],
    })

    assert "selected_source_scores" in outputs
    assert "moe_shared_source" not in outputs
    assert "moe_shared_query" not in outputs
    assert "moe_valid_mask" not in outputs


def test_joint_quality_source_mixer_is_identity_then_trains_router():
    torch.manual_seed(4901)
    reranker = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_adaptive_source_mixing=True,
        source_count=3, shared_source_index=0,
    )
    features, baseline, valid = _inputs(features=8)
    source_scores = torch.randn(2, 5, 3)
    source_validity = torch.ones(2, 5, 3, dtype=torch.bool)
    source_validity[0, 2, 2] = False
    optimizer = torch.optim.SGD(
        reranker.adaptive_source_mixer.parameters(), lr=0.5
    )

    initial = reranker(
        features, baseline, valid,
        source_score_stack=source_scores,
        source_validity=source_validity,
    )
    assert torch.equal(initial["scores"], baseline)
    assert torch.count_nonzero(
        initial["source_mix_residual_logit"]
    ).item() == 0
    assert initial["source_mix_weights"][0, 2, 2].item() == 0.0
    (-initial["scores"][:, 2].mean()).backward()
    strength_gradient = reranker.adaptive_source_mixer.strength_head[-1]\
        .weight.grad
    assert strength_gradient is not None
    assert strength_gradient.abs().sum().item() > 0.0
    optimizer.step()
    optimizer.zero_grad()

    updated = reranker(
        features, baseline, valid,
        source_score_stack=source_scores,
        source_validity=source_validity,
    )
    assert updated["source_mix_residual_logit"].abs().sum().item() > 0.0
    (-updated["scores"][:, 2].mean()).backward()
    router_gradient = reranker.adaptive_source_mixer.source_router[-1]\
        .weight.grad
    assert router_gradient is not None
    assert router_gradient.abs().sum().item() > 0.0


def test_joint_quality_source_mixer_is_routed_source_permutation_equivariant():
    torch.manual_seed(4902)
    mixer = JointQualityAdaptiveSourceMixer(
        hidden_dim=8, source_count=3, shared_index=0,
    )
    with torch.no_grad():
        mixer.source_router[-1].weight.normal_()
        mixer.strength_head[-1].weight.normal_()
    hidden = torch.randn(2, 5, 8)
    quality = torch.rand(2, 5, 6)
    parent = torch.randn(2, 5)
    scores = torch.randn(2, 5, 3)
    validity = torch.ones(2, 5, 3, dtype=torch.bool)
    valid = torch.ones(2, 5, dtype=torch.bool)

    original = mixer(hidden, quality, parent, scores, validity, valid)
    permutation = torch.tensor([0, 2, 1])
    changed = mixer(
        hidden, quality, parent,
        scores[..., permutation], validity[..., permutation], valid,
    )

    assert torch.allclose(
        changed["source_mix_residual_logit"],
        original["source_mix_residual_logit"], atol=1e-6,
    )
    assert torch.allclose(
        changed["source_mix_weights"],
        original["source_mix_weights"][..., permutation], atol=1e-6,
    )


def test_source_distribution_reliability_is_finite_and_fails_closed():
    scores = torch.tensor([[[3.0, 1.0, 0.0],
                            [1.0, 2.0, 0.0],
                            [0.0, 0.0, 0.0]]])
    validity = torch.ones_like(scores, dtype=torch.bool)
    validity[..., 2] = False

    features = build_source_distribution_reliability_features(
        scores, validity, shared_index=0
    )

    assert features.shape == (1, 3, 3, SOURCE_DISTRIBUTION_RELIABILITY_DIM)
    assert torch.isfinite(features).all()
    assert torch.count_nonzero(features[..., 2, :]).item() == 0
    assert torch.count_nonzero(features[..., 0, 4:]).item() == 0
    assert bool((features[..., :2, 1:][validity[..., :2].unsqueeze(-1)
                .expand(-1, -1, -1, 5)] >= 0.0).all().item())


def test_distribution_reliability_mixer_preserves_identity_and_permutation():
    torch.manual_seed(4906)
    mixer = JointQualityAdaptiveSourceMixer(
        hidden_dim=8, source_count=3, shared_index=0,
        use_distribution_reliability=True,
    )
    hidden = torch.randn(2, 5, 8)
    quality = torch.rand(2, 5, 6)
    parent = torch.randn(2, 5)
    scores = torch.randn(2, 5, 3)
    validity = torch.ones(2, 5, 3, dtype=torch.bool)
    valid = torch.ones(2, 5, dtype=torch.bool)

    identity = mixer(hidden, quality, parent, scores, validity, valid)
    assert torch.count_nonzero(
        identity["source_mix_residual_logit"]
    ).item() == 0
    assert identity["source_mix_distribution_reliability"].shape == (
        2, 5, 3, SOURCE_DISTRIBUTION_RELIABILITY_DIM
    )

    with torch.no_grad():
        mixer.source_router[-1].weight.normal_()
        mixer.strength_head[-1].weight.normal_()
    original = mixer(hidden, quality, parent, scores, validity, valid)
    permutation = torch.tensor([0, 2, 1])
    changed = mixer(
        hidden, quality, parent,
        scores[..., permutation], validity[..., permutation], valid,
    )
    torch.testing.assert_close(
        changed["source_mix_residual_logit"],
        original["source_mix_residual_logit"], atol=1e-6, rtol=0.0,
    )
    torch.testing.assert_close(
        changed["source_mix_weights"],
        original["source_mix_weights"][..., permutation],
        atol=1e-6, rtol=0.0,
    )
    torch.testing.assert_close(
        changed["source_mix_distribution_reliability"],
        original["source_mix_distribution_reliability"][..., permutation, :],
        atol=1e-6, rtol=0.0,
    )
    original["source_mix_weights"][..., 1].sum().backward()
    reliability_gradient = mixer.source_encoder[0].weight.grad[
        :, -SOURCE_DISTRIBUTION_RELIABILITY_DIM:
    ]
    assert torch.isfinite(reliability_gradient).all()
    assert reliability_gradient.abs().sum().item() > 0.0


def test_distribution_reliability_is_optional_and_v50_shape_is_unchanged():
    baseline = JointQualityAdaptiveSourceMixer(
        hidden_dim=8, source_count=4, shared_index=0
    )
    enhanced = JointQualityAdaptiveSourceMixer(
        hidden_dim=8, source_count=4, shared_index=0,
        use_distribution_reliability=True,
    )
    assert baseline.source_encoder[0].in_features == 8 + 9
    assert enhanced.source_encoder[0].in_features == (
        8 + 9 + SOURCE_DISTRIBUTION_RELIABILITY_DIM
    )
    with pytest.raises(ValueError, match="requires adaptive source mixing"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4,
            use_source_distribution_reliability=True,
        )


def test_source_mix_alignment_trains_router_at_step_zero_identity():
    torch.manual_seed(4903)
    reranker = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_adaptive_source_mixing=True,
        source_count=3, shared_source_index=0,
    )
    features, baseline, valid = _inputs(features=8)
    source_scores = torch.stack((
        baseline,
        torch.flip(baseline, dims=(1,)),
        torch.roll(baseline, shifts=1, dims=1),
    ), dim=-1)
    source_validity = torch.ones_like(source_scores, dtype=torch.bool)
    outputs = reranker(
        features, baseline, valid,
        source_score_stack=source_scores,
        source_validity=source_validity,
    )
    box_ious = torch.tensor([
        [0.9, 0.1, 0.7, 0.2, 0.4],
        [0.2, 0.8, 0.3, 0.6, 0.1],
    ])
    mask_ious = torch.tensor([
        [0.8, 0.2, 0.6, 0.1, 0.5],
        [0.1, 0.7, 0.4, 0.5, 0.2],
    ])

    supervision = compute_joint_query_quality_loss(
        outputs, box_ious, mask_ious,
        source_mix_loss_weight=0.25,
        source_mix_alignment_temperature=0.25,
        source_mix_query_focus_weight=0.75,
    )

    assert torch.equal(outputs["scores"], baseline)
    assert supervision["source_mix_alignment_loss"].item() > 0.0
    supervision["loss"].backward()
    gradient = reranker.adaptive_source_mixer.source_router[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_source_mix_alignment_reaches_trainable_source_at_step_zero():
    torch.manual_seed(4905)
    reranker = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_adaptive_source_mixing=True,
        source_count=4, shared_source_index=0,
        detach_inputs=False,
    )
    features, baseline, valid = _inputs(features=8)
    structured_scores = torch.randn(2, 5, requires_grad=True)
    source_scores = torch.stack((
        baseline.detach(),
        torch.flip(baseline.detach(), dims=(1,)),
        torch.roll(baseline.detach(), shifts=1, dims=1),
        structured_scores,
    ), dim=-1)
    source_validity = torch.ones_like(source_scores, dtype=torch.bool)
    outputs = reranker(
        features, baseline, valid,
        source_score_stack=source_scores,
        source_validity=source_validity,
    )
    box_ious = torch.tensor([
        [0.9, 0.1, 0.7, 0.2, 0.4],
        [0.2, 0.8, 0.3, 0.6, 0.1],
    ])
    mask_ious = torch.tensor([
        [0.8, 0.2, 0.6, 0.1, 0.5],
        [0.1, 0.7, 0.4, 0.5, 0.2],
    ])
    supervision = compute_joint_query_quality_loss(
        outputs, box_ious, mask_ious,
        source_mix_loss_weight=0.25,
        source_mix_alignment_temperature=0.25,
        source_mix_query_focus_weight=0.75,
    )

    assert torch.equal(outputs["scores"], baseline)
    supervision["loss"].backward()
    assert structured_scores.grad is not None
    assert torch.isfinite(structured_scores.grad).all()
    assert structured_scores.grad.abs().sum().item() > 0.0


def test_source_mix_alignment_is_routed_source_permutation_invariant():
    torch.manual_seed(4904)
    mixer = JointQualityAdaptiveSourceMixer(
        hidden_dim=8, source_count=3, shared_index=0,
    )
    with torch.no_grad():
        mixer.source_router[-1].weight.normal_()
    hidden = torch.randn(2, 5, 8)
    quality_evidence = torch.rand(2, 5, 6)
    parent = torch.randn(2, 5)
    source_scores = torch.randn(2, 5, 3)
    source_validity = torch.ones(2, 5, 3, dtype=torch.bool)
    valid = torch.ones(2, 5, dtype=torch.bool)
    target = torch.rand(2, 5)
    original = mixer(
        hidden, quality_evidence, parent, source_scores,
        source_validity, valid,
    )
    permutation = torch.tensor([0, 2, 1])
    changed = mixer(
        hidden, quality_evidence, parent,
        source_scores[..., permutation],
        source_validity[..., permutation], valid,
    )

    original_loss = compute_joint_query_source_mix_alignment_loss(
        dict(original, valid_mask=valid), target,
        query_relevance=torch.softmax(target / 0.25, dim=1),
        query_focus_weight=0.75,
    )
    changed_loss = compute_joint_query_source_mix_alignment_loss(
        dict(changed, valid_mask=valid), target,
        query_relevance=torch.softmax(target / 0.25, dim=1),
        query_focus_weight=0.75,
    )

    torch.testing.assert_close(
        changed_loss["loss"], original_loss["loss"], atol=1e-7, rtol=0.0
    )
    torch.testing.assert_close(
        changed_loss["target_top1_acc"],
        original_loss["target_top1_acc"], atol=0.0, rtol=0.0
    )


def test_source_mix_alignment_focus_prioritizes_high_quality_queries():
    outputs = {
        "source_mix_weights": torch.tensor([[
            [0.1, 0.9],
            [0.1, 0.9],
        ]]),
        "source_mix_ranks": torch.tensor([[
            [1.0, 0.0],
            [1.0, 0.0],
        ]]),
        "source_mix_validity": torch.ones(1, 2, 2, dtype=torch.bool),
        "valid_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    target_quality = torch.tensor([[1.0, 0.0]])
    relevance = torch.tensor([[0.99, 0.01]])

    uniform = compute_joint_query_source_mix_alignment_loss(
        outputs, target_quality, query_focus_weight=0.0,
    )
    focused = compute_joint_query_source_mix_alignment_loss(
        outputs, target_quality, query_relevance=relevance,
        query_focus_weight=0.75,
    )

    assert focused["loss"] > uniform["loss"]


def test_source_mix_alignment_zero_focus_exactly_matches_v49_objective():
    outputs = {
        "source_mix_weights": torch.tensor([[
            [0.7, 0.3],
            [0.4, 0.6],
            [0.2, 0.8],
        ]]),
        "source_mix_ranks": torch.tensor([[
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ]]),
        "source_mix_validity": torch.ones(1, 3, 2, dtype=torch.bool),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    target_quality = torch.tensor([[0.9, 0.4, 0.1]])
    target_rank = torch.tensor([[1.0, 0.5, 0.0]])
    target_logits = -(
        outputs["source_mix_ranks"] - target_rank.unsqueeze(-1)
    ).abs() / 0.25
    target_weights = torch.softmax(target_logits, dim=-1)
    target_weights = target_weights / target_weights.sum(
        dim=-1, keepdim=True
    ).clamp(min=1e-6)
    expected = -(
        target_weights
        * outputs["source_mix_weights"].clamp(min=1e-8).log()
    ).sum(dim=-1).mean()

    observed = compute_joint_query_source_mix_alignment_loss(
        outputs,
        target_quality,
        temperature=0.25,
        query_focus_weight=0.0,
    )["loss"]

    torch.testing.assert_close(observed, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan")])
def test_source_mix_alignment_rejects_invalid_query_focus(weight):
    outputs = {
        "source_mix_weights": torch.full((1, 2, 2), 0.5),
        "source_mix_ranks": torch.tensor([[
            [1.0, 0.0],
            [0.0, 1.0],
        ]]),
        "source_mix_validity": torch.ones(1, 2, 2, dtype=torch.bool),
        "valid_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="query_focus_weight"):
        compute_joint_query_source_mix_alignment_loss(
            outputs, torch.tensor([[1.0, 0.0]]),
            query_relevance=torch.tensor([[0.9, 0.1]]),
            query_focus_weight=weight,
        )


def test_v41_v42_v43_v46_v48_and_v49_parameter_contracts_are_exact():
    v41 = JointQueryQualityReranker(152, hidden_dim=128, num_heads=4)
    v42 = JointQueryQualityReranker(
        152, hidden_dim=128, num_heads=4, use_mask_calibration=True
    )
    v43 = JointQueryQualityReranker(
        152, hidden_dim=128, num_heads=4, use_mask_calibration=True,
        use_source_mask_evidence=True,
    )
    v46 = JointQueryQualityReranker(
        152, hidden_dim=128, num_heads=4, use_mask_calibration=True,
        use_source_mask_evidence=True, use_gate_evidence=True,
    )
    v48 = JointQueryQualityReranker(
        152, hidden_dim=128, num_heads=4, use_mask_calibration=True,
        use_source_mask_evidence=True,
        use_spatial_mask_refiner=True,
    )
    v49 = JointQueryQualityReranker(
        152, hidden_dim=128, num_heads=4, use_mask_calibration=True,
        use_source_mask_evidence=True,
        use_spatial_mask_refiner=True,
        use_adaptive_source_mixing=True,
        source_count=3, shared_source_index=0,
    )

    assert len(v41.state_dict()) == 20
    assert sum(parameter.numel() for parameter in v41.parameters()) == 153531
    assert len(v42.state_dict()) == 22
    assert sum(parameter.numel() for parameter in v42.parameters()) == 153919
    assert len(v43.state_dict()) == 22
    assert sum(parameter.numel() for parameter in v43.parameters()) == 155219
    assert len(v46.state_dict()) == 22
    assert sum(parameter.numel() for parameter in v46.parameters()) == 158339
    assert len(v48.state_dict()) == 34
    assert sum(parameter.numel() for parameter in v48.parameters()) == 176979
    assert len(v49.state_dict()) == 45
    assert sum(parameter.numel() for parameter in v49.parameters()) == 229460


class _TrainModeToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(3, 3), torch.nn.Dropout(0.5)
        )
        self.joint_query_quality_reranker = JointQueryQualityReranker(
            3, hidden_dim=8, num_heads=2, dropout=0.1
        )


def _joint_only_args(**overrides):
    values = {
        "source_choice_selector_train_only": False,
        "source_moe_train_only": False,
        "source_moe_gate_train_only": False,
        "source_moe_gate_new_heads_only": False,
        "query_mask_fusion_train_only": False,
        "joint_query_quality_train_only": True,
        "use_joint_query_quality_reranker": True,
        "joint_query_quality_lr": 3e-4,
        "lr": 2e-5,
        "lr_backbone": 2e-4,
        "text_encoder_lr": 3e-6,
        "weight_decay": 5e-4,
        "frozen": False,
        "small_lr": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_joint_only_optimizer_contains_exactly_the_new_module():
    model = _TrainModeToy()
    optimizer = BaseTrainTester.get_optimizer(_joint_only_args(), model)
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {
        id(parameter)
        for parameter in model.joint_query_quality_reranker.parameters()
    }
    assert optimized == expected
    assert all(
        parameter.requires_grad
        for parameter in model.joint_query_quality_reranker.parameters()
    )
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_joint_only_train_mode_keeps_frozen_backbone_in_eval():
    model = _TrainModeToy().train()
    BaseTrainTester._set_source_moe_train_mode(model, _joint_only_args())
    assert model.training is False
    assert model.backbone.training is False
    assert model.joint_query_quality_reranker.training is True


def test_sacr_joint_only_optimizer_and_train_mode_include_structured_modules():
    model = _TrainModeToy().train()
    model.structured_slot_builder = torch.nn.Sequential(
        torch.nn.Linear(3, 3), torch.nn.Dropout(0.5)
    )
    model.sacr_head = torch.nn.Sequential(
        torch.nn.Linear(3, 3), torch.nn.Dropout(0.5)
    )
    model.sacr_residual_scale = torch.nn.Parameter(torch.tensor([0.1]))
    args = _joint_only_args(use_sacr_source=True)
    optimizer = BaseTrainTester.get_optimizer(args, model)
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith((
            "joint_query_quality_reranker.",
            "structured_slot_builder.",
            "sacr_head.",
            "sacr_residual_scale",
        ))
    }
    assert optimized == expected

    BaseTrainTester._set_source_moe_train_mode(model, args)
    assert model.training is False
    assert model.backbone.training is False
    assert model.joint_query_quality_reranker.training is True
    assert model.structured_slot_builder.training is True
    assert model.sacr_head.training is True
