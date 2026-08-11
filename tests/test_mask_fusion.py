import torch
from torch import nn

from main_utils import BaseTrainTester
from models.losses import SetCriterion
from models.mask_fusion import (
    QUERY_MASK_SOURCE_EVIDENCE_DIM,
    QUERY_MASK_SOURCE_EVIDENCE_NAMES,
    QueryMaskFusionCalibrator,
    apply_query_mask_calibration,
    build_query_mask_source_evidence,
    fuse_batched_query_mask_logits,
    fuse_query_mask_logits,
    gather_query_fusion_weight,
)


def test_query_mask_source_evidence_is_bounded_source_specific_and_deployable():
    text_probability = torch.tensor([
        [[0.2, 0.8, 0.5], [0.5, 0.5, 0.5]],
        [[0.1, 0.4, 0.7], [0.9, 0.6, 0.3]],
    ])
    query_probability = torch.tensor([
        [[0.8, 0.2, 0.5], [0.5, 0.5, 0.5]],
        [[0.2, 0.5, 0.8], [0.7, 0.4, 0.1]],
    ])
    text = [torch.logit(row) for row in text_probability]
    query = [torch.logit(row) for row in query_probability]

    evidence = build_query_mask_source_evidence(text, query)

    assert QUERY_MASK_SOURCE_EVIDENCE_DIM == 10
    assert len(set(QUERY_MASK_SOURCE_EVIDENCE_NAMES)) == 10
    assert evidence.shape == (2, 2, 10)
    assert torch.isfinite(evidence).all()
    assert bool(((evidence >= 0.0) & (evidence <= 1.0)).all().item())
    torch.testing.assert_close(evidence[..., 0], text_probability.mean(-1))
    torch.testing.assert_close(evidence[..., 1], query_probability.mean(-1))
    torch.testing.assert_close(
        evidence[..., 8],
        (text_probability - query_probability).abs().mean(-1),
    )
    torch.testing.assert_close(
        evidence[..., 9],
        ((text_probability > 0.5) != (query_probability > 0.5)).float().mean(-1),
    )


def test_query_mask_source_evidence_supports_variable_superpoint_counts():
    text = [torch.randn(3, 7), torch.randn(3, 4)]
    query = [torch.randn(3, 7), torch.randn(3, 4)]

    evidence = build_query_mask_source_evidence(text, query)

    assert evidence.shape == (2, 3, QUERY_MASK_SOURCE_EVIDENCE_DIM)


def test_scalar_fusion_preserves_legacy_formula_exactly():
    text = torch.randn(5, 9)
    query = torch.randn(5, 9)
    alpha = torch.tensor(0.37)

    actual = fuse_query_mask_logits(text, query, alpha)
    expected = alpha * text + (1.0 - alpha) * query

    assert torch.equal(actual, expected)


def test_query_fusion_and_gather_follow_original_query_indices():
    text = torch.zeros(4, 3)
    query = torch.ones(4, 3)
    alpha = torch.tensor([0.0, 0.25, 0.75, 1.0])
    indices = torch.tensor([3, 1])

    gathered = gather_query_fusion_weight(alpha, indices, 4, text)
    fused = fuse_query_mask_logits(
        text.index_select(0, indices),
        query.index_select(0, indices),
        gathered,
    )

    assert gathered.shape == (2, 1)
    assert torch.equal(gathered[:, 0], torch.tensor([1.0, 0.25]))
    assert torch.equal(fused[:, 0], torch.tensor([0.0, 0.75]))


def test_batched_fusion_accepts_per_query_weights():
    text = torch.zeros(2, 3, 4)
    query = torch.ones(2, 3, 4)
    alpha = torch.tensor([
        [0.0, 0.5, 1.0],
        [1.0, 0.25, 0.75],
    ])

    fused = fuse_batched_query_mask_logits(text, query, alpha)

    assert torch.equal(fused[..., 0], 1.0 - alpha)


def test_query_mask_bias_is_equivalent_after_source_fusion():
    torch.manual_seed(5)
    text = [torch.randn(3, 7), torch.randn(3, 5)]
    query = [torch.randn(3, 7), torch.randn(3, 5)]
    alpha = torch.tensor([
        [0.2, 0.5, 0.8],
        [0.7, 0.4, 0.1],
    ])
    bias = torch.tensor([
        [0.1, -0.2, 0.3],
        [-0.4, 0.2, 0.5],
    ])

    calibrated_text, calibrated_query, calibrated_alpha = (
        apply_query_mask_calibration(text, query, alpha, bias)
    )

    for batch_idx in range(2):
        original = fuse_query_mask_logits(
            text[batch_idx], query[batch_idx], alpha[batch_idx]
        )
        calibrated = fuse_query_mask_logits(
            calibrated_text[batch_idx],
            calibrated_query[batch_idx],
            calibrated_alpha[batch_idx],
        )
        torch.testing.assert_close(
            calibrated, original + bias[batch_idx].unsqueeze(-1)
        )


def _calibrator_inputs(requires_grad=False):
    query = torch.randn(2, 4, 8, requires_grad=requires_grad)
    text = torch.randn(2, 5, 8, requires_grad=requires_grad)
    padding = torch.tensor([
        [False, False, False, True, True],
        [False, False, True, True, True],
    ])
    boxes = torch.randn(2, 4, 6, requires_grad=requires_grad)
    boxes = torch.cat((boxes[..., :3], boxes[..., 3:].abs() + 0.1), -1)
    alpha = torch.tensor([0.3, 0.7], requires_grad=requires_grad)
    return query, text, padding, boxes, alpha


def test_zero_initialized_calibrator_is_bitwise_identity():
    module = QueryMaskFusionCalibrator(
        d_model=8, hidden_dim=12, dropout=0.0, max_delta=0.2
    )
    query, text, padding, boxes, alpha = _calibrator_inputs()

    output = module(query, text, padding, boxes, alpha)

    expected = alpha.unsqueeze(1).expand_as(output["weights"])
    assert torch.equal(output["weights"], expected)
    assert torch.count_nonzero(output["residual"]) == 0


def test_calibrator_detaches_backbone_inputs_and_trains_residual_head():
    module = QueryMaskFusionCalibrator(
        d_model=8, hidden_dim=12, dropout=0.0, max_delta=0.2,
        detach_inputs=True,
    )
    query, text, padding, boxes, alpha = _calibrator_inputs(
        requires_grad=True
    )

    output = module(query, text, padding, boxes, alpha)
    output["weights"].sum().backward()

    assert query.grad is None
    assert text.grad is None
    assert alpha.grad is None
    assert module.residual_head.weight.grad is not None
    assert torch.isfinite(module.residual_head.weight.grad).all()
    assert torch.count_nonzero(module.residual_head.weight.grad) > 0


def test_nonzero_calibrator_head_produces_query_specific_weights():
    module = QueryMaskFusionCalibrator(
        d_model=8, hidden_dim=12, dropout=0.0, max_delta=0.2
    )
    with torch.no_grad():
        module.residual_head.weight.normal_(mean=0.0, std=0.1)
        module.residual_head.bias.zero_()
    query, text, padding, boxes, alpha = _calibrator_inputs()

    weights = module(query, text, padding, boxes, alpha)["weights"]

    assert torch.isfinite(weights).all()
    assert bool((weights.std(dim=1) > 0).all().item())


def test_calibrator_only_training_keeps_frozen_model_state_unchanged():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(8, 8),
                nn.BatchNorm1d(8),
                nn.Dropout(0.5),
            )
            self.query_mask_fusion_calibrator = QueryMaskFusionCalibrator(
                d_model=8, hidden_dim=12, dropout=0.2, max_delta=0.2
            )

        def forward(self, query, text, padding, boxes, alpha):
            batch_size, query_count, feature_dim = query.shape
            query = self.backbone(
                query.reshape(batch_size * query_count, feature_dim)
            ).reshape(batch_size, query_count, feature_dim)
            return self.query_mask_fusion_calibrator(
                query, text, padding, boxes, alpha
            )["weights"]

    model = ToyModel()
    args = type("Args", (), {
        "source_moe_train_only": False,
        "source_moe_gate_train_only": False,
        "source_moe_gate_new_heads_only": False,
        "query_mask_fusion_train_only": True,
    })()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(
            "query_mask_fusion_calibrator."
        )
    frozen_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("query_mask_fusion_calibrator.")
    }

    BaseTrainTester._set_source_moe_train_mode(model, args)

    assert model.training is False
    assert model.backbone.training is False
    assert model.backbone[1].training is False
    assert model.backbone[2].training is False
    assert model.query_mask_fusion_calibrator.training is True
    assert all(
        not module.training
        for name, module in model.named_modules()
        if name and not name.startswith("query_mask_fusion_calibrator")
    )

    optimizer = torch.optim.Adam(
        model.query_mask_fusion_calibrator.parameters(), lr=1e-3
    )
    query, text, padding, boxes, alpha = _calibrator_inputs()
    optimizer.zero_grad()
    model(query, text, padding, boxes, alpha).sum().backward()
    optimizer.step()

    for name, expected in frozen_state.items():
        assert torch.equal(model.state_dict()[name], expected), name


def test_query_fusion_fast_loss_matches_full_mask_loss_and_gradient():
    class FixedMatcher:
        def __call__(self, _outputs, _targets):
            return [
                (torch.tensor([1]), torch.tensor([0])),
                (torch.tensor([0]), torch.tensor([0])),
            ]

    criterion = SetCriterion(FixedMatcher(), losses=["masks"])
    full_weights = torch.tensor(
        [[0.2, 0.7], [0.6, 0.4]], requires_grad=True
    )
    fast_weights = full_weights.detach().clone().requires_grad_(True)
    text_masks = [
        torch.tensor([[0.1, -0.2, 0.4], [0.3, 0.2, -0.5]]),
        torch.tensor([[-0.4, 0.6, 0.1], [0.2, -0.1, 0.3]]),
    ]
    query_masks = [
        torch.tensor([[0.5, 0.1, -0.3], [-0.2, 0.4, 0.7]]),
        torch.tensor([[0.3, -0.6, 0.8], [-0.5, 0.2, 0.4]]),
    ]
    superpoints = torch.tensor([[0, 1, 2, 2], [0, 1, 1, 2]])
    targets = [
        {"masks": torch.tensor([[1, 0, 1, 1]])},
        {"masks": torch.tensor([[0, 1, 1, 0]])},
    ]
    indices = FixedMatcher()(None, None)
    num_boxes = torch.tensor([2.0])

    full_outputs = {
        "pred_masks": text_masks,
        "sp_pred_masks": query_masks,
        "adaptive_weights": full_weights,
        "superpoints": superpoints,
        "super_xyz_list": [torch.zeros(1, 3, 3), torch.zeros(1, 3, 3)],
    }
    fast_outputs = dict(full_outputs, adaptive_weights=fast_weights)
    full = criterion.loss_masks(
        full_outputs, targets, indices, num_boxes, None
    )
    fast = criterion.loss_query_mask_fusion(
        fast_outputs, targets, indices, num_boxes
    )

    assert torch.equal(
        fast["adaptive_weight_loss_mask"],
        full["adaptive_weight_loss_mask"],
    )
    assert torch.equal(
        fast["adaptive_weight_loss_dice"],
        full["adaptive_weight_loss_dice"],
    )
    full_loss = (
        10 * full["adaptive_weight_loss_mask"]
        + 2 * full["adaptive_weight_loss_dice"]
    )
    fast_loss = (
        10 * fast["adaptive_weight_loss_mask"]
        + 2 * fast["adaptive_weight_loss_dice"]
    )
    full_gradient = torch.autograd.grad(full_loss, full_weights)[0]
    fast_gradient = torch.autograd.grad(fast_loss, fast_weights)[0]
    assert torch.equal(fast_gradient, full_gradient)
