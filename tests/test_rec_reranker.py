import pytest
import torch

from models.rec_reranker import (
    QueryReranker,
    blend_candidate_scores,
    compute_candidate_oracle,
    compute_query_ious,
    compute_rec_reranker_loss,
    select_listwise_targets,
    select_candidate_indices,
)


def test_select_candidate_indices_unions_sources_without_duplicates():
    default = torch.tensor([[0.9, 0.8, 0.7, 0.1]])
    contrastive = torch.tensor([[0.1, 0.95, 0.6, 0.7]])

    indices, valid = select_candidate_indices(
        default,
        contrastive,
        topk_per_source=2,
        max_candidates=4,
    )

    assert indices.tolist() == [[0, 1, 3, 2]]
    assert valid.tolist() == [[True, True, True, True]]


def test_select_candidate_indices_pads_when_queries_are_exhausted():
    default = torch.tensor([[0.1, 0.9]])
    contrastive = torch.tensor([[0.8, 0.2]])

    indices, valid = select_candidate_indices(
        default,
        contrastive,
        topk_per_source=2,
        max_candidates=4,
    )

    assert indices.tolist() == [[1, 0, 0, 0]]
    assert valid.tolist() == [[True, True, False, False]]


@pytest.mark.parametrize(
    "default,contrastive,topk,max_candidates",
    [
        (torch.zeros(2, 3), torch.zeros(1, 3), 1, 2),
        (torch.zeros(2, 3, 1), torch.zeros(2, 3, 1), 1, 2),
        (torch.zeros(2, 3), torch.zeros(2, 3), 0, 2),
        (torch.zeros(2, 3), torch.zeros(2, 3), 1, 0),
    ],
)
def test_select_candidate_indices_rejects_invalid_inputs(
        default, contrastive, topk, max_candidates):
    with pytest.raises(ValueError):
        select_candidate_indices(
            default,
            contrastive,
            topk_per_source=topk,
            max_candidates=max_candidates,
        )


def test_compute_query_ious_matches_known_axis_aligned_boxes():
    candidates = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                                [1.0, 0.0, 0.0, 2.0, 2.0, 2.0]]])
    gt_boxes = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]])
    gt_mask = torch.tensor([[True]])

    ious = compute_query_ious(candidates, gt_boxes, gt_mask)

    assert torch.allclose(ious, torch.tensor([[1.0, 1.0 / 3.0]]))


def test_compute_query_ious_takes_best_valid_gt_and_ignores_masked_gt():
    candidates = torch.tensor([[[4.0, 0.0, 0.0, 2.0, 2.0, 2.0]]])
    gt_boxes = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                              [4.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                              [8.0, 0.0, 0.0, 2.0, 2.0, 2.0]]])

    included = compute_query_ious(
        candidates, gt_boxes, torch.tensor([[True, True, False]])
    )
    excluded = compute_query_ious(
        candidates, gt_boxes, torch.tensor([[True, False, True]])
    )

    assert included.item() == pytest.approx(1.0)
    assert excluded.item() == pytest.approx(0.0)


def test_compute_query_ious_returns_zero_without_valid_gt():
    candidates = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]])
    gt_boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]])

    ious = compute_query_ious(
        candidates, gt_boxes, torch.tensor([[False]])
    )

    assert torch.equal(ious, torch.zeros(1, 1))


@pytest.mark.parametrize(
    "candidates,gt_boxes,gt_mask",
    [
        (torch.zeros(2, 6), torch.zeros(1, 1, 6), torch.ones(1, 1)),
        (torch.zeros(1, 2, 5), torch.zeros(1, 1, 6), torch.ones(1, 1)),
        (torch.zeros(1, 2, 6), torch.zeros(2, 1, 6), torch.ones(2, 1)),
        (torch.zeros(1, 2, 6), torch.zeros(1, 1, 6), torch.ones(1, 2)),
    ],
)
def test_compute_query_ious_rejects_invalid_shapes(
        candidates, gt_boxes, gt_mask):
    with pytest.raises(ValueError):
        compute_query_ious(candidates, gt_boxes, gt_mask)


def test_candidate_oracle_uses_strict_thresholds():
    ious = torch.tensor([[0.25], [0.2501], [0.50], [0.5001]])
    valid = torch.ones_like(ious, dtype=torch.bool)

    metrics = compute_candidate_oracle(ious, valid)

    assert metrics["acc025"] == pytest.approx(0.75)
    assert metrics["acc050"] == pytest.approx(0.25)


def test_candidate_oracle_ignores_invalid_high_iou_candidates():
    ious = torch.tensor([[0.9, 0.1], [0.8, 0.6]])
    valid = torch.tensor([[False, True], [False, False]])

    metrics = compute_candidate_oracle(ious, valid)

    assert metrics == {"acc025": 0.0, "acc050": 0.0}


def test_candidate_oracle_rejects_empty_or_mismatched_batches():
    with pytest.raises(ValueError):
        compute_candidate_oracle(torch.empty(0, 2), torch.empty(0, 2).bool())
    with pytest.raises(ValueError):
        compute_candidate_oracle(torch.zeros(1, 2), torch.ones(1, 3).bool())


def test_query_reranker_outputs_masked_multitask_predictions():
    torch.manual_seed(3)
    features = torch.randn(2, 4, 6)
    valid = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
    ])
    model = QueryReranker(input_dim=6, hidden_dim=16, dropout=0.0)

    output = model(features, valid)

    assert output["ranking_logits"].shape == (2, 4)
    assert output["threshold_logits"].shape == (2, 4, 2)
    assert output["iou_estimate"].shape == (2, 4)
    assert torch.all(output["ranking_logits"][~valid] == -1e4)
    assert torch.all((output["iou_estimate"] >= 0.0)
                     & (output["iou_estimate"] <= 1.0))
    assert torch.all(output["iou_estimate"][~valid] == 0.0)


def test_listwise_targets_use_threshold_priority_and_ignore_padding():
    ious = torch.tensor([
        [0.49, 0.51, 0.90],
        [0.24, 0.30, 0.99],
    ])
    valid = torch.tensor([
        [True, True, False],
        [True, True, False],
    ])

    targets = select_listwise_targets(ious, valid)

    assert targets.tolist() == [1, 1]


def test_rec_reranker_loss_has_gradients_and_decreases_on_toy_batch():
    torch.manual_seed(7)
    features = torch.tensor([[
        [0.0, 0.0],
        [1.0, 1.0],
        [-1.0, -1.0],
    ]])
    ious = torch.tensor([[0.1, 0.8, 0.2]])
    valid = torch.tensor([[True, True, False]])
    model = QueryReranker(input_dim=2, hidden_dim=8, dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    initial, _ = compute_rec_reranker_loss(model(features, valid), ious, valid)
    initial_value = initial.item()
    for _ in range(20):
        optimizer.zero_grad()
        loss, stats = compute_rec_reranker_loss(
            model(features, valid), ious, valid
        )
        loss.backward()
        optimizer.step()

    final, _ = compute_rec_reranker_loss(model(features, valid), ious, valid)

    assert final.item() < initial_value
    assert set(stats) == {
        "loss_listwise", "loss_best_tier_pairwise", "loss_ranking",
        "loss_threshold", "loss_iou", "loss_total",
        "tier_pairwise_informative_rows", "tier_pairwise_pair_count",
        "tier_pairwise_positive_count", "tier_pairwise_negative_count",
    }


def _ranking_only_loss(logits, candidate_ious, valid, alpha):
    outputs = {
        "ranking_logits": logits,
        "threshold_logits": torch.zeros(
            candidate_ious.shape + (2,), dtype=logits.dtype
        ),
        "iou_estimate": torch.zeros_like(candidate_ious, dtype=logits.dtype),
    }
    return compute_rec_reranker_loss(
        outputs,
        candidate_ious,
        valid,
        listwise_weight=1.0,
        threshold_weight=0.0,
        iou_weight=0.0,
        tier_pairwise_alpha=alpha,
    )


def test_tier_pairwise_alpha_zero_preserves_listwise_endpoint():
    candidate_ious = torch.tensor([[0.10, 0.30, 0.80]])
    valid = torch.ones_like(candidate_ious, dtype=torch.bool)
    outputs = {
        "ranking_logits": torch.tensor([[0.0, 1.0, 2.0]]),
        "threshold_logits": torch.zeros(1, 3, 2),
        "iou_estimate": torch.zeros(1, 3),
    }

    default_loss, default_stats = compute_rec_reranker_loss(
        outputs, candidate_ious, valid
    )
    endpoint_loss, endpoint_stats = compute_rec_reranker_loss(
        outputs,
        candidate_ious,
        valid,
        tier_pairwise_alpha=0.0,
    )

    assert torch.equal(endpoint_loss, default_loss)
    assert torch.equal(
        endpoint_stats["loss_ranking"], endpoint_stats["loss_listwise"]
    )
    for name in (
            "loss_listwise", "loss_threshold", "loss_iou", "loss_total"):
        assert torch.equal(endpoint_stats[name], default_stats[name])


def test_tier_pairwise_alpha_one_uses_every_best_tier_candidate():
    logits = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    candidate_ious = torch.tensor([[0.10, 0.30, 0.80]])
    valid = torch.ones_like(candidate_ious, dtype=torch.bool)

    loss, stats = _ranking_only_loss(
        logits, candidate_ious, valid, alpha=1.0
    )
    loss.backward()

    assert loss.item() == pytest.approx(
        0.22009484469890594, rel=1e-6, abs=1e-7
    )
    assert stats["loss_ranking"].item() == pytest.approx(loss.item())
    assert stats["loss_best_tier_pairwise"].item() == pytest.approx(
        loss.item()
    )
    assert logits.grad.tolist()[0] == pytest.approx(
        [0.05960145965218544, 0.13447071611881256,
         -0.1940721720457077],
        rel=1e-6,
        abs=1e-7,
    )


@pytest.mark.parametrize(
    "candidate_ious,expected_gradient",
    [
        ([0.25, 0.2501, 0.50], [0.5, -0.25, -0.25]),
        ([0.50, 0.5001], [0.5, -0.5]),
    ],
)
def test_tier_pairwise_uses_strict_thresholds_and_multi_positive_tiers(
        candidate_ious, expected_gradient):
    ious = torch.tensor([candidate_ious])
    logits = torch.zeros_like(ious, requires_grad=True)
    valid = torch.ones_like(ious, dtype=torch.bool)

    loss, _stats = _ranking_only_loss(logits, ious, valid, alpha=1.0)
    loss.backward()

    assert loss.item() == pytest.approx(
        0.6931471824645996, rel=1e-6, abs=1e-7
    )
    assert logits.grad.tolist()[0] == pytest.approx(
        expected_gradient, rel=1e-6, abs=1e-7
    )


def test_tier_pairwise_masks_invalid_high_iou_and_high_logit_candidate():
    logits = torch.tensor([[0.0, 0.0, 100.0]], requires_grad=True)
    candidate_ious = torch.tensor([[0.80, 0.10, 0.99]])
    valid = torch.tensor([[True, True, False]])

    loss, _stats = _ranking_only_loss(
        logits, candidate_ious, valid, alpha=1.0
    )
    loss.backward()

    assert loss.item() == pytest.approx(
        0.6931471824645996, rel=1e-6, abs=1e-7
    )
    assert logits.grad.tolist()[0] == pytest.approx(
        [-0.5, 0.5, 0.0], rel=1e-6, abs=1e-7
    )


def test_tier_pairwise_reduces_per_row_and_ignores_no_pair_rows():
    logits = torch.tensor([
        [0.0, 0.0, 25.0],
        [2.0, 0.0, 0.0],
        [3.0, -2.0, 50.0],
    ], requires_grad=True)
    candidate_ious = torch.tensor([
        [0.80, 0.10, 0.00],
        [0.80, 0.10, 0.20],
        [0.80, 0.90, 0.00],
    ])
    valid = torch.tensor([
        [True, True, False],
        [True, True, True],
        [True, True, False],
    ])

    loss, _stats = _ranking_only_loss(
        logits, candidate_ious, valid, alpha=1.0
    )
    loss.backward()

    assert loss.item() == pytest.approx(
        0.4100375771522522, rel=1e-6, abs=1e-7
    )
    expected_gradient = torch.tensor([
        [-0.25, 0.25, 0.0],
        [-0.05960145965218544, 0.02980072982609272,
         0.02980072982609272],
        [0.0, 0.0, 0.0],
    ])
    assert torch.allclose(
        logits.grad, expected_gradient, rtol=1e-6, atol=1e-7
    )


def test_tier_pairwise_all_no_pair_rows_return_differentiable_zero():
    logits = torch.tensor([[3.0, -2.0]], requires_grad=True)
    candidate_ious = torch.tensor([[0.80, 0.90]])
    valid = torch.ones_like(candidate_ious, dtype=torch.bool)

    loss, stats = _ranking_only_loss(
        logits, candidate_ious, valid, alpha=1.0
    )
    loss.backward()

    assert loss.item() == 0.0
    assert stats["loss_best_tier_pairwise"].item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_tier_pairwise_stats_report_detached_batch_coverage_counts():
    logits = torch.zeros(3, 5, requires_grad=True)
    candidate_ious = torch.tensor([
        [0.80, 0.70, 0.30, 0.10, 0.99],
        [0.30, 0.40, 0.50, 0.99, 0.99],
        [0.5001, 0.50, 0.25, 0.99, 0.99],
    ])
    valid = torch.tensor([
        [True, True, True, True, False],
        [True, True, True, False, False],
        [True, True, True, False, False],
    ])

    loss, stats = _ranking_only_loss(
        logits, candidate_ious, valid, alpha=1.0
    )

    expected = {
        "tier_pairwise_informative_rows": 2,
        "tier_pairwise_pair_count": 6,
        "tier_pairwise_positive_count": 6,
        "tier_pairwise_negative_count": 4,
    }
    for name, count in expected.items():
        value = stats[name]
        assert isinstance(value, torch.Tensor)
        assert value.shape == torch.Size([])
        assert value.dtype == torch.long
        assert value.requires_grad is False
        assert value.grad_fn is None
        assert value.item() == count

    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "alpha",
    [True, False, float("nan"), float("inf"), float("-inf"), -0.1, 1.1],
)
def test_tier_pairwise_alpha_rejects_bool_nonfinite_and_out_of_range(alpha):
    logits = torch.zeros(1, 2)
    candidate_ious = torch.tensor([[0.10, 0.80]])
    valid = torch.ones_like(candidate_ious, dtype=torch.bool)

    with pytest.raises(ValueError, match="tier_pairwise_alpha"):
        _ranking_only_loss(logits, candidate_ious, valid, alpha=alpha)


def test_rec_reranker_loss_rejects_rows_without_valid_candidates():
    model = QueryReranker(input_dim=2, hidden_dim=8, dropout=0.0)
    features = torch.zeros(1, 2, 2)
    valid = torch.zeros(1, 2, dtype=torch.bool)
    outputs = model(features, valid)

    with pytest.raises(ValueError):
        compute_rec_reranker_loss(outputs, torch.zeros(1, 2), valid)


def test_rec_reranker_loss_masks_invalid_logits_supplied_by_caller():
    """The public loss must not rely on the model to pre-mask padding."""
    candidate_ious = torch.tensor([[0.1, 0.8, 0.0]])
    valid = torch.tensor([[True, True, False]])
    outputs = {
        "ranking_logits": torch.tensor([[0.0, 1.0, 100.0]]),
        "threshold_logits": torch.zeros(1, 3, 2),
        "iou_estimate": torch.zeros(1, 3),
    }

    loss, stats = compute_rec_reranker_loss(
        outputs,
        candidate_ious,
        valid,
        listwise_weight=1.0,
        threshold_weight=0.0,
        iou_weight=0.0,
    )

    expected = torch.nn.functional.cross_entropy(
        torch.tensor([[0.0, 1.0, -1e4]]), torch.tensor([1])
    )
    assert loss.item() == pytest.approx(expected.item())
    assert stats["loss_listwise"].item() == pytest.approx(expected.item())


def test_blend_candidate_scores_preserves_default_and_learned_endpoints():
    default_scores = torch.tensor([[3.0, 2.0, 100.0]])
    ranking_logits = torch.tensor([[0.0, 5.0, 999.0]])
    valid = torch.tensor([[True, True, False]])

    default_only = blend_candidate_scores(
        default_scores, ranking_logits, valid, reranker_weight=0.0
    )
    learned_only = blend_candidate_scores(
        default_scores, ranking_logits, valid, reranker_weight=1.0
    )

    assert default_only.argmax(dim=1).item() == 0
    assert learned_only.argmax(dim=1).item() == 1
    assert default_only[0, 2].item() == -1e4
    assert learned_only[0, 2].item() == -1e4


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan")])
def test_blend_candidate_scores_rejects_invalid_weights(weight):
    scores = torch.zeros(1, 2)
    valid = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="weight"):
        blend_candidate_scores(scores, scores, valid, weight)
