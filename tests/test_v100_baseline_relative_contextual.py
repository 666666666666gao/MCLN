import torch

from models.rec_baseline_relative_contextual import (
    BaselineRelativeContextualReranker,
    apply_baseline_relative_policy,
)
from scripts.run_v100_baseline_relative_contextual import (
    build_relative_targets,
    relative_effect_loss,
    stratified_epoch_batches,
)


def _batch(batch_size=3):
    torch.manual_seed(4)
    query_valid = torch.ones(batch_size, 16, dtype=torch.bool)
    variant_valid = torch.ones(batch_size, 16, 7, dtype=torch.bool)
    query_aux_binary = torch.zeros(batch_size, 16, 2, dtype=torch.bool)
    variant_aux_binary = torch.zeros(batch_size, 16, 7, 2, dtype=torch.bool)
    for row in range(batch_size):
        query_aux_binary[row, 0, 0] = True
        query_aux_binary[row, 0, 1] = True
        variant_aux_binary[row, 0, row % 7, 1] = True
    return {
        "query_features": torch.randn(batch_size, 16, 152),
        "variant_features": torch.randn(batch_size, 16, 7, 25),
        "query_aux_continuous": torch.randn(batch_size, 16, 4),
        "query_aux_binary": query_aux_binary,
        "variant_aux_continuous": torch.randn(batch_size, 16, 7, 2),
        "variant_aux_binary": variant_aux_binary,
        "query_valid": query_valid,
        "variant_valid": variant_valid,
    }


def test_relative_targets_encode_break_neutral_fix_and_baseline_anchor():
    ious = torch.tensor([[[0.10, 0.30, 0.60, 0.20, 0.20, 0.20, 0.20]] * 16])
    valid = torch.ones(1, 16, 7, dtype=torch.bool)
    target = build_relative_targets(ious, torch.tensor([1]), valid)
    assert target["classes"][0, 0, 0].tolist() == [0, 1]
    assert target["classes"][0, 0, 1].tolist() == [1, 1]
    assert target["classes"][0, 0, 2].tolist() == [1, 2]


def test_policy_keeps_zero_anchor_or_selects_doubly_positive_candidate():
    logits = torch.zeros(1, 16, 7, 2, 3)
    valid = torch.ones(1, 16, 7, dtype=torch.bool)
    baseline = torch.tensor([0])
    kept = apply_baseline_relative_policy(logits, valid, baseline)
    assert kept["selected_indices"].tolist() == [0]
    logits[0, 0, 1, :, 2] = 5.0
    switched = apply_baseline_relative_policy(logits, valid, baseline)
    assert switched["selected_indices"].tolist() == [1]
    logits[0, 0, 2, 0, 2] = 9.0
    logits[0, 0, 2, 1, 0] = 9.0
    vetoed = apply_baseline_relative_policy(logits, valid, baseline)
    assert vetoed["selected_indices"].tolist() == [1]


def test_model_padding_isolation_and_loss_has_finite_gradients():
    batch = _batch(3)
    batch["query_valid"][:, -1] = False
    batch["variant_valid"][:, -1] = False
    model = BaselineRelativeContextualReranker()
    model.eval()
    first = model(**batch)["relative_logits"].detach()
    batch["query_features"][:, -1] = 1e6
    batch["variant_features"][:, -1] = -1e6
    second = model(**batch)["relative_logits"].detach()
    assert torch.equal(first[:, :-1], second[:, :-1])
    model.train()
    outputs = model(**batch)
    ious = torch.rand(3, 16, 7)
    ious[:, -1] = 0.0
    for row, value in enumerate((0.1, 0.4, 0.8)):
        baseline = int(outputs["baseline_indices"][row].item())
        ious[row].reshape(-1)[baseline] = value
    loss, _ = relative_effect_loss(outputs, ious, batch["variant_valid"])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_query_permutation_equivariance_preserves_flat_candidate_semantics():
    batch = _batch(2)
    model = BaselineRelativeContextualReranker().eval()
    first = model(**batch)["relative_logits"].detach()
    permutation = torch.tensor([3, 0, 2, 1] + list(range(4, 16)))
    permuted = dict(batch)
    for name in ("query_features", "query_aux_continuous", "query_aux_binary",
                 "query_valid", "variant_features", "variant_aux_continuous",
                 "variant_aux_binary", "variant_valid"):
        permuted[name] = batch[name][:, permutation]
    second = model(**permuted)["relative_logits"].detach()
    torch.testing.assert_close(
        first[:, permutation], second, rtol=1e-6, atol=1e-7
    )


def test_stratified_batches_partition_rows_and_cover_all_bins():
    records = []
    for index, value in enumerate([0.1] * 20 + [0.4] * 20 + [0.8] * 260):
        ious = torch.zeros(16, 7)
        ious.reshape(-1)[0] = value
        records.append({"candidate_ious": ious, "baseline_index": 0})
    import random
    batches = stratified_epoch_batches(records, random.Random(0))
    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(len(records)))
    for batch in batches:
        values = [float(records[index]["candidate_ious"].reshape(-1)[0]) for index in batch]
        assert any(value <= 0.25 for value in values)
        assert any(0.25 < value <= 0.50 for value in values)
        assert any(value > 0.50 for value in values)
