import torch

from models.cs_mcln_modules import GeometryEvidenceReadback, MaskSupportBoxRefiner
from models.modules import ClsAgnosticPredictHead


def test_refiner_evidence_preserves_boxes_and_readback_starts_at_identity():
    torch.manual_seed(2027)
    batch, queries, channels = 2, 3, 8
    query = torch.randn(batch, queries, channels)
    mask_query = torch.randn_like(query)
    super_features = [torch.randn(channels, 5) for _ in range(batch)]
    super_xyz = [torch.randn(1, 5, 3) for _ in range(batch)]
    centers = torch.randn(batch, queries, 3)
    sizes = torch.rand(batch, queries, 3) + 0.5

    refiner = MaskSupportBoxRefiner(channels).eval()
    plain_center, plain_size = refiner(
        query, mask_query, super_features, super_xyz, centers, sizes,
    )
    center, size, evidence = refiner(
        query, mask_query, super_features, super_xyz, centers, sizes,
        return_evidence=True,
    )
    assert torch.equal(center, plain_center)
    assert torch.equal(size, plain_size)
    assert evidence.shape == (batch, queries, channels + 15)

    readback = GeometryEvidenceReadback(channels, context_dim=8)
    updated = readback(query, evidence)
    assert torch.equal(updated, query)
    updated.square().sum().backward()
    assert readback.output.weight.grad.abs().sum().item() > 0

    with torch.no_grad():
        readback.output.weight.normal_(std=0.1)
        changed_evidence = evidence.clone()
        changed_evidence[:, 1] += 2
        original = readback(query, evidence)
        changed = readback(query, changed_evidence)
    assert not torch.allclose(original[:, 0], changed[:, 0])


def test_deferred_semantic_head_runs_once_and_preserves_initial_logits():
    torch.manual_seed(2027)
    head = ClsAgnosticPredictHead(
        num_class=6, num_heading_bin=1, num_proposal=3,
        seed_feat_dim=8, objectness=False, heading=False,
    ).eval()
    features = torch.randn(2, 8, 3)
    base_xyz = torch.randn(2, 3, 3)
    standard = {}
    head(features, base_xyz, standard, prefix='last_')

    calls = []
    hook = head.sem_cls_scores_head.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(True)
    )
    delayed = {}
    head(features, base_xyz, delayed, prefix='last_', defer_semantic=True)
    assert 'last_sem_cls_scores' not in delayed
    head.write_semantic_scores(features, delayed, prefix='last_')
    hook.remove()

    assert len(calls) == 1
    assert torch.equal(standard['last_sem_cls_scores'], delayed['last_sem_cls_scores'])
    assert torch.equal(standard['last_center'], delayed['last_center'])
    assert torch.equal(standard['last_pred_size'], delayed['last_pred_size'])
