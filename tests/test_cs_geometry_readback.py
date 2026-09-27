import torch

from models.cs_mcln_modules import GeometryEvidenceReadback, MaskSupportBoxRefiner
from models.cs_native_root_quality import root_matched_quality_loss
from models.modules import ClsAgnosticPredictHead
from models.source_choice_adapter import compute_default_source_scores


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
    evidence = evidence.detach()

    readback = GeometryEvidenceReadback(channels, context_dim=8)
    updated = readback(query, evidence)
    assert torch.equal(updated, query)
    updated.square().sum().backward()
    assert readback.output.weight.grad.abs().sum().item() > 0

    optimizer = torch.optim.AdamW(readback.parameters(), lr=1e-2)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    readback(query, evidence).square().sum().backward()
    assert readback.encode[0].weight.grad.abs().sum().item() > 0
    for encoder in (readback.coarse_encoder, readback.support_encoder,
                    readback.refined_encoder):
        assert encoder.weight.grad.abs().sum().item() > 0
    assert readback.set_attention.in_proj_weight.grad.abs().sum().item() > 0

    with torch.no_grad():
        readback.output.weight.normal_(std=0.1)
        changed_evidence = evidence.clone()
        changed_evidence[:, 1] += 2
        original = readback(query, evidence)
        changed = readback(query, changed_evidence)
    assert not torch.allclose(original[:, 0], changed[:, 0])
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        reordered = readback(query[:, permutation], evidence[:, permutation])
    torch.testing.assert_close(reordered, original[:, permutation])


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


def test_native_quality_gradient_reaches_readback_then_refiner():
    torch.manual_seed(2027)
    query = torch.randn(1, 3, 8)
    mask_query = torch.randn_like(query)
    super_features = [torch.randn(8, 5)]
    super_xyz = [torch.randn(1, 5, 3)]
    centers = torch.tensor([[
        [0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [0.0, 0.0, 0.0],
    ]])
    sizes = torch.full_like(centers, 2.0)
    gt = torch.cat((centers[:, :1], sizes[:, :1]), dim=-1)
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    positive = torch.tensor([[[1.0, 0.0, 0.0]]])
    zeros = torch.zeros_like(positive)
    refiner = MaskSupportBoxRefiner(8)
    readback = GeometryEvidenceReadback(8, context_dim=8)
    head = ClsAgnosticPredictHead(
        num_class=3, num_heading_bin=1, num_proposal=3,
        seed_feat_dim=8, objectness=False, heading=False,
    ).eval()

    def forward_loss():
        center, size, evidence = refiner(
            query, mask_query, super_features, super_xyz,
            centers, sizes, return_evidence=True,
        )
        output = {
            'positive_map': positive,
            'modify_positive_map': zeros,
            'pron_positive_map': zeros,
            'rel_positive_map': zeros,
            'other_entity_map': zeros,
        }
        updated = readback(query, evidence)
        head.write_semantic_scores(
            updated.transpose(1, 2), output, 'last_',
        )
        scores = compute_default_source_scores(output, output)
        boxes = torch.cat((center, size), dim=-1)
        return root_matched_quality_loss(
            scores, boxes, gt, matches, ['scanrefer'], 0.1,
        )

    optimizer = torch.optim.AdamW(readback.parameters(), lr=1e-2)
    forward_loss().backward()
    assert readback.output.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    refiner.zero_grad(set_to_none=True)
    forward_loss().backward()
    assert refiner.delta[-1].weight.grad.abs().sum() > 0
