import torch

from models.source_choice_adapter import (
    build_mcln_source_choice_batch,
    compute_default_source_scores,
)


def test_default_source_binarizes_main_positive_map_like_evaluator():
    sem_logits = torch.log(torch.tensor([[
        [0.50, 0.30, 0.20],
        [0.45, 0.10, 0.45],
    ]]))
    end_points = {"last_sem_cls_scores": sem_logits}
    inputs = {
        "positive_map": torch.tensor([[[0.75, 0.25, 0.0]]]),
        "modify_positive_map": torch.zeros(1, 1, 3),
        "pron_positive_map": torch.zeros(1, 1, 3),
        "other_entity_map": torch.zeros(1, 1, 3),
        "rel_positive_map": torch.zeros(1, 1, 3),
    }

    scores = compute_default_source_scores(end_points, inputs)

    assert torch.allclose(scores, torch.tensor([[0.80, 0.55]]), atol=1e-6)
    assert scores.argmax(dim=1).item() == 0


def test_mcln_adapter_outputs_two_source_scores_with_candidate_shapes():
    batch_size, num_queries, num_tokens, feat_dim, num_superpoints = 2, 3, 5, 8, 4
    end_points = {
        "last_center": torch.zeros(batch_size, num_queries, 3),
        "last_pred_size": torch.ones(batch_size, num_queries, 3),
        "last_sem_cls_scores": torch.randn(batch_size, num_queries, num_tokens),
        "source_choice_candidate_feats": torch.randn(batch_size, num_queries, feat_dim),
        "last_pred_masks": [
            torch.randn(1, num_queries, num_superpoints)
            for _ in range(batch_size)
        ],
        "sp_last_pred_masks": [
            torch.randn(num_queries, num_superpoints)
            for _ in range(batch_size)
        ],
        "adaptive_weights": [torch.tensor(0.6), torch.tensor(0.4)],
        "text_feats": torch.randn(batch_size, num_tokens, feat_dim),
        "text_attention_mask": torch.zeros(batch_size, num_tokens, dtype=torch.bool),
    }
    gate_candidate_feats = torch.randn(batch_size, num_queries, 288)
    end_points["source_moe_gate_candidate_feats"] = gate_candidate_feats
    inputs = {
        "positive_map": torch.zeros(batch_size, 1, num_tokens),
        "modify_positive_map": torch.zeros(batch_size, 1, num_tokens),
        "pron_positive_map": torch.zeros(batch_size, 1, num_tokens),
        "other_entity_map": torch.zeros(batch_size, 1, num_tokens),
        "rel_positive_map": torch.zeros(batch_size, 1, num_tokens),
    }
    inputs["positive_map"][:, :, 1] = 1.0

    batch = build_mcln_source_choice_batch(
        end_points,
        inputs,
        source_names=["default", "mask_text"],
    )

    assert batch["candidate_boxes"].shape == (batch_size, num_queries, 6)
    assert batch["candidate_feats"].shape == (batch_size, num_queries, feat_dim)
    assert batch["gate_candidate_feats"] is gate_candidate_feats
    assert batch["valid_mask"].shape == (batch_size, num_queries)
    assert batch["source_validity"].shape == (
        batch_size, num_queries, 2
    )
    assert bool(batch["source_validity"].all().item())
    assert sorted(batch["source_scores"]) == ["default", "mask_text"]
    for scores in batch["source_scores"].values():
        assert scores.shape == (batch_size, num_queries)
    assert not torch.equal(
        batch["source_scores"]["default"],
        batch["source_scores"]["mask_text"],
    )


def test_mcln_adapter_marks_decoder_gate_evidence_as_optional():
    end_points = {
        "last_center": torch.zeros(1, 2, 3),
        "last_pred_size": torch.ones(1, 2, 3),
        "last_sem_cls_scores": torch.randn(1, 2, 3),
        "last_proj_queries": torch.randn(1, 2, 4),
        "text_feats": torch.randn(1, 3, 4),
        "text_attention_mask": torch.zeros(1, 3, dtype=torch.bool),
    }
    inputs = {
        "positive_map": torch.tensor([[[1.0, 0.0, 0.0]]]),
        "modify_positive_map": torch.zeros(1, 1, 3),
        "pron_positive_map": torch.zeros(1, 1, 3),
        "other_entity_map": torch.zeros(1, 1, 3),
        "rel_positive_map": torch.zeros(1, 1, 3),
    }

    batch = build_mcln_source_choice_batch(
        end_points, inputs, source_names=["default"]
    )

    assert batch["gate_candidate_feats"] is None


def test_mcln_adapter_outputs_contrastive_text_source_scores():
    batch_size, num_queries, num_tokens, feat_dim, proj_dim = 1, 2, 4, 8, 3
    proj_queries = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    proj_tokens = torch.tensor(
        [[[1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
          [1.0, 1.0, 0.0]]]
    )
    end_points = {
        "last_center": torch.zeros(batch_size, num_queries, 3),
        "last_pred_size": torch.ones(batch_size, num_queries, 3),
        "last_sem_cls_scores": torch.randn(batch_size, num_queries, num_tokens),
        "source_choice_candidate_feats": torch.randn(batch_size, num_queries, feat_dim),
        "last_proj_queries": proj_queries,
        "proj_tokens": proj_tokens,
        "text_feats": torch.randn(batch_size, num_tokens, feat_dim),
        "text_attention_mask": torch.zeros(batch_size, num_tokens, dtype=torch.bool),
    }
    inputs = {
        "positive_map": torch.zeros(batch_size, 1, num_tokens),
        "modify_positive_map": torch.zeros(batch_size, 1, num_tokens),
        "pron_positive_map": torch.zeros(batch_size, 1, num_tokens),
        "other_entity_map": torch.zeros(batch_size, 1, num_tokens),
        "rel_positive_map": torch.zeros(batch_size, 1, num_tokens),
    }
    inputs["positive_map"][:, :, 0] = 1.0

    batch = build_mcln_source_choice_batch(
        end_points,
        inputs,
        source_names=["default", "contrastive_text"],
    )

    assert sorted(batch["source_scores"]) == ["contrastive_text", "default"]
    expected = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(
        batch["source_scores"]["contrastive_text"],
        expected,
        atol=1e-6,
    )


def test_mcln_adapter_marks_missing_optional_source_invalid_not_valid_zero():
    end_points = {
        "last_center": torch.zeros(1, 2, 3),
        "last_pred_size": torch.ones(1, 2, 3),
        "last_sem_cls_scores": torch.randn(1, 2, 3),
        "source_choice_candidate_feats": torch.randn(1, 2, 4),
    }
    inputs = {
        "positive_map": torch.tensor([[[1.0, 0.0, 0.0]]]),
        "modify_positive_map": torch.zeros(1, 1, 3),
        "pron_positive_map": torch.zeros(1, 1, 3),
        "other_entity_map": torch.zeros(1, 1, 3),
        "rel_positive_map": torch.zeros(1, 1, 3),
    }

    batch = build_mcln_source_choice_batch(
        end_points, inputs,
        source_names=["default", "contrastive_text", "mask_text"],
    )

    assert bool(batch["source_validity"][..., 0].all().item())
    assert not bool(batch["source_validity"][..., 1:].any().item())
    assert torch.equal(
        batch["source_scores"]["contrastive_text"], torch.zeros(1, 2)
    )
    assert torch.equal(
        batch["source_scores"]["mask_text"], torch.zeros(1, 2)
    )


def test_mcln_adapter_outputs_default_rank_blend_contrastive_source_scores():
    batch_size, num_queries, num_tokens, feat_dim, proj_dim = 1, 3, 2, 8, 2
    end_points = {
        "last_center": torch.zeros(batch_size, num_queries, 3),
        "last_pred_size": torch.ones(batch_size, num_queries, 3),
        "last_sem_cls_scores": torch.tensor([[[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]),
        "source_choice_candidate_feats": torch.randn(batch_size, num_queries, feat_dim),
        "last_proj_queries": torch.tensor(
            [[[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]]]
        ),
        "proj_tokens": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        "text_feats": torch.randn(batch_size, num_tokens, feat_dim),
        "text_attention_mask": torch.zeros(batch_size, num_tokens, dtype=torch.bool),
    }
    inputs = {
        "positive_map": torch.zeros(batch_size, 1, num_tokens),
        "modify_positive_map": torch.zeros(batch_size, 1, num_tokens),
        "pron_positive_map": torch.zeros(batch_size, 1, num_tokens),
        "other_entity_map": torch.zeros(batch_size, 1, num_tokens),
        "rel_positive_map": torch.zeros(batch_size, 1, num_tokens),
    }
    inputs["positive_map"][:, :, 0] = 1.0

    batch = build_mcln_source_choice_batch(
        end_points,
        inputs,
        source_names=["default", "default_rank_blend_contrastive010"],
    )

    blend = batch["source_scores"]["default_rank_blend_contrastive010"]
    default = batch["source_scores"]["default"]
    assert blend.shape == default.shape
    assert blend.argmax(dim=1).item() == default.argmax(dim=1).item()
    assert not torch.allclose(blend, default)


def test_mcln_adapter_outputs_152_dimensional_deployable_rich_evidence():
    batch_size, num_queries, num_tokens = 2, 3, 5
    proj_dim, num_superpoints = 64, 4
    end_points = {
        "last_center": torch.rand(batch_size, num_queries, 3),
        "last_pred_size": torch.rand(batch_size, num_queries, 3) + 0.1,
        "last_sem_cls_scores": torch.randn(
            batch_size, num_queries, num_tokens
        ),
        "last_proj_queries": torch.randn(
            batch_size, num_queries, proj_dim
        ),
        "proj_tokens": torch.randn(batch_size, num_tokens, proj_dim),
        "last_pred_masks": [
            torch.randn(1, num_queries, num_superpoints)
            for _ in range(batch_size)
        ],
        "sp_last_pred_masks": [
            torch.randn(num_queries, num_superpoints)
            for _ in range(batch_size)
        ],
        "adaptive_weights": [torch.tensor(0.6), torch.tensor(0.4)],
        "text_feats": torch.randn(batch_size, num_tokens, 288),
        "text_attention_mask": torch.zeros(
            batch_size, num_tokens, dtype=torch.bool
        ),
    }
    inputs = {
        "point_clouds": torch.rand(batch_size, 16, 6),
        "positive_map": torch.zeros(batch_size, 1, num_tokens),
        "modify_positive_map": torch.zeros(batch_size, 1, num_tokens),
        "pron_positive_map": torch.zeros(batch_size, 1, num_tokens),
        "other_entity_map": torch.zeros(batch_size, 1, num_tokens),
        "rel_positive_map": torch.zeros(batch_size, 1, num_tokens),
    }
    inputs["positive_map"][:, :, 1] = 1.0

    ordinary = build_mcln_source_choice_batch(
        end_points, inputs, source_names=["default"]
    )
    rich = build_mcln_source_choice_batch(
        end_points,
        inputs,
        source_names=["default"],
        include_rich_candidate_feats=True,
    )

    assert ordinary["rich_candidate_feats"] is None
    assert rich["rich_candidate_feats"].shape == (
        batch_size, num_queries, 152
    )
    assert torch.isfinite(rich["rich_candidate_feats"]).all()
