import sys
import hashlib
from types import SimpleNamespace

import pytest
import torch

from main_utils import parse_option
from models.rec_candidate_adapter import (
    FEATURE_SCHEMA_VERSION,
    build_rec_candidate_batch,
    scatter_candidate_scores,
)
from models.rec_reranker import blend_candidate_scores
from train_dist_mod import (
    TrainTester,
    build_rec_reranker_outputs,
    build_rec_reranker_scores,
    validate_rec_reranker_provenance,
)


class PositionReranker(torch.nn.Module):
    def forward(self, features, valid_mask):
        logits = torch.arange(
            features.shape[1], device=features.device, dtype=features.dtype
        ).unsqueeze(0).expand(features.shape[0], -1)
        return {"ranking_logits": logits.masked_fill(~valid_mask, -1e4)}


def _runtime_batch():
    torch.manual_seed(23)
    num_queries = 4
    num_tokens = 5
    end_points = {
        "last_center": torch.tensor([[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]]),
        "last_pred_size": torch.ones(1, num_queries, 3),
        "last_sem_cls_scores": torch.randn(1, num_queries, num_tokens),
        "last_proj_queries": torch.randn(1, num_queries, 3),
        "proj_tokens": torch.randn(1, num_tokens, 3),
        "last_pred_masks": [torch.randn(1, num_queries, 3)],
        "sp_last_pred_masks": [torch.randn(num_queries, 3)],
        "adaptive_weights": [torch.tensor(0.5)],
    }
    inputs = {
        "point_clouds": torch.tensor([[
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 2.0, 1.0, 0.0],
        ]]),
        "positive_map": torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]]]),
        "modify_positive_map": torch.zeros(1, 1, num_tokens),
        "pron_positive_map": torch.zeros(1, 1, num_tokens),
        "other_entity_map": torch.zeros(1, 1, num_tokens),
        "rel_positive_map": torch.zeros(1, 1, num_tokens),
    }
    return end_points, inputs


def _artifact_for(candidate_batch):
    feature_dim = candidate_batch["features"].shape[-1]
    return {
        "adapter_schema_version": FEATURE_SCHEMA_VERSION,
        "input_dim": feature_dim,
        "feature_names": list(candidate_batch["feature_names"]),
        "candidate_rule": {
            "topk_per_source": 2,
            "max_candidates": 3,
        },
        "feature_mean": torch.zeros(feature_dim),
        "feature_std": torch.ones(feature_dim),
        "score_mode": "rank_blend",
        "reranker_weight": 1.0,
    }


def test_build_rec_reranker_scores_normalizes_and_scatters_candidates():
    end_points, inputs = _runtime_batch()
    candidate_batch = build_rec_candidate_batch(
        end_points, inputs, topk_per_source=2, max_candidates=3
    )
    artifact = _artifact_for(candidate_batch)

    scores = build_rec_reranker_scores(
        end_points, inputs, PositionReranker(), artifact
    )

    ranking_logits = torch.arange(3, dtype=scores.dtype).unsqueeze(0)
    compact = blend_candidate_scores(
        candidate_batch["default_scores"],
        ranking_logits,
        candidate_batch["valid_mask"],
        reranker_weight=1.0,
    )
    expected = scatter_candidate_scores(
        compact,
        candidate_batch["query_indices"],
        candidate_batch["valid_mask"],
        num_queries=4,
    )
    assert torch.equal(scores, expected)
    assert torch.isneginf(scores).sum().item() == 1


@pytest.mark.parametrize("tied", (False, True))
def test_build_rec_reranker_outputs_preserves_wrapper_query_scores_bitwise(
        tied):
    end_points, inputs = _runtime_batch()
    candidate_batch = build_rec_candidate_batch(
        end_points, inputs, topk_per_source=2, max_candidates=3
    )
    artifact = _artifact_for(candidate_batch)
    if tied:
        artifact["reranker_weight"] = 0.5

    outputs = build_rec_reranker_outputs(
        end_points, inputs, PositionReranker(), artifact
    )
    wrapper_scores = build_rec_reranker_scores(
        end_points, inputs, PositionReranker(), artifact
    )

    assert set(outputs) == {
        "candidate_batch", "compact_scores", "query_scores",
    }
    assert outputs["candidate_batch"]["query_indices"].shape == (1, 3)
    assert outputs["compact_scores"].shape == (1, 3)
    assert torch.equal(wrapper_scores, outputs["query_scores"])


def test_build_rec_reranker_outputs_uses_canonical_query_tie_order(
        monkeypatch):
    end_points, inputs = _runtime_batch()
    candidate_batch = build_rec_candidate_batch(
        end_points, inputs, topk_per_source=2, max_candidates=3
    )
    candidate_batch = dict(candidate_batch)
    candidate_batch["query_indices"] = torch.tensor([[3, 2, 1]])
    candidate_batch["default_scores"] = torch.tensor([[1.0, 0.5, 0.0]])
    candidate_batch["valid_mask"] = torch.ones(1, 3, dtype=torch.bool)
    candidate_batch["num_queries"] = 4
    artifact = _artifact_for(candidate_batch)
    artifact["reranker_weight"] = 0.5
    monkeypatch.setattr(
        "train_dist_mod.build_rec_candidate_batch",
        lambda *_args, **_kwargs: candidate_batch,
    )

    outputs = build_rec_reranker_outputs(
        end_points, inputs, PositionReranker(), artifact
    )

    assert torch.equal(
        outputs["compact_scores"], torch.full((1, 3), 0.5)
    )
    assert outputs["query_scores"][0, 1].item() == 0.5
    assert outputs["query_scores"][0, 2].item() == 0.5
    assert outputs["query_scores"][0, 3].item() == 0.5
    assert outputs["query_scores"][0].argmax().item() == 1


def test_parent_output_builder_disables_outer_autocast_and_grad(monkeypatch):
    end_points, inputs = _runtime_batch()
    candidate_batch = build_rec_candidate_batch(
        end_points, inputs, topk_per_source=2, max_candidates=3
    )
    artifact = _artifact_for(candidate_batch)
    original = build_rec_candidate_batch
    execution = []

    def recording_builder(*args, **kwargs):
        execution.append((
            torch.is_autocast_cpu_enabled(), torch.is_grad_enabled()
        ))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "train_dist_mod.build_rec_candidate_batch", recording_builder
    )
    initial_autocast = torch.is_autocast_cpu_enabled()
    with torch.cpu.amp.autocast(enabled=True):
        build_rec_reranker_outputs(
            end_points, inputs, PositionReranker().train(), artifact
        )
        assert torch.is_autocast_cpu_enabled() is True

    assert torch.is_autocast_cpu_enabled() is initial_autocast
    assert execution == [(False, False)]


def test_build_rec_reranker_scores_honors_default_only_artifact_weight():
    end_points, inputs = _runtime_batch()
    candidate_batch = build_rec_candidate_batch(
        end_points, inputs, topk_per_source=2, max_candidates=3
    )
    artifact = _artifact_for(candidate_batch)
    artifact["reranker_weight"] = 0.0

    scores = build_rec_reranker_scores(
        end_points, inputs, PositionReranker(), artifact
    )

    selected_query = scores.argmax(dim=1)
    assert torch.equal(
        selected_query, candidate_batch["default_top1_query_index"]
    )


def test_build_rec_reranker_scores_rejects_feature_schema_mismatch():
    end_points, inputs = _runtime_batch()
    candidate_batch = build_rec_candidate_batch(
        end_points, inputs, topk_per_source=2, max_candidates=3
    )
    artifact = _artifact_for(candidate_batch)
    artifact["feature_names"] = artifact["feature_names"][:-1] + ["wrong"]

    with pytest.raises(ValueError, match="feature"):
        build_rec_reranker_scores(
            end_points, inputs, PositionReranker(), artifact
        )


def test_parse_option_exposes_rec_reranker_runtime_flags(monkeypatch, tmp_path):
    artifact = tmp_path / "reranker.pth"
    geometry_artifact = tmp_path / "geometry.pth"
    monkeypatch.setattr(sys, "argv", [
        "train_dist_mod.py",
        "--rec_reranker_checkpoint", str(artifact),
        "--rec_geometry_reranker_checkpoint", str(geometry_artifact),
        "--eval_use_rec_reranker_scores",
        "--eval_use_rec_geometry_reranker_scores",
    ])

    args = parse_option()

    assert args.rec_reranker_checkpoint == str(artifact)
    assert args.rec_geometry_reranker_checkpoint == str(geometry_artifact)
    assert args.eval_use_rec_reranker_scores is True
    assert args.eval_use_rec_geometry_reranker_scores is True
    assert args.batch_size == 8


def test_train_tester_attaches_preloaded_rec_reranker_scores():
    end_points, inputs = _runtime_batch()
    candidate_batch = build_rec_candidate_batch(
        end_points, inputs, topk_per_source=2, max_candidates=3
    )
    tester = TrainTester.__new__(TrainTester)
    tester.rec_reranker = PositionReranker()
    tester.rec_reranker_artifact = _artifact_for(candidate_batch)
    args = SimpleNamespace(
        eval_use_rec_reranker_scores=True,
        rec_reranker_checkpoint="unused-preloaded-artifact.pth",
    )

    tester._attach_rec_reranker_scores(end_points, inputs, args)

    assert end_points["rec_reranker_scores"].shape == (1, 4)
    assert torch.isneginf(end_points["rec_reranker_scores"]).sum().item() == 1


def test_train_tester_builds_evaluator_with_reranker_override():
    tester = TrainTester.__new__(TrainTester)
    tester.logger = None
    args = SimpleNamespace(
        butd_cls=False,
        model="MCLN",
        eval_use_selector_choice_scores=True,
        eval_use_rec_reranker_scores=True,
        eval_use_rec_geometry_reranker_scores=True,
    )

    evaluator = tester._build_grounding_evaluator(args, ["last_"])

    assert evaluator.eval_use_selector_choice_scores is True
    assert evaluator.eval_use_rec_reranker_scores is True
    assert evaluator.eval_use_rec_geometry_reranker_scores is True


def test_runtime_provenance_checks_backbone_and_model_inputs(tmp_path):
    checkpoint = tmp_path / "backbone.pth"
    checkpoint.write_bytes(b"matching backbone checkpoint")
    artifact = {
        "checkpoint_sha256": hashlib.sha256(
            b"matching backbone checkpoint"
        ).hexdigest(),
        "model_inputs": {
            "use_color": True,
            "use_height": False,
            "use_multiview": False,
            "butd": True,
            "butd_gt": False,
            "butd_cls": False,
        },
        "backbone_config": {
            "model": "MCLN",
            "num_target": 256,
            "num_decoder_layers": 6,
            "self_position_embedding": "loc_learned",
            "self_attend": True,
            "use_soft_token_loss": True,
            "use_contrastive_align": True,
            "detect_intermediate": True,
            "use_source_choice_selector": True,
            "source_choice_selector_sources": (
                "default,default_rank_blend_contrastive010"
            ),
            "source_choice_selector_hidden_dim": 288,
        },
    }
    args = SimpleNamespace(
        checkpoint_path=str(checkpoint),
        use_color=True,
        use_height=False,
        use_multiview=False,
        butd=True,
        butd_gt=False,
        butd_cls=False,
        model="MCLN",
        num_target=256,
        num_decoder_layers=6,
        self_position_embedding="loc_learned",
        self_attend=True,
        use_soft_token_loss=True,
        use_contrastive_align=True,
        detect_intermediate=True,
        use_source_choice_selector=True,
        source_choice_selector_sources="default,default_rank_blend_contrastive010",
        source_choice_selector_hidden_dim=288,
    )

    validate_rec_reranker_provenance(args, artifact)

    args.use_color = False
    with pytest.raises(ValueError, match="model input"):
        validate_rec_reranker_provenance(args, artifact)
    args.use_color = True
    args.self_attend = False
    with pytest.raises(ValueError, match="model config"):
        validate_rec_reranker_provenance(args, artifact)
    args.self_attend = True

    artifact["backbone_config"].update({
        "use_source_choice_selector": False,
        "use_source_moe": True,
        "source_moe_shared_source": "default",
        "source_moe_top_k": 1,
        "source_moe_balance_loss_weight": 0.01,
        "source_moe_query_layers": 1,
        "source_moe_query_heads": 4,
        "source_moe_query_dropout": 0.1,
        "source_moe_query_max_delta": 0.25,
    })
    args.use_source_choice_selector = False
    args.use_source_moe = True
    args.source_moe_shared_source = "default"
    args.source_moe_top_k = 1
    args.source_moe_balance_loss_weight = 0.01
    args.source_moe_query_layers = 1
    args.source_moe_query_heads = 4
    args.source_moe_query_dropout = 0.1
    args.source_moe_query_max_delta = 0.25
    validate_rec_reranker_provenance(args, artifact)

    artifact["backbone_config"].update({
        "source_moe_use_fallback_gate": True,
        "source_moe_gate_hidden_dim": 128,
        "source_moe_gate_candidate_top_k": 8,
        "source_moe_gate_break_cost": 2.0,
        "source_moe_gate_decision_margin": 0.0,
        "source_moe_gate_mask_utility_weight": 0.25,
        "source_moe_gate_use_evidence_features": True,
    })
    args.source_moe_use_fallback_gate = True
    args.source_moe_gate_hidden_dim = 128
    args.source_moe_gate_candidate_top_k = 8
    args.source_moe_gate_break_cost = 2.0
    args.source_moe_gate_decision_margin = 0.0
    args.source_moe_gate_mask_utility_weight = 0.25
    args.source_moe_gate_use_evidence_features = True
    validate_rec_reranker_provenance(args, artifact)
    args.source_moe_gate_use_evidence_features = False
    with pytest.raises(ValueError, match="model config"):
        validate_rec_reranker_provenance(args, artifact)
    args.source_moe_gate_use_evidence_features = True
    args.source_moe_gate_candidate_top_k = 4
    with pytest.raises(ValueError, match="model config"):
        validate_rec_reranker_provenance(args, artifact)
    args.source_moe_gate_candidate_top_k = 8
    args.source_moe_top_k = 2
    with pytest.raises(ValueError, match="model config"):
        validate_rec_reranker_provenance(args, artifact)
    args.source_moe_top_k = 1

    artifact["checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_rec_reranker_provenance(args, artifact)
