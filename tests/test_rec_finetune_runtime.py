import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import models.rec_finetune as rec_finetune
import train_dist_mod
from models.rec_candidate_adapter import FEATURE_SCHEMA_VERSION
from models.rec_geometry_reranker import (
    FLAT_PARENT_PRIOR_VERSION,
    REC_GEOMETRY_MODEL_SCHEMA_VERSION,
)
from models.rec_mask_geometry import (
    DEFAULT_REC_MASK_GEOMETRY_VARIANTS,
    MASK_GEOMETRY_SCHEMA_VERSION,
    REC_MASK_GEOMETRY_FEATURE_NAMES,
)
from models.rec_reranker import QueryReranker


INITIAL_PARENT_PATH = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/artifacts/"
    "reranker_h256_d010_lr1e3_seed0_final_contract.pth"
)
INITIAL_GEOMETRY_PATH = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
    "geometry_artifacts/selected_geometry_reranker.pth"
)


def test_rec_finetune_artifact_api_is_public():
    assert rec_finetune.REC_FINETUNE_PARENT_SCHEMA == (
        "rec-finetune-parent-v2"
    )
    assert rec_finetune.REC_FINETUNE_GEOMETRY_SCHEMA == (
        "rec-finetune-geometry-v2"
    )
    assert callable(rec_finetune.build_rec_finetune_parent_artifact)
    assert callable(rec_finetune.build_rec_finetune_geometry_artifact)
    assert callable(rec_finetune.validate_rec_finetune_artifact_pair)
    assert callable(rec_finetune.load_rec_finetune_runtime_artifacts)
    assert callable(rec_finetune.save_rec_finetune_artifact)
    assert callable(rec_finetune.sha256_file)


def test_runtime_dispatches_two_new_schemas_to_finetune_loader(
        tmp_path, monkeypatch):
    parent_path = tmp_path / "parent.pth"
    geometry_path = tmp_path / "geometry.pth"
    torch.save({"schema": "rec-finetune-parent-v2"}, parent_path)
    torch.save({"schema": "rec-finetune-geometry-v2"}, geometry_path)
    sentinel = (object(), object(), object(), object())
    calls = []

    def load_new(parent, geometry, device):
        calls.append((parent, geometry, str(device)))
        return sentinel

    monkeypatch.setattr(
        rec_finetune, "load_rec_finetune_runtime_artifacts", load_new
    )

    loaded = train_dist_mod.load_rec_geometry_runtime_artifacts(
        parent_path, geometry_path, device="cpu"
    )

    assert loaded is sentinel
    assert calls == [(parent_path, geometry_path, "cpu")]


def test_stable_snapshot_rejects_symlink_entry(tmp_path):
    target = tmp_path / "target.pth"
    target.write_bytes(b"stable artifact bytes")
    link = tmp_path / "link.pth"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink|regular"):
        rec_finetune._stable_artifact_snapshot(link, "test artifact")


def test_stable_snapshot_rejects_replace_between_lstat_and_open(
        tmp_path, monkeypatch):
    logical = tmp_path / "artifact.pth"
    replacement = tmp_path / "replacement.pth"
    logical.write_bytes(b"initial artifact bytes")
    replacement.write_bytes(b"replacement artifact bytes")
    real_open = rec_finetune.os.open
    open_calls = []

    def replace_then_open(path, flags, *args, **kwargs):
        open_calls.append(path)
        replacement.replace(logical)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(rec_finetune.os, "open", replace_then_open)

    with pytest.raises(ValueError, match="changed|stable"):
        rec_finetune._stable_artifact_snapshot(logical, "test artifact")
    assert open_calls


def test_stable_snapshot_returns_exact_normal_bytes_and_sha(tmp_path):
    path = tmp_path / "artifact.pth"
    payload = b"ordinary stable artifact bytes"
    path.write_bytes(payload)

    resolved, snapshot, digest = rec_finetune._stable_artifact_snapshot(
        path, "test artifact"
    )

    assert resolved == path.resolve()
    assert snapshot == payload
    assert digest == hashlib.sha256(payload).hexdigest()


def _metrics(hits025, hits050, sample_count=10):
    acc025 = hits025 / float(sample_count)
    acc050 = hits050 / float(sample_count)
    return {
        "sample_count": sample_count,
        "hits025": hits025,
        "hits050": hits050,
        "acc025": acc025,
        "acc050": acc050,
        "score": min(acc025 / 0.60, acc050 / 0.47)
        + 0.1 * (acc025 + acc050),
    }


def _model_inputs():
    return {
        "use_color": True,
        "use_height": False,
        "use_multiview": False,
        "butd": True,
        "butd_gt": False,
        "butd_cls": False,
    }


def _backbone_config():
    return {
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
    }


def _provenance():
    baseline = _metrics(6, 4)
    selected = _metrics(7, 5)
    return {
        "initial_backbone_sha256": (
            rec_finetune.AUTHORITATIVE_REC_FINETUNE_INITIAL_BACKBONE_SHA256
        ),
        "initial_parent_artifact_sha256": (
            rec_finetune
            .AUTHORITATIVE_REC_FINETUNE_INITIAL_PARENT_ARTIFACT_SHA256
        ),
        "initial_geometry_artifact_sha256": (
            rec_finetune
            .AUTHORITATIVE_REC_FINETUNE_INITIAL_GEOMETRY_ARTIFACT_SHA256
        ),
        "authoritative_split_mapping_sha256": (
            rec_finetune.AUTHORITATIVE_REC_FINETUNE_SPLIT_SEED0[
                "mapping_sha256"
            ]
        ),
        "selected_step": 306,
        "validation_data_accessed": False,
        "normalization_policy": "fixed-initial-artifact-v1",
        "parent_reranker_weight": 0.9,
        "geometry_reranker_weight": 1.0,
        "matcher_costs": {"class": 1.0, "bbox": 5.0, "giou": 2.0},
        "loss_scales": {
            "mask": 0.1,
            "consistency": 0.1,
            "source_choice": 0.0,
            "parent": 1.0,
            "geometry": 1.0,
        },
        "reranker_loss_weights": {
            "ranking": 1.0,
            "threshold": 1.0,
            "iou": 0.5,
        },
        "reranker_ranking_objectives": {
            "parent": {
                "name": "single-best-iou-listwise-v1",
                "tier_pairwise_alpha": 0.0,
            },
            "geometry": {
                "name": "best-tier-pairwise-v1",
                "tier_pairwise_alpha": 1.0,
                "thresholds": [0.25, 0.50],
                "threshold_operator": "strict_gt",
                "positive_policy": "all_valid_candidates_in_best_tier",
                "negative_policy": "all_valid_candidates_below_best_tier",
                "loss": "softplus(negative_logit-positive_logit)",
                "pair_reduction": "mean_within_row",
                "row_reduction": "mean_over_informative_rows",
                "no_pair_policy": "differentiable_zero",
            },
        },
        "optimizer_groups": [
            {
                "name": "mcln_decoder_box",
                "lr": 2e-5,
                "weight_decay": 5e-4,
                "grad_clip": 0.1,
            },
            {
                "name": "parent_reranker",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "grad_clip": 1.0,
            },
            {
                "name": "geometry_reranker",
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "grad_clip": 1.0,
            },
        ],
        "max_steps": 1836,
        "calibration_steps": list(rec_finetune.CALIBRATION_STEPS),
        "mcln_trainable_parameter_names": [
            "decoder.layer.weight",
            "decoder_query_proj.weight",
            "prediction_heads.weight",
            "proposal_head.weight",
        ],
        "calibration_history": [
            {
                "step": 0,
                "metrics": baseline,
                "eligible": True,
                "regression": False,
                "action": "continue",
                "best_step": 0,
            },
            {
                "step": 306,
                "metrics": selected,
                "eligible": True,
                "regression": False,
                "action": "continue",
                "best_step": 306,
            },
        ],
    }


def test_provenance_objective_validation_is_independent_of_mutable_helper(
        monkeypatch):
    provenance = _provenance()
    selected_metrics = _metrics(7, 5)
    monkeypatch.setattr(
        rec_finetune,
        "rec_finetune_ranking_objective_contract",
        lambda: {
            "parent": {"name": "forged", "tier_pairwise_alpha": 1.0},
            "geometry": {"name": "forged", "tier_pairwise_alpha": 0.0},
        },
    )

    assert rec_finetune._validate_provenance(
        provenance, selected_metrics
    ) is None


def _initial_artifacts(parent_dim=152):
    parent_names = [
        "parent_{:03d}".format(index) for index in range(parent_dim)
    ]
    variants = [dict(value) for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS]
    geometry_names = (
        parent_names + list(REC_MASK_GEOMETRY_FEATURE_NAMES)
        + ["parent_score", "parent_is_deployed_top1"]
    )
    common = {
        "candidate_rule": {"topk_per_source": 8, "max_candidates": 16},
        "checkpoint_sha256": (
            rec_finetune.AUTHORITATIVE_REC_FINETUNE_INITIAL_BACKBONE_SHA256
        ),
        "checkpoint_epoch": 71,
        "target_iou_policy": "root_only",
        "model_inputs": _model_inputs(),
        "backbone_config": _backbone_config(),
    }
    parent = dict(common)
    parent.update({
        "artifact_version": 1,
        "adapter_schema_version": FEATURE_SCHEMA_VERSION,
        "input_dim": len(parent_names),
        "feature_names": parent_names,
        "feature_mean": torch.arange(len(parent_names), dtype=torch.float64),
        "feature_std": torch.arange(
            1, len(parent_names) + 1, dtype=torch.float64
        ),
        "score_mode": "rank_blend",
        "reranker_weight": 0.9,
    })
    geometry = dict(common)
    geometry.update({
        "artifact_version": 2,
        "model_schema_version": REC_GEOMETRY_MODEL_SCHEMA_VERSION,
        "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "input_dim": len(geometry_names),
        "feature_names": geometry_names,
        "feature_mean": torch.arange(
            len(geometry_names), dtype=torch.float64
        ),
        "feature_std": torch.arange(
            1, len(geometry_names) + 1, dtype=torch.float64
        ),
        "variant_names": [value["name"] for value in variants],
        "variant_configs": variants,
        "regressed_variant_index": 0,
        "min_points": 8,
        "max_point_fraction": 0.5,
        "num_queries": 256,
        "flat_parent_prior_version": FLAT_PARENT_PRIOR_VERSION,
        "tie_policy": "score-desc-flat-index-asc-v1",
        "score_mode": "parent-flat-rank-blend-v1",
        "geometry_weight": 1.0,
        "evaluator_filter_policy": "evaluator-valid-no-gt-filter-v1",
        "filter_non_gt_boxes": False,
    })
    return parent, geometry


def _payload_equal(first, second):
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (
            isinstance(first, torch.Tensor)
            and isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and first.device == second.device
            and torch.equal(first, second)
        )
    if isinstance(first, dict) or isinstance(second, dict):
        return (
            isinstance(first, dict)
            and isinstance(second, dict)
            and set(first) == set(second)
            and all(_payload_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(_payload_equal(a, b) for a, b in zip(first, second))
        )
    return first == second


def _artifact_pair(tmp_path, checkpoint_sha="4" * 64,
                   initial_parent_path=INITIAL_PARENT_PATH,
                   initial_geometry_path=INITIAL_GEOMETRY_PATH):
    torch.manual_seed(4)
    parent_model = QueryReranker(
        rec_finetune.REC_FINETUNE_PARENT_INPUT_DIM,
        hidden_dim=8,
        dropout=0.0,
    )
    geometry_model = QueryReranker(
        rec_finetune.REC_FINETUNE_GEOMETRY_INPUT_DIM,
        hidden_dim=8,
        dropout=0.0,
    )
    provenance = _provenance()
    metrics = copy.deepcopy(
        provenance["calibration_history"][-1]["metrics"]
    )
    parent_artifact = rec_finetune.build_rec_finetune_parent_artifact(
        parent_model,
        initial_parent_path,
        checkpoint_sha,
        72,
        provenance,
        metrics,
    )
    parent_path = tmp_path / "parent.pth"
    rec_finetune.save_rec_finetune_artifact(parent_path, parent_artifact)
    parent_sha = rec_finetune.sha256_file(parent_path)
    geometry_artifact = rec_finetune.build_rec_finetune_geometry_artifact(
        geometry_model,
        initial_geometry_path,
        parent_artifact,
        parent_sha,
        checkpoint_sha,
        72,
        provenance,
        metrics,
    )
    return {
        "initial_parent_path": initial_parent_path,
        "initial_geometry_path": initial_geometry_path,
        "parent_model": parent_model,
        "geometry_model": geometry_model,
        "parent_artifact": parent_artifact,
        "geometry_artifact": geometry_artifact,
        "parent_path": parent_path,
        "parent_sha": parent_sha,
    }


def test_artifact_pair_round_trip_preserves_exact_scores_and_metadata(tmp_path):
    fixture = _artifact_pair(tmp_path)
    geometry_path = tmp_path / "geometry.pth"
    rec_finetune.save_rec_finetune_artifact(
        geometry_path, fixture["geometry_artifact"]
    )
    parent, parent_artifact, geometry, geometry_artifact = (
        rec_finetune.load_rec_finetune_runtime_artifacts(
            fixture["parent_path"], geometry_path, device="cpu"
        )
    )
    parent_features = torch.randn(2, 5, parent.input_dim)
    geometry_features = torch.randn(2, 7, geometry.input_dim)
    parent_valid = torch.ones(2, 5, dtype=torch.bool)
    geometry_valid = torch.ones(2, 7, dtype=torch.bool)
    fixture["parent_model"].eval()
    fixture["geometry_model"].eval()
    with torch.no_grad():
        expected_parent = fixture["parent_model"](
            parent_features, parent_valid
        )["ranking_logits"]
        actual_parent = parent(parent_features, parent_valid)["ranking_logits"]
        expected_geometry = fixture["geometry_model"](
            geometry_features, geometry_valid
        )["ranking_logits"]
        actual_geometry = geometry(
            geometry_features, geometry_valid
        )["ranking_logits"]

    assert torch.equal(actual_parent, expected_parent)
    assert torch.equal(actual_geometry, expected_geometry)
    assert _payload_equal(parent_artifact, fixture["parent_artifact"])
    assert _payload_equal(geometry_artifact, fixture["geometry_artifact"])
    assert parent._artifact_sha256 == fixture["parent_sha"]
    assert geometry._artifact_sha256 == rec_finetune.sha256_file(geometry_path)
    assert not parent.training and not geometry.training
    assert not any(value.requires_grad for value in parent.parameters())
    assert not any(value.requires_grad for value in geometry.parameters())


def test_builders_copy_initial_metadata_without_mutating_inputs(tmp_path):
    parent_bytes = INITIAL_PARENT_PATH.read_bytes()
    geometry_bytes = INITIAL_GEOMETRY_PATH.read_bytes()
    initial_parent = torch.load(INITIAL_PARENT_PATH, map_location="cpu")
    initial_geometry = torch.load(INITIAL_GEOMETRY_PATH, map_location="cpu")

    fixture = _artifact_pair(tmp_path)

    assert INITIAL_PARENT_PATH.read_bytes() == parent_bytes
    assert INITIAL_GEOMETRY_PATH.read_bytes() == geometry_bytes
    assert torch.equal(
        fixture["parent_artifact"]["feature_mean"],
        initial_parent["feature_mean"].detach().cpu().float(),
    )
    assert torch.equal(
        fixture["geometry_artifact"]["feature_std"],
        initial_geometry["feature_std"].detach().cpu().float(),
    )


def test_builders_reject_non_authoritative_initial_lineage():
    initial_parent = torch.load(INITIAL_PARENT_PATH, map_location="cpu")
    provenance = _provenance()
    metrics = copy.deepcopy(
        provenance["calibration_history"][-1]["metrics"]
    )

    with pytest.raises(ValueError, match="path|path-like"):
        rec_finetune.build_rec_finetune_parent_artifact(
            QueryReranker(
                initial_parent["input_dim"], hidden_dim=8, dropout=0.0
            ),
            initial_parent,
            "4" * 64,
            72,
            provenance,
            metrics,
        )


@pytest.mark.parametrize("family", ["parent", "geometry"])
def test_builders_reject_tampered_initial_file_bytes(
        tmp_path, family):
    source = (
        INITIAL_PARENT_PATH if family == "parent" else INITIAL_GEOMETRY_PATH
    )
    tampered = bytearray(source.read_bytes())
    tampered[-17] ^= 1
    tampered_path = tmp_path / "tampered-{}.pth".format(family)
    tampered_path.write_bytes(tampered)
    provenance = _provenance()
    metrics = copy.deepcopy(
        provenance["calibration_history"][-1]["metrics"]
    )

    with pytest.raises(
            ValueError, match="{} initial artifact SHA".format(family)):
        if family == "parent":
            rec_finetune.build_rec_finetune_parent_artifact(
                QueryReranker(152, hidden_dim=8, dropout=0.0),
                tampered_path,
                "4" * 64,
                72,
                provenance,
                metrics,
            )
        else:
            fixture = _artifact_pair(tmp_path)
            rec_finetune.build_rec_finetune_geometry_artifact(
                QueryReranker(179, hidden_dim=8, dropout=0.0),
                tampered_path,
                fixture["parent_artifact"],
                fixture["parent_sha"],
                "4" * 64,
                72,
                provenance,
                metrics,
            )


def test_pair_rejects_matching_tamper_of_initial_parent_sha(tmp_path):
    fixture = _artifact_pair(tmp_path)
    parent = copy.deepcopy(fixture["parent_artifact"])
    geometry = copy.deepcopy(fixture["geometry_artifact"])
    parent["provenance"]["initial_parent_artifact_sha256"] = "5" * 64
    geometry["provenance"]["initial_parent_artifact_sha256"] = "5" * 64

    with pytest.raises(ValueError, match="initial.*parent|provenance"):
        rec_finetune.validate_rec_finetune_artifact_pair(
            parent, geometry, fixture["parent_sha"]
        )


def test_builders_reject_generic_four_dimensional_artifact_pair(tmp_path):
    fixture = _artifact_pair(tmp_path)
    artifact = copy.deepcopy(fixture["parent_artifact"])
    model = QueryReranker(4, hidden_dim=8, dropout=0.0)
    state = {
        name: value.detach().cpu().float().clone()
        for name, value in model.state_dict().items()
    }
    artifact["input_dim"] = 4
    artifact["model_config"]["input_dim"] = 4
    artifact["model_state_dict"] = state
    artifact["model_state_sha256"] = rec_finetune._state_dict_sha256(state)
    artifact["feature_names"] = artifact["feature_names"][:4]
    artifact["feature_mean"] = artifact["feature_mean"][:4].clone()
    artifact["feature_std"] = artifact["feature_std"][:4].clone()
    artifact["feature_mean_sha256"] = rec_finetune._tensor_sha256(
        artifact["feature_mean"]
    )
    artifact["feature_std_sha256"] = rec_finetune._tensor_sha256(
        artifact["feature_std"]
    )

    with pytest.raises(ValueError, match="152|dimension"):
        rec_finetune.save_rec_finetune_artifact(
            tmp_path / "bad-parent.pth", artifact
        )


def test_geometry_validator_rejects_coherent_nonproduction_dimension(tmp_path):
    fixture = _artifact_pair(tmp_path)
    artifact = copy.deepcopy(fixture["geometry_artifact"])
    bad_dim = artifact["input_dim"] - 1
    model = QueryReranker(bad_dim, hidden_dim=8, dropout=0.0)
    state = {
        name: value.detach().cpu().float().clone()
        for name, value in model.state_dict().items()
    }
    artifact["input_dim"] = bad_dim
    artifact["model_config"]["input_dim"] = bad_dim
    artifact["model_state_dict"] = state
    artifact["model_state_sha256"] = rec_finetune._state_dict_sha256(state)
    artifact["feature_names"] = artifact["feature_names"][:bad_dim]
    artifact["feature_mean"] = artifact["feature_mean"][:bad_dim].clone()
    artifact["feature_std"] = artifact["feature_std"][:bad_dim].clone()
    artifact["feature_mean_sha256"] = rec_finetune._tensor_sha256(
        artifact["feature_mean"]
    )
    artifact["feature_std_sha256"] = rec_finetune._tensor_sha256(
        artifact["feature_std"]
    )

    with pytest.raises(ValueError, match="179|dimension"):
        rec_finetune.save_rec_finetune_artifact(
            tmp_path / "bad-geometry.pth", artifact
        )


@pytest.mark.parametrize(
    "field,value",
    [("butd", False), ("butd_gt", True), ("butd_cls", True)],
)
def test_pair_rejects_coherent_non_no_gt_model_inputs(
        tmp_path, field, value):
    fixture = _artifact_pair(tmp_path)
    parent = copy.deepcopy(fixture["parent_artifact"])
    geometry = copy.deepcopy(fixture["geometry_artifact"])
    parent["model_inputs"][field] = value
    geometry["model_inputs"][field] = value
    geometry["parent_runtime_binding"]["model_inputs"][field] = value
    geometry["parent_inference_contract"] = (
        rec_finetune._parent_inference_contract(
            geometry["parent_runtime_binding"],
            geometry["calibration_metrics"]["sample_count"],
        )
    )

    with pytest.raises(ValueError, match="no-GT|model inputs"):
        rec_finetune.validate_rec_finetune_artifact_pair(
            parent, geometry, fixture["parent_sha"]
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p, g: p.update(checkpoint_sha256="5" * 64), "checkpoint"),
        (
            lambda p, g: p["provenance"].update(
                authoritative_split_mapping_sha256="5" * 64
            ),
            "split",
        ),
        (
            lambda p, g: g["provenance"].update(selected_step=17),
            "step",
        ),
        (
            lambda p, g: g["provenance"].update(
                validation_data_accessed=True
            ),
            "validation",
        ),
        (
            lambda p, g: g.update(parent_artifact_sha256="5" * 64),
            "parent",
        ),
        (
            lambda p, g: p["feature_mean"].add_(1.0),
            "normalization",
        ),
        (
            lambda p, g: next(iter(g["model_state_dict"].values())).add_(1.0),
            "model state",
        ),
    ],
)
def test_artifact_pair_rejects_independent_binding_tamper(
        tmp_path, mutation, match):
    fixture = _artifact_pair(tmp_path)
    parent = copy.deepcopy(fixture["parent_artifact"])
    geometry = copy.deepcopy(fixture["geometry_artifact"])
    mutation(parent, geometry)

    with pytest.raises(ValueError, match=match):
        rec_finetune.validate_rec_finetune_artifact_pair(
            parent, geometry, fixture["parent_sha"]
        )


def test_artifact_pair_rejects_coherent_ranking_objective_tamper(tmp_path):
    fixture = _artifact_pair(tmp_path)
    parent = copy.deepcopy(fixture["parent_artifact"])
    geometry = copy.deepcopy(fixture["geometry_artifact"])
    for artifact in (parent, geometry):
        artifact["provenance"]["reranker_ranking_objectives"][
            "geometry"
        ]["tier_pairwise_alpha"] = 0.0

    with pytest.raises(ValueError, match="ranking objective"):
        rec_finetune.validate_rec_finetune_artifact_pair(
            parent, geometry, fixture["parent_sha"]
        )


def test_atomic_save_leaves_no_partial_final_on_replace_failure(
        tmp_path, monkeypatch):
    fixture = _artifact_pair(tmp_path)
    output = tmp_path / "unpublished.pth"

    def fail_replace(_source, _destination):
        raise OSError("injected publication failure")

    monkeypatch.setattr(rec_finetune.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        rec_finetune.save_rec_finetune_artifact(
            output, fixture["parent_artifact"]
        )

    assert not output.exists()
    assert not list(tmp_path.glob("unpublished.pth.tmp.*"))


def test_runtime_dispatch_rejects_mixed_new_and_legacy_artifacts(
        tmp_path, monkeypatch):
    parent_path = tmp_path / "parent.pth"
    geometry_path = tmp_path / "geometry.pth"
    torch.save({"schema": rec_finetune.REC_FINETUNE_PARENT_SCHEMA}, parent_path)
    torch.save({"artifact_version": 2}, geometry_path)
    monkeypatch.setattr(
        rec_finetune,
        "load_rec_finetune_runtime_artifacts",
        lambda *_args, **_kwargs: pytest.fail("mixed pair reached new loader"),
    )

    with pytest.raises(ValueError, match="mixed"):
        train_dist_mod.load_rec_geometry_runtime_artifacts(
            parent_path, geometry_path, device="cpu"
        )


def test_new_runtime_provenance_uses_new_validator_and_common_checkpoint(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"synthetic selected MCLN checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    fixture = _artifact_pair(tmp_path, checkpoint_sha=checkpoint_sha)
    geometry_path = tmp_path / "geometry.pth"
    rec_finetune.save_rec_finetune_artifact(
        geometry_path, fixture["geometry_artifact"]
    )
    parent, parent_artifact, geometry, geometry_artifact = (
        rec_finetune.load_rec_finetune_runtime_artifacts(
            fixture["parent_path"], geometry_path, device="cpu"
        )
    )
    args = SimpleNamespace(
        checkpoint_path=str(checkpoint),
        **_model_inputs(),
        **_backbone_config()
    )
    pair_calls = []
    original_pair_validator = (
        rec_finetune.validate_rec_finetune_artifact_pair
    )

    def validate_new(*values, **kwargs):
        pair_calls.append((values, kwargs))
        return original_pair_validator(*values, **kwargs)

    monkeypatch.setattr(
        rec_finetune, "validate_rec_finetune_artifact_pair", validate_new
    )
    monkeypatch.setattr(
        "scripts.train_rec_geometry_reranker._validate_live_parent_state",
        lambda *_args, **_kwargs: pytest.fail("new pair reached old validator"),
    )
    monkeypatch.setattr(
        "scripts.train_rec_geometry_reranker.validate_geometry_artifact",
        lambda *_args, **_kwargs: pytest.fail("new pair reached old validator"),
    )
    monkeypatch.setattr(
        train_dist_mod, "_module_device", lambda _model: torch.device("cuda:0")
    )

    projection = train_dist_mod.validate_rec_geometry_runtime_provenance(
        args,
        parent,
        parent_artifact,
        geometry,
        geometry_artifact,
        torch.device("cuda:0"),
    )

    assert projection["device_type"] == "cuda"
    assert len(pair_calls) == 1
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="fingerprint"):
        train_dist_mod.validate_rec_geometry_runtime_provenance(
            args,
            parent,
            parent_artifact,
            geometry,
            geometry_artifact,
            torch.device("cuda:0"),
        )
