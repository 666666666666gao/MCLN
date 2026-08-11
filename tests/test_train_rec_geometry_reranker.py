import copy
import hashlib
from pathlib import Path

import pytest
import torch

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
from scripts.rec_geometry_cache import canonical_json_sha256
from scripts.train_rec_reranker import save_reranker_artifact
from scripts.train_rec_geometry_reranker import (
    AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
    AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256,
    CALIBRATION_METRIC_FIELDS,
    DEFAULT_GEOMETRY_WEIGHTS,
    GEOMETRY_ARTIFACT_FIELDS,
    GEOMETRY_ARTIFACT_VERSION,
    GEOMETRY_INPUT_DIM,
    PARENT_INFERENCE_CONTRACT_FIELDS,
    PARENT_INFERENCE_LOCAL_BATCH_SIZE,
    build_geometry_artifact,
    build_geometry_training_batch,
    build_scene_split_metadata,
    calibration_score,
    choose_best_geometry_blend,
    compute_geometry_feature_stats,
    deterministic_scene_split,
    evaluate_geometry_blends,
    fit_and_save_geometry_model,
    load_geometry_reranker_artifact,
    load_geometry_training_data,
    load_parent_reranker_snapshot,
    materialize_parent_scores,
    parse_args,
    save_geometry_reranker_artifact,
    train_geometry_reranker,
    validate_geometry_artifact,
)


BASE_FEATURE_NAMES = tuple("base_{:03d}".format(index) for index in range(152))
GEOMETRY_FEATURE_NAMES = tuple(REC_MASK_GEOMETRY_FEATURE_NAMES)
VARIANT_COUNT = len(DEFAULT_REC_MASK_GEOMETRY_VARIANTS)
CANDIDATE_COUNT = 16


def _rank_fixture_scores():
    default_scores = torch.empty(CANDIDATE_COUNT, dtype=torch.float32)
    default_scores[1] = 20.0
    for rank, position in enumerate(range(2, 10), 1):
        default_scores[position] = 20.0 - rank
    default_scores[0] = 11.0
    for rank, position in enumerate(range(10, 16), 10):
        default_scores[position] = 20.0 - rank

    ranking_logits = torch.empty(CANDIDATE_COUNT, dtype=torch.float32)
    ranking_logits[0] = 20.0
    ranking_logits[1] = 19.0
    for rank, position in enumerate(range(2, 16), 2):
        ranking_logits[position] = 20.0 - rank
    return default_scores, ranking_logits


def _joined_row(index, scan_id, feature_offset=0.0):
    default_scores, _ = _rank_fixture_scores()
    query_indices = torch.tensor(
        [5, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        dtype=torch.long,
    )
    base_features = torch.arange(
        CANDIDATE_COUNT * len(BASE_FEATURE_NAMES), dtype=torch.float32
    ).reshape(CANDIDATE_COUNT, -1)
    base_features = base_features / 1000.0 + float(feature_offset)
    base_valid = torch.ones(CANDIDATE_COUNT, dtype=torch.bool)
    boxes = torch.zeros(CANDIDATE_COUNT, 6, dtype=torch.float32)
    boxes[:, 3:] = 1.0

    geometry_features = torch.zeros(
        CANDIDATE_COUNT,
        VARIANT_COUNT,
        len(GEOMETRY_FEATURE_NAMES),
        dtype=torch.float32,
    )
    for query_index in range(CANDIDATE_COUNT):
        for variant_index in range(VARIANT_COUNT):
            geometry_features[query_index, variant_index, 0] = (
                10.0 * query_index + variant_index + feature_offset
            )
    geometry_boxes = boxes[:, None, :].repeat(1, VARIANT_COUNT, 1)
    geometry_boxes[..., :3] += torch.arange(
        VARIANT_COUNT, dtype=torch.float32
    ).view(1, -1, 1) / 10.0
    geometry_valid = base_valid[:, None].expand(-1, VARIANT_COUNT).clone()
    geometry_ious = torch.linspace(
        0.0,
        1.0,
        steps=CANDIDATE_COUNT * VARIANT_COUNT,
        dtype=torch.float32,
    ).reshape(CANDIDATE_COUNT, VARIANT_COUNT)
    base = {
        "dataset_index": index,
        "scan_id": scan_id,
        "target_id": index,
        "features": base_features,
        "boxes": boxes,
        "query_indices": query_indices,
        "valid_mask": base_valid,
        "default_scores": default_scores,
        "contrastive_scores": -default_scores,
        "candidate_ious": geometry_ious[:, 0].clone(),
        "default_top1_query_index": 1,
    }
    geometry = {
        "dataset_index": index,
        "scan_id": scan_id,
        "target_id": index,
        "default_top1_query_index": 1,
        "query_indices": query_indices.clone(),
        "candidate_valid": base_valid.clone(),
        "geometry_boxes": geometry_boxes,
        "geometry_valid": geometry_valid,
        "evaluator_valid": geometry_valid.clone(),
        "geometry_features": geometry_features,
        "geometry_ious": geometry_ious,
        "source_rejection_codes": torch.zeros(
            CANDIDATE_COUNT, VARIANT_COUNT, dtype=torch.int16
        ),
    }
    return {"base": base, "geometry": geometry}


class RecordingParent(torch.nn.Module):
    def __init__(self, ranking_logits):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("ranking_logits", ranking_logits.float().clone())
        self.last_features = None
        self.saw_grad_enabled = None
        self.call_count = 0
        self.batch_sizes = []
        self.input_dtypes = []
        self.autocast_enabled = []
        self.cpu_autocast_enabled = []

    def forward(self, features, valid_mask):
        self.call_count += 1
        self.batch_sizes.append(int(features.shape[0]))
        self.input_dtypes.append(features.dtype)
        self.autocast_enabled.append(torch.is_autocast_enabled())
        self.cpu_autocast_enabled.append(torch.is_autocast_cpu_enabled())
        self.last_features = features.detach().cpu().clone()
        self.saw_grad_enabled = torch.is_grad_enabled()
        logits = self.ranking_logits.to(features).view(1, -1).expand(
            features.shape[0], -1
        )
        return {
            "ranking_logits": logits.masked_fill(~valid_mask.bool(), -1e4),
            "threshold_logits": features.new_zeros(
                features.shape[:2] + (2,)
            ),
            "iou_estimate": features.new_zeros(features.shape[:2]),
        }


def _parent():
    _, ranking_logits = _rank_fixture_scores()
    model = RecordingParent(ranking_logits)
    model.train()
    artifact = {
        "feature_mean": torch.ones(len(BASE_FEATURE_NAMES)),
        "feature_std": torch.full((len(BASE_FEATURE_NAMES),), 2.0),
        "feature_names": list(BASE_FEATURE_NAMES),
        "input_dim": len(BASE_FEATURE_NAMES),
        "reranker_weight": 0.9,
        "score_mode": "rank_blend",
        "backbone_config": {"num_target": 256},
    }
    return model, artifact


def test_parent_scores_materialize_in_production_batches_and_natural_remainder():
    rows = [
        _joined_row(index, "scene_{:02d}".format(index))
        for index in range(25)
    ]
    parent = _parent()

    scores = materialize_parent_scores(
        list(reversed(rows)), parent, device="cpu"
    )

    assert PARENT_INFERENCE_LOCAL_BATCH_SIZE == 12
    assert scores.shape == (25, CANDIDATE_COUNT)
    assert scores.dtype == torch.float32
    assert scores.device.type == "cpu"
    assert parent[0].batch_sizes == [12, 12, 1]
    assert parent[0].input_dtypes == [torch.float32] * 3
    assert parent[0].autocast_enabled == [False] * 3
    assert parent[0].saw_grad_enabled is False
    assert parent[0].training is False
    with pytest.raises(ValueError, match="local batch size"):
        materialize_parent_scores(
            rows,
            parent,
            device="cpu",
            local_batch_size=12.0,
        )


def test_parent_materialization_rejects_world_size_before_move_forward_or_cache(
        monkeypatch):
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent = _parent()
    to_calls = []
    original_to = parent[0].to

    def record_to(*args, **kwargs):
        to_calls.append((args, kwargs))
        return original_to(*args, **kwargs)

    monkeypatch.setattr(parent[0], "to", record_to)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match="world_size=1"):
        materialize_parent_scores(rows, parent, device="cpu")

    assert to_calls == []
    assert parent[0].call_count == 0
    assert not hasattr(parent[0], "_geometry_parent_score_cache")


def test_cpu_parent_materialization_disables_outer_autocast_without_leak():
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent = _parent()
    initial_cpu_autocast = torch.is_autocast_cpu_enabled()

    with torch.cpu.amp.autocast(enabled=True):
        assert torch.is_autocast_cpu_enabled() is True
        scores = materialize_parent_scores(rows, parent, device="cpu")
        assert torch.is_autocast_cpu_enabled() is True

    assert torch.is_autocast_cpu_enabled() is initial_cpu_autocast
    assert parent[0].cpu_autocast_enabled == [False]
    assert parent[0].input_dtypes == [torch.float32]
    assert scores.dtype == torch.float32
    assert parent[0]._geometry_parent_score_cache[
        "parent_inference_contract"
    ]["autocast"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_parent_materialization_disables_outer_autocast_without_leak():
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent = _parent()
    initial_cuda_autocast = torch.is_autocast_enabled()

    with torch.cuda.amp.autocast(enabled=True):
        assert torch.is_autocast_enabled() is True
        scores = materialize_parent_scores(rows, parent, device="cuda:0")
        assert torch.is_autocast_enabled() is True

    assert torch.is_autocast_enabled() is initial_cuda_autocast
    assert parent[0].autocast_enabled == [False]
    assert parent[0].input_dtypes == [torch.float32]
    assert scores.dtype == torch.float32
    assert parent[0]._geometry_parent_score_cache[
        "parent_inference_contract"
    ]["autocast"] is False


def test_sealed_parent_score_cache_rejects_a_missing_row_without_forward():
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent = _parent()
    materialize_parent_scores(rows, parent, device="cpu")
    call_count = parent[0].call_count
    uncached_copy = copy.deepcopy(rows[0])

    with pytest.raises(ValueError, match="sealed.*missing"):
        build_geometry_training_batch([uncached_copy], parent)

    assert parent[0].call_count == call_count


def test_sealed_parent_cache_signature_binds_device_and_matmul_tf32():
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent = _parent()
    materialize_parent_scores(rows, parent, device="cpu")
    cache = parent[0]._geometry_parent_score_cache

    assert cache["signature"]["device_type"] == "cpu"
    assert cache["signature"]["device_index"] is None
    assert cache["signature"]["allow_tf32"] is False

    cache["signature"] = dict(cache["signature"], allow_tf32=True)
    with pytest.raises(ValueError, match="sealed.*signature"):
        build_geometry_training_batch(rows, parent)


@pytest.mark.parametrize("change", ["tf32", "device"])
def test_explicit_materialization_rejects_sealed_signature_change_immutably(
        monkeypatch, change):
    import scripts.train_rec_geometry_reranker as trainer

    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent = _parent()
    if change == "tf32":
        tf32 = {"enabled": True}
        monkeypatch.setattr(
            trainer,
            "_parent_matmul_allow_tf32",
            lambda _device: tf32["enabled"],
        )
    materialize_parent_scores(rows, parent, device="cpu")
    cache = parent[0]._geometry_parent_score_cache
    expected_contract = copy.deepcopy(cache["parent_inference_contract"])
    expected_digest = cache["train_parent_score_content_sha256"]
    expected_order = cache["ordered_row_object_ids"]
    expected_rows = {
        key: (entry[0], entry[1].clone())
        for key, entry in cache["rows"].items()
    }
    call_count = parent[0].call_count

    if change == "tf32":
        tf32["enabled"] = False
    else:
        original = trainer._parent_score_cache_signature

        def changed_signature(*args, **kwargs):
            signature = original(*args, **kwargs)
            if kwargs.get("device") is None:
                return dict(
                    signature,
                    device_type="cuda",
                    device_index=0,
                )
            return signature

        monkeypatch.setattr(
            trainer, "_parent_score_cache_signature", changed_signature
        )

    with pytest.raises(ValueError, match="sealed.*signature"):
        materialize_parent_scores(rows, parent, device="cpu")

    assert parent[0].call_count == call_count
    assert parent[0]._geometry_parent_score_cache is cache
    assert cache["parent_inference_contract"] == expected_contract
    assert cache["train_parent_score_content_sha256"] == expected_digest
    assert cache["ordered_row_object_ids"] == expected_order
    assert set(cache["rows"]) == set(expected_rows)
    for key, (expected_row, expected_score) in expected_rows.items():
        assert cache["rows"][key][0] is expected_row
        assert torch.equal(cache["rows"][key][1], expected_score)


def test_sealed_materialization_only_allows_exact_row_set_reentry():
    rows = [
        _joined_row(0, "scene_a"),
        _joined_row(1, "scene_b"),
        _joined_row(2, "scene_c"),
    ]
    parent = _parent()
    expected_scores = materialize_parent_scores(rows, parent, device="cpu")
    cache = parent[0]._geometry_parent_score_cache
    expected_metadata = {
        key: copy.deepcopy(cache[key])
        for key in (
            "parent_inference_contract",
            "train_parent_score_content_sha256",
            "ordered_row_object_ids",
        )
    }
    call_count = parent[0].call_count

    changed_sets = [
        rows[:2],
        rows + [_joined_row(3, "scene_d")],
        [copy.deepcopy(rows[0]), rows[1], rows[2]],
    ]
    for changed_rows in changed_sets:
        with pytest.raises(ValueError, match="sealed.*exact.*row"):
            materialize_parent_scores(changed_rows, parent, device="cpu")
        assert parent[0].call_count == call_count
        for key, expected in expected_metadata.items():
            assert cache[key] == expected

    repeated = materialize_parent_scores(
        list(reversed(rows)), parent, device="cpu"
    )
    assert torch.equal(repeated, expected_scores)
    assert parent[0].call_count == call_count


def test_exact_sealed_reentry_revalidates_cache_without_resealing_tamper():
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent = _parent()
    materialize_parent_scores(rows, parent, device="cpu")
    cache = parent[0]._geometry_parent_score_cache
    expected_contract = copy.deepcopy(cache["parent_inference_contract"])
    expected_digest = cache["train_parent_score_content_sha256"]
    first_id = cache["ordered_row_object_ids"][0]
    cache["rows"][first_id][1].add_(0.25)
    call_count = parent[0].call_count

    with pytest.raises(ValueError, match="sealed parent score.*digest"):
        materialize_parent_scores(rows, parent, device="cpu")

    assert parent[0].call_count == call_count
    assert cache["parent_inference_contract"] == expected_contract
    assert cache["train_parent_score_content_sha256"] == expected_digest


def test_materialized_parent_contract_and_content_digest_are_exact():
    rows = [
        _joined_row(0, "scene_b", 0.25),
        _joined_row(1, "scene_a", 0.50),
        _joined_row(2, "scene_a", 0.75),
    ]
    parent = _parent()

    scores = materialize_parent_scores(rows, parent, device="cpu")
    cache = parent[0]._geometry_parent_score_cache
    contract = cache["parent_inference_contract"]
    digest = cache["train_parent_score_content_sha256"]

    assert set(contract) == set(PARENT_INFERENCE_CONTRACT_FIELDS)
    assert contract == {
        "schema": "rec-parent-inference-contract",
        "version": 1,
        "device_type": "cpu",
        "device_index": None,
        "local_batch_size": 12,
        "world_size": 1,
        "row_order": "dataset-index-contiguous",
        "remainder_policy": "natural-remainder",
        "feature_source": "bound-base-cache-features",
        "dtype": "float32",
        "autocast": False,
        "allow_tf32": False,
        "eval": True,
        "no_grad": True,
        "score_builder": "normalized-query-reranker-rank-blend",
        "score_builder_version": 1,
        "canonical_query_tie_policy": "score-desc-query-index-asc-v1",
        "content_digest_version": (
            "ordered-identity-raw-float32-sha256-v1"
        ),
        "row_count": len(rows),
        "score_content_sha256": digest,
    }
    expected = hashlib.sha256()
    for row, score in zip(rows, scores):
        base = row["base"]
        scan_bytes = base["scan_id"].encode("utf-8")
        expected.update(base["dataset_index"].to_bytes(8, "little"))
        expected.update(len(scan_bytes).to_bytes(8, "little"))
        expected.update(scan_bytes)
        expected.update(base["target_id"].to_bytes(8, "little"))
        expected.update(score.contiguous().numpy().tobytes(order="C"))
    assert digest == expected.hexdigest()
    assert contract["score_content_sha256"] == digest


def test_materialized_parent_scores_are_reused_across_geometry_batch_sizes():
    rows = [
        _joined_row(index, "scene_{:02d}".format(index // 4), index / 100.0)
        for index in range(25)
    ]
    parent = _parent()
    expected = materialize_parent_scores(rows, parent, device="cpu")
    call_count = parent[0].call_count

    compute_geometry_feature_stats(rows, parent, batch_size=7)
    recovered = []
    for start in range(0, len(rows), 5):
        batch_rows = list(reversed(rows[start:start + 5]))
        batch = build_geometry_training_batch(batch_rows, parent)
        recovered.extend(batch["parent_state"]["compact_scores"].flip(0))

    assert parent[0].call_count == call_count
    assert torch.equal(torch.stack(recovered), expected)


def test_nested_scene_split_is_deterministic_disjoint_and_digest_bound():
    rows = [
        _joined_row(0, "scene_b"),
        _joined_row(1, "scene_a"),
        _joined_row(2, "scene_c"),
        _joined_row(3, "scene_a"),
        _joined_row(4, "scene_d"),
    ]

    fit, calibration = deterministic_scene_split(
        rows, seed=7, calibration_fraction=0.25
    )
    fit_again, calibration_again = deterministic_scene_split(
        rows, seed=7, calibration_fraction=0.25
    )

    fit_scenes = sorted({row["base"]["scan_id"] for row in fit})
    calibration_scenes = sorted({
        row["base"]["scan_id"] for row in calibration
    })
    assert fit == fit_again
    assert calibration == calibration_again
    assert set(fit_scenes).isdisjoint(calibration_scenes)
    assert sorted(fit_scenes + calibration_scenes) == [
        "scene_a", "scene_b", "scene_c", "scene_d"
    ]
    assert [row["base"]["dataset_index"] for row in fit + calibration]

    metadata = build_scene_split_metadata(
        fit, calibration, split_seed=7, calibration_fraction=0.25
    )
    assert metadata["fit_scene_sha256"] == canonical_json_sha256(fit_scenes)
    assert metadata["calibration_scene_sha256"] == canonical_json_sha256(
        calibration_scenes
    )
    assert metadata["mapping_sha256"] == canonical_json_sha256({
        "fit": fit_scenes,
        "calibration": calibration_scenes,
    })
    assert metadata["fit_sample_count"] + metadata[
        "calibration_sample_count"
    ] == len(rows)


def test_batch_uses_frozen_normalized_parent_and_scattered_q_axis_top1():
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b", 0.25)]
    parent = _parent()

    batch = build_geometry_training_batch(rows, parent)

    assert GEOMETRY_INPUT_DIM == 152 + 25 + 2 == 179
    assert batch["features"].shape == (2, 112, 179)
    assert batch["boxes"].shape == (2, 112, 6)
    assert batch["valid_mask"].shape == (2, 112)
    assert batch["candidate_ious"].shape == (2, 112)
    assert batch["feature_names"] == (
        BASE_FEATURE_NAMES
        + GEOMETRY_FEATURE_NAMES
        + ("parent_score", "parent_is_deployed_top1")
    )
    expected_parent_input = (
        torch.stack([row["base"]["features"] for row in rows]) - 1.0
    ) / 2.0
    assert torch.equal(parent[0].last_features, expected_parent_input)
    assert parent[0].saw_grad_enabled is False
    assert parent[0].training is False
    assert all(not parameter.requires_grad for parameter in parent[0].parameters())

    # Compact scores tie at positions 0 and 1. Compact argmax picks query 5,
    # while the deployed Q=256 axis resolves the tie to query 1.
    assert batch["parent_state"]["compact_scores"][0, 0] == batch[
        "parent_state"
    ]["compact_scores"][0, 1]
    assert batch["parent_state"]["compact_scores"][0].argmax().item() == 0
    assert batch["parent_state"]["top1_query_index"].tolist() == [1, 1]
    top1_feature = batch["features"][..., -1].reshape(
        2, CANDIDATE_COUNT, VARIANT_COUNT
    )
    assert top1_feature[:, 0].eq(0.0).all()
    assert top1_feature[:, 1].eq(1.0).all()


def test_geometry_ious_are_labels_only_and_never_change_model_features():
    rows = [_joined_row(0, "scene_a")]
    changed = copy.deepcopy(rows)
    changed[0]["geometry"]["geometry_ious"].copy_(
        1.0 - changed[0]["geometry"]["geometry_ious"]
    )

    original_batch = build_geometry_training_batch(rows, _parent())
    changed_batch = build_geometry_training_batch(changed, _parent())

    assert torch.equal(original_batch["features"], changed_batch["features"])
    assert not torch.equal(
        original_batch["candidate_ious"], changed_batch["candidate_ious"]
    )


def test_frozen_parent_scores_each_base_row_only_once_across_epochs():
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent = _parent()

    first = build_geometry_training_batch(rows, parent)
    second = build_geometry_training_batch(list(reversed(rows)), parent)

    assert parent[0].call_count == 1
    assert torch.equal(
        first["parent_state"]["compact_scores"],
        second["parent_state"]["compact_scores"].flip(0),
    )


def test_feature_stats_stream_fit_evaluator_valid_candidates_only():
    fit_rows = [
        _joined_row(0, "fit_a", 0.0),
        _joined_row(1, "fit_b", 0.5),
    ]
    calibration_row = _joined_row(2, "calibration", 10000.0)
    for row in fit_rows:
        evaluator_valid = row["geometry"]["evaluator_valid"]
        evaluator_valid.zero_()
        evaluator_valid[0, 0] = True
        evaluator_valid[1, 1] = True
        row["geometry"]["geometry_features"][2:].fill_(1e20)

    expected_batch = build_geometry_training_batch(fit_rows, _parent())
    expected_values = expected_batch["features"][
        expected_batch["valid_mask"]
    ].to(torch.float64)
    expected_mean = expected_values.mean(dim=0).float()
    expected_std = expected_values.std(dim=0, unbiased=False).clamp(
        min=1e-6
    ).float()

    mean, std = compute_geometry_feature_stats(
        fit_rows, _parent(), batch_size=1
    )

    assert mean.dtype == torch.float32
    assert std.dtype == torch.float32
    assert torch.equal(mean, expected_mean)
    assert torch.equal(std, expected_std)
    mean_with_calibration, _ = compute_geometry_feature_stats(
        fit_rows + [calibration_row], _parent(), batch_size=2
    )
    assert not torch.equal(mean, mean_with_calibration)


@pytest.mark.parametrize("bad", ["flat", "empty", "identity"])
def test_nested_scene_split_rejects_malformed_joined_rows(bad):
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    if bad == "flat":
        rows[0] = rows[0]["base"]
    elif bad == "empty":
        rows = []
    else:
        rows[0]["geometry"]["scan_id"] = "different"

    with pytest.raises(ValueError):
        deterministic_scene_split(rows, seed=0)


class GeometryFeatureScorer(torch.nn.Module):
    def __init__(self, feature_index=len(BASE_FEATURE_NAMES)):
        super().__init__()
        self.feature_index = feature_index

    def forward(self, features, valid_mask):
        logits = features[..., self.feature_index]
        return {
            "ranking_logits": logits.masked_fill(~valid_mask.bool(), -1e4),
            "threshold_logits": features.new_zeros(
                features.shape[:2] + (2,)
            ),
            "iou_estimate": features.new_zeros(features.shape[:2]),
        }


def _calibration_rows():
    rows = [_joined_row(0, "cal_a"), _joined_row(1, "cal_b")]
    for row in rows:
        geometry = row["geometry"]
        geometry["geometry_features"].zero_()
        geometry["geometry_ious"].zero_()
        # The learned scorer ranks compact position 0, variant 1 first.
        geometry["geometry_features"][0, 1, 0] = 10.0
        # Keep a distinct oracle candidate to prove oracle accounting does not
        # accidentally use the learned selection.
        geometry["geometry_ious"][2, 2] = 0.95
    # Deployed parent selects query 1 (compact position 1), not compact
    # position 0. The first row is fixed and the second is broken at weight 1.
    rows[0]["geometry"]["geometry_ious"][1, 0] = 0.10
    rows[0]["geometry"]["geometry_ious"][0, 1] = 0.90
    rows[1]["geometry"]["geometry_ious"][1, 0] = 0.90
    rows[1]["geometry"]["geometry_ious"][0, 1] = 0.10
    for row in rows:
        row["base"]["candidate_ious"] = row[
            "geometry"
        ]["geometry_ious"][:, 0].clone()
    return rows


def test_geometry_evaluator_uses_exact_q_axis_fallback_and_reports_deltas():
    rows = _calibration_rows()
    metrics = evaluate_geometry_blends(
        GeometryFeatureScorer(),
        rows,
        torch.zeros(GEOMETRY_INPUT_DIM),
        torch.ones(GEOMETRY_INPUT_DIM),
        _parent(),
        geometry_weights=DEFAULT_GEOMETRY_WEIGHTS,
        batch_size=1,
        device="cpu",
    )

    assert tuple(metrics) == DEFAULT_GEOMETRY_WEIGHTS
    baseline = metrics[0.0]
    assert baseline["sample_count"] == 2
    assert baseline["hits025"] == baseline["parent_hits025"] == 1
    assert baseline["hits050"] == baseline["parent_hits050"] == 1
    assert baseline["acc025"] == baseline["parent_acc025"] == 0.5
    assert baseline["acc050"] == baseline["parent_acc050"] == 0.5
    assert baseline["fixes025"] == baseline["breaks025"] == 0
    assert baseline["fixes050"] == baseline["breaks050"] == 0
    # Compact argmax is position 0 and would have produced the opposite hit
    # vector. The exact parent baseline follows scattered query index 1.
    assert baseline["selected_ious"] == pytest.approx((0.10, 0.90))

    learned = metrics[1.0]
    assert learned["selected_ious"] == pytest.approx((0.90, 0.10))
    assert learned["fixes025"] == learned["breaks025"] == 1
    assert learned["fixes050"] == learned["breaks050"] == 1
    assert learned["geometry_oracle_hits025"] == 2
    assert learned["geometry_oracle_hits050"] == 2
    assert learned["geometry_oracle_acc025"] == 1.0
    assert learned["geometry_oracle_acc050"] == 1.0


def test_geometry_evaluator_computes_stable_ranks_only_once_per_batch(
        monkeypatch):
    import models.rec_geometry_reranker as geometry_model_helpers
    import scripts.train_rec_geometry_reranker as trainer

    original = geometry_model_helpers.stable_flat_descending_indices
    calls = []

    def counted(scores, valid):
        calls.append(tuple(scores.shape))
        return original(scores, valid)

    monkeypatch.setattr(
        geometry_model_helpers, "stable_flat_descending_indices", counted
    )
    monkeypatch.setattr(trainer, "stable_flat_descending_indices", counted)

    evaluate_geometry_blends(
        GeometryFeatureScorer(),
        _calibration_rows(),
        torch.zeros(GEOMETRY_INPUT_DIM),
        torch.ones(GEOMETRY_INPUT_DIM),
        _parent(),
        batch_size=2,
        device="cpu",
    )

    assert calls == [(2, 112), (2, 112)]


def _choice_metrics(acc025, acc050):
    return {
        "acc025": float(acc025),
        "acc050": float(acc050),
        "score": calibration_score(acc025, acc050),
    }


def test_geometry_blend_chooser_rejects_regressions_and_ties_to_lower_weight():
    metrics = {
        weight: _choice_metrics(0.50, 0.50)
        for weight in DEFAULT_GEOMETRY_WEIGHTS
    }
    metrics[0.05] = _choice_metrics(0.99, 0.49)  # regress @0.50
    metrics[0.10] = _choice_metrics(0.60, 0.52)
    metrics[0.20] = copy.deepcopy(metrics[0.10])  # exact score tie
    metrics[0.35] = _choice_metrics(0.63, 0.54)
    metrics[0.50] = copy.deepcopy(metrics[0.35])  # exact score tie

    weight, selected = choose_best_geometry_blend(metrics)

    assert calibration_score(0.60, 0.47) == pytest.approx(1.107)
    assert weight == 0.35
    assert selected == metrics[0.35]


def test_geometry_blend_chooser_requires_the_exact_declared_grid():
    metrics = {
        weight: _choice_metrics(0.5, 0.5)
        for weight in DEFAULT_GEOMETRY_WEIGHTS
    }
    del metrics[0.9]
    with pytest.raises(ValueError, match="weight grid"):
        choose_best_geometry_blend(metrics)


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


def _base_manifest(sample_count=4):
    return {
        "cache_schema_version": 1,
        "feature_schema_version": "rec-query-v1",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_epoch": 71,
        "split": "train",
        "candidate_rule": {
            "topk_per_source": 8,
            "max_candidates": CANDIDATE_COUNT,
        },
        "feature_dim": len(BASE_FEATURE_NAMES),
        "feature_names": list(BASE_FEATURE_NAMES),
        "target_iou_policy": "root_only",
        "model_inputs": _model_inputs(),
        "backbone_config": _backbone_config(),
        "sample_count": sample_count,
        "dataset_size": sample_count,
        "source_dataset_size": sample_count,
        "shards": ["shard_000000.pt"],
    }


def _geometry_manifest(sample_count=4):
    base_manifest = _base_manifest(sample_count)
    return {
        "geometry_cache_schema_version": 1,
        "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
        "variant_names": [
            config["name"] for config in DEFAULT_REC_MASK_GEOMETRY_VARIANTS
        ],
        "variant_configs": [
            dict(config) for config in DEFAULT_REC_MASK_GEOMETRY_VARIANTS
        ],
        "regressed_variant_index": 0,
        "min_points": 8,
        "max_point_fraction": 0.5,
        "split": "train",
        "sample_count": sample_count,
        "dataset_size": sample_count,
        "source_dataset_size": sample_count,
        "candidate_rule": copy.deepcopy(base_manifest["candidate_rule"]),
        "target_iou_policy": "root_only",
        "checkpoint_sha256": base_manifest["checkpoint_sha256"],
        "checkpoint_epoch": base_manifest["checkpoint_epoch"],
        "model_inputs": copy.deepcopy(base_manifest["model_inputs"]),
        "backbone_config": copy.deepcopy(base_manifest["backbone_config"]),
        "filter_non_gt_boxes": False,
        "base_cache_binding": {
            "cache_schema_version": 1,
            "feature_schema_version": "rec-query-v1",
            "feature_dim": len(BASE_FEATURE_NAMES),
            "sample_count": sample_count,
            "content_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
        },
        "cache_content_digest": "d" * 64,
        "immutable_metadata_digest": "e" * 64,
    }


def _parent_artifact_path(tmp_path, seed=0):
    torch.manual_seed(seed)
    model = QueryReranker(
        input_dim=len(BASE_FEATURE_NAMES), hidden_dim=8, dropout=0.0
    )
    path = Path(tmp_path) / "parent_{}.pth".format(seed)
    save_reranker_artifact(
        path,
        model,
        torch.zeros(len(BASE_FEATURE_NAMES)),
        torch.ones(len(BASE_FEATURE_NAMES)),
        _base_manifest(),
        epoch=3,
        calibration_metrics={"acc025": 0.5, "acc050": 0.4},
        training_args={"seed": seed},
        reranker_weight=0.9,
    )
    return path


def _scene_metadata():
    rows = [
        _joined_row(0, "fit_a"),
        _joined_row(1, "fit_b"),
        _joined_row(2, "fit_b"),
        _joined_row(3, "cal_a"),
    ]
    return build_scene_split_metadata(
        rows[:3], rows[3:], split_seed=0, calibration_fraction=0.25
    )


def _calibration_metrics():
    return {
        "sample_count": 1,
        "hits025": 1,
        "hits050": 1,
        "parent_hits025": 0,
        "parent_hits050": 0,
        "fixes025": 1,
        "breaks025": 0,
        "fixes050": 1,
        "breaks050": 0,
        "geometry_oracle_hits025": 1,
        "geometry_oracle_hits050": 1,
        "acc025": 1.0,
        "acc050": 1.0,
        "parent_acc025": 0.0,
        "parent_acc050": 0.0,
        "geometry_oracle_acc025": 1.0,
        "geometry_oracle_acc050": 1.0,
        "score": calibration_score(1.0, 1.0),
    }


def _training_args():
    return {
        "split_seed": 0,
        "model_seed": 3,
        "calibration_fraction": 0.25,
        "hidden_dim": 16,
        "dropout": 0.0,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 4,
        "max_epochs": 10,
        "patience": 3,
        "device": "cpu",
        "parent_allow_tf32": False,
        "grad_clip_norm": 1.0,
        "geometry_weight_grid": list(DEFAULT_GEOMETRY_WEIGHTS),
    }


def _geometry_artifact_fixture(tmp_path):
    parent_path = _parent_artifact_path(tmp_path)
    parent = load_parent_reranker_snapshot(parent_path)
    materialize_parent_scores([
        _joined_row(0, "fit_a"),
        _joined_row(1, "fit_b"),
        _joined_row(2, "fit_b"),
        _joined_row(3, "cal_a"),
    ], parent, device="cpu")
    model = QueryReranker(
        input_dim=GEOMETRY_INPUT_DIM, hidden_dim=16, dropout=0.0
    )
    artifact = build_geometry_artifact(
        model,
        torch.zeros(GEOMETRY_INPUT_DIM),
        torch.ones(GEOMETRY_INPUT_DIM),
        parent,
        _base_manifest(),
        _geometry_manifest(),
        _scene_metadata(),
        epoch=4,
        calibration_metrics=_calibration_metrics(),
        training_args=_training_args(),
        geometry_weight=0.35,
    )
    return artifact, model, parent_path, parent


def test_geometry_artifact_build_requires_sealed_parent_materialization(
        tmp_path):
    parent_path = _parent_artifact_path(tmp_path)
    parent = load_parent_reranker_snapshot(parent_path)
    model = QueryReranker(
        input_dim=GEOMETRY_INPUT_DIM, hidden_dim=16, dropout=0.0
    )

    with pytest.raises(ValueError, match="sealed parent.*materialization"):
        build_geometry_artifact(
            model,
            torch.zeros(GEOMETRY_INPUT_DIM),
            torch.ones(GEOMETRY_INPUT_DIM),
            parent,
            _base_manifest(),
            _geometry_manifest(),
            _scene_metadata(),
            epoch=4,
            calibration_metrics=_calibration_metrics(),
            training_args=_training_args(),
            geometry_weight=0.35,
        )


@pytest.mark.parametrize("tamper", [
    "delete-row",
    "extra-row",
    "replace-object",
    "score-inplace",
    "score-dtype",
    "score-shape",
    "score-nonfinite",
])
def test_geometry_artifact_build_revalidates_every_sealed_parent_cache_row(
        tmp_path, tamper):
    _, model, _, parent = _geometry_artifact_fixture(tmp_path)
    cache = parent[0]._geometry_parent_score_cache
    key = cache["ordered_row_object_ids"][0]
    row, score = cache["rows"][key]
    if tamper == "delete-row":
        cache["rows"].pop(key)
    elif tamper == "extra-row":
        extra = _joined_row(4, "extra")["base"]
        cache["rows"][id(extra)] = (
            extra, torch.zeros(CANDIDATE_COUNT, dtype=torch.float32)
        )
    elif tamper == "replace-object":
        cache["rows"][key] = (copy.deepcopy(row), score)
    elif tamper == "score-inplace":
        score.add_(0.125)
    elif tamper == "score-dtype":
        cache["rows"][key] = (row, score.double())
    elif tamper == "score-shape":
        cache["rows"][key] = (row, score[:-1])
    else:
        score[0] = float("nan")

    with pytest.raises(ValueError, match="sealed parent score"):
        build_geometry_artifact(
            model,
            torch.zeros(GEOMETRY_INPUT_DIM),
            torch.ones(GEOMETRY_INPUT_DIM),
            parent,
            _base_manifest(),
            _geometry_manifest(),
            _scene_metadata(),
            epoch=4,
            calibration_metrics=_calibration_metrics(),
            training_args=_training_args(),
            geometry_weight=0.35,
        )


def test_parent_snapshot_hashes_the_same_bytes_it_loads_and_freezes(tmp_path):
    path = _parent_artifact_path(tmp_path)
    model, artifact = load_parent_reranker_snapshot(path)

    assert model._artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert model._artifact_path == str(path.resolve())
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert artifact["feature_names"] == list(BASE_FEATURE_NAMES)


def test_parent_live_state_must_still_match_hashed_artifact(tmp_path):
    path = _parent_artifact_path(tmp_path)
    parent = load_parent_reranker_snapshot(path)
    with torch.no_grad():
        next(parent[0].parameters()).add_(100.0)

    with pytest.raises(ValueError, match="parent.*state"):
        build_geometry_training_batch([_joined_row(0, "scene")], parent)


def test_parent_in_memory_metadata_must_still_match_hashed_artifact(tmp_path):
    path = _parent_artifact_path(tmp_path)
    parent = load_parent_reranker_snapshot(path)
    row = _joined_row(0, "scene")
    build_geometry_training_batch([row], parent)
    parent[1]["feature_mean"][0] += 100.0

    with pytest.raises(ValueError, match="parent.*artifact"):
        build_geometry_training_batch([row], parent)


def test_geometry_artifact_is_exact_train_only_and_binds_all_provenance(
        tmp_path):
    artifact, _, parent_path, parent = _geometry_artifact_fixture(tmp_path)

    assert set(artifact) == set(GEOMETRY_ARTIFACT_FIELDS)
    assert artifact["artifact_version"] == GEOMETRY_ARTIFACT_VERSION
    assert artifact["artifact_version"] == 2
    assert artifact["model_schema_version"] == REC_GEOMETRY_MODEL_SCHEMA_VERSION
    assert artifact["flat_parent_prior_version"] == FLAT_PARENT_PRIOR_VERSION
    assert artifact["input_dim"] == GEOMETRY_INPUT_DIM
    assert artifact["feature_names"] == (
        list(BASE_FEATURE_NAMES)
        + list(GEOMETRY_FEATURE_NAMES)
        + ["parent_score", "parent_is_deployed_top1"]
    )
    assert artifact["parent_artifact_sha256"] == hashlib.sha256(
        parent_path.read_bytes()
    ).hexdigest()
    assert artifact["parent_inference_contract"] == parent[
        0
    ]._geometry_parent_score_cache["parent_inference_contract"]
    assert artifact["train_parent_score_content_sha256"] == parent[
        0
    ]._geometry_parent_score_cache["train_parent_score_content_sha256"]
    assert artifact["parent_inference_contract"][
        "score_content_sha256"
    ] == artifact["train_parent_score_content_sha256"]
    assert artifact["train_base_cache_content_digest"] == "b" * 64
    assert artifact["train_base_cache_manifest_digest"] == "c" * 64
    assert artifact["train_geometry_cache_content_digest"] == "d" * 64
    assert artifact["train_geometry_immutable_metadata_digest"] == "e" * 64
    assert artifact["scene_split"] == _scene_metadata()
    assert tuple(artifact["calibration_metrics"]) == CALIBRATION_METRIC_FIELDS
    assert not any(
        key.lower().startswith("val_") or "validation" in key.lower()
        for key in artifact
    )
    validate_geometry_artifact(
        artifact,
        parent=parent,
        base_manifest=_base_manifest(),
        geometry_manifest=_geometry_manifest(),
        scene_split=_scene_metadata(),
    )


def test_geometry_artifact_rejects_missing_or_tampered_parent_materialization(
        tmp_path):
    artifact, _, _, _ = _geometry_artifact_fixture(tmp_path)
    for field in (
            "parent_inference_contract",
            "train_parent_score_content_sha256"):
        changed = copy.deepcopy(artifact)
        changed.pop(field)
        with pytest.raises(ValueError):
            validate_geometry_artifact(changed)

    mutations = [
        ("schema", "other"),
        ("version", 2),
        ("version", True),
        ("device_type", "tpu"),
        ("device_index", 0),
        ("local_batch_size", 13),
        ("world_size", 2),
        ("world_size", True),
        ("row_order", "shuffled"),
        ("remainder_policy", "drop-last"),
        ("feature_source", "unbound-features"),
        ("dtype", "float16"),
        ("autocast", True),
        ("autocast", 0),
        ("allow_tf32", True),
        ("eval", False),
        ("eval", 1),
        ("no_grad", False),
        ("no_grad", 1),
        ("score_builder", "other"),
        ("score_builder_version", 2),
        ("score_builder_version", True),
        ("canonical_query_tie_policy", "unstable"),
        ("content_digest_version", "other"),
        ("row_count", 3),
        ("score_content_sha256", "f" * 64),
    ]
    for field, value in mutations:
        changed = copy.deepcopy(artifact)
        changed["parent_inference_contract"][field] = value
        with pytest.raises(ValueError, match="parent inference"):
            validate_geometry_artifact(changed)

    changed = copy.deepcopy(artifact)
    changed["train_parent_score_content_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="parent inference"):
        validate_geometry_artifact(changed)


def test_geometry_artifact_binds_cuda_matmul_tf32_outside_contract(tmp_path):
    artifact, _, _, _ = _geometry_artifact_fixture(tmp_path)
    artifact["parent_inference_contract"].update(
        device_type="cuda",
        device_index=0,
        allow_tf32=False,
    )
    artifact["training_args"]["device"] = "cuda:0"
    validate_geometry_artifact(artifact)

    artifact["parent_inference_contract"]["allow_tf32"] = True
    with pytest.raises(ValueError, match="TF32"):
        validate_geometry_artifact(artifact)


@pytest.mark.parametrize("runtime", ["cpu", "tf32-off"])
def test_authoritative_geometry_artifact_requires_cuda0_tf32(
        tmp_path, runtime):
    artifact, _, _, _ = _geometry_artifact_fixture(tmp_path)
    artifact["train_base_cache_content_digest"] = (
        AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256
    )
    if runtime == "tf32-off":
        artifact["parent_inference_contract"].update(
            device_type="cuda",
            device_index=0,
            allow_tf32=False,
        )
        artifact["training_args"].update(
            device="cuda:0",
            parent_allow_tf32=False,
        )

    with pytest.raises(ValueError, match="authoritative.*cuda:0.*TF32"):
        validate_geometry_artifact(artifact)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(artifact_version=999),
        lambda value: value.update(artifact_version=True),
        lambda value: value.update(model_schema_version="other"),
        lambda value: value.update(geometry_cache_schema_version=999),
        lambda value: value.update(geometry_cache_schema_version=True),
        lambda value: value.update(base_cache_schema_version=999),
        lambda value: value.update(base_cache_schema_version=True),
        lambda value: value["feature_names"].__setitem__(0, "wrong"),
        lambda value: value.update(feature_mean=torch.zeros(178)),
        lambda value: value["feature_std"].__setitem__(0, 1e-45),
        lambda value: value["variant_names"].reverse(),
        lambda value: value["variant_configs"][1].update(source="query"),
        lambda value: value.update(regressed_variant_index=1),
        lambda value: value.update(regressed_variant_index=False),
        lambda value: value.update(min_points=0),
        lambda value: value.update(max_point_fraction=2.0),
        lambda value: value.update(checkpoint_sha256="f" * 64),
        lambda value: value.update(checkpoint_epoch=70),
        lambda value: value["model_inputs"].update(butd_gt=True),
        lambda value: value["backbone_config"].update(num_target=128),
        lambda value: value["candidate_rule"].update(max_candidates=8),
        lambda value: value.update(parent_artifact_sha256="f" * 64),
        lambda value: value["parent_provenance"].update(
            feature_mean_sha256="f" * 64
        ),
        lambda value: value.update(flat_parent_prior_version="other"),
        lambda value: value.update(tie_policy="unstable"),
        lambda value: value.update(geometry_weight=0.3),
        lambda value: value.update(train_base_cache_content_digest="f" * 64),
        lambda value: value.update(train_geometry_cache_content_digest="f" * 64),
        lambda value: value.update(target_iou_policy="all_targets"),
        lambda value: value.update(evaluator_filter_policy="gt_filter"),
        lambda value: value["scene_split"].update(mapping_sha256="f" * 64),
        lambda value: value["training_args"].update(batch_size=0),
        lambda value: value["training_args"].update(
            calibration_fraction="0.25"
        ),
        lambda value: value.update(val_metrics={"acc025": 1.0}),
    ],
)
def test_geometry_artifact_rejects_every_contract_mismatch(tmp_path, mutation):
    artifact, _, _, parent = _geometry_artifact_fixture(tmp_path)
    changed = copy.deepcopy(artifact)
    mutation(changed)

    with pytest.raises(ValueError):
        validate_geometry_artifact(
            changed,
            parent=parent,
            base_manifest=_base_manifest(),
            geometry_manifest=_geometry_manifest(),
            scene_split=_scene_metadata(),
        )


def test_geometry_artifact_atomic_round_trip_reproduces_model_outputs(tmp_path):
    artifact, model, parent_path, _ = _geometry_artifact_fixture(tmp_path)
    output = tmp_path / "geometry.pth"
    features = torch.randn(2, 112, GEOMETRY_INPUT_DIM)
    valid = torch.ones(2, 112, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        expected = model(features, valid)["ranking_logits"]

    save_geometry_reranker_artifact(output, artifact)
    restored, loaded = load_geometry_reranker_artifact(
        output,
        parent_artifact_path=parent_path,
        base_manifest=_base_manifest(),
        geometry_manifest=_geometry_manifest(),
    )
    with torch.no_grad():
        actual = restored(features, valid)["ranking_logits"]

    assert torch.equal(actual, expected)
    assert loaded["parent_artifact_sha256"] == artifact[
        "parent_artifact_sha256"
    ]
    assert not output.with_name(output.name + ".tmp").exists()


def test_geometry_artifact_rejects_a_different_actual_parent_file(tmp_path):
    artifact, _, _, _ = _geometry_artifact_fixture(tmp_path)
    output = tmp_path / "geometry.pth"
    save_geometry_reranker_artifact(output, artifact)
    other_parent = _parent_artifact_path(tmp_path, seed=99)

    with pytest.raises(ValueError, match="parent artifact SHA"):
        load_geometry_reranker_artifact(
            output,
            parent_artifact_path=other_parent,
            base_manifest=_base_manifest(),
            geometry_manifest=_geometry_manifest(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["parent_provenance"]["model_config"].update(
            hidden_dim=0
        ),
        lambda value: value["parent_provenance"].update(score_mode="other"),
        lambda value: value["parent_provenance"].update(reranker_weight=2.0),
        lambda value: value["parent_provenance"].update(epoch=-1),
        lambda value: value["parent_provenance"]["feature_names"].__setitem__(
            0, ""
        ),
    ],
)
def test_geometry_artifact_standalone_rejects_malformed_parent_structure(
        tmp_path, mutation):
    artifact, _, _, _ = _geometry_artifact_fixture(tmp_path)
    changed = copy.deepcopy(artifact)
    mutation(changed)

    with pytest.raises(ValueError):
        validate_geometry_artifact(changed)


def test_geometry_artifact_scene_total_must_match_train_manifest(tmp_path):
    artifact, _, _, parent = _geometry_artifact_fixture(tmp_path)
    changed = copy.deepcopy(artifact)
    changed["scene_split"].update(
        sample_count=5,
        fit_sample_count=4,
    )

    with pytest.raises(ValueError, match="sample count"):
        validate_geometry_artifact(
            changed,
            parent=parent,
            base_manifest=_base_manifest(),
            geometry_manifest=_geometry_manifest(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: next(iter(value["model_state_dict"].values())).fill_(
            float("nan")
        ),
        lambda value: value["model_state_dict"].__setitem__(
            next(iter(value["model_state_dict"])),
            next(iter(value["model_state_dict"].values())).double().fill_(
                1e100
            ),
        ),
        lambda value: value["training_args"].update(lr=float("nan")),
        lambda value: value["training_args"].update(
            weight_decay=float("inf")
        ),
        lambda value: value.update(geometry_weight=False),
        lambda value: value["calibration_metrics"].update(
            hits025=0,
            hits050=0,
            parent_hits025=1,
            parent_hits050=1,
            fixes025=0,
            fixes050=0,
            breaks025=1,
            breaks050=1,
            acc025=0.0,
            acc050=0.0,
            parent_acc025=1.0,
            parent_acc050=1.0,
            score=calibration_score(0.0, 0.0),
        ),
        lambda value: value["scene_split"].update(
            sample_count=5,
            calibration_sample_count=2,
        ),
        lambda value: value.update(epoch=0),
    ],
)
def test_geometry_artifact_rejects_nonfinite_or_nonselected_training_state(
        tmp_path, mutation):
    artifact, _, _, _ = _geometry_artifact_fixture(tmp_path)
    changed = copy.deepcopy(artifact)
    mutation(changed)

    with pytest.raises(ValueError):
        validate_geometry_artifact(changed)


def test_atomic_save_validates_temporary_before_replacing_existing_output(
        tmp_path, monkeypatch):
    import scripts.train_rec_geometry_reranker as trainer

    artifact, _, _, _ = _geometry_artifact_fixture(tmp_path)
    output = tmp_path / "existing.pth"
    original = b"preserve-existing-output"
    output.write_bytes(original)

    def reject_reload(*_args, **_kwargs):
        raise ValueError("synthetic strict reload failure")

    monkeypatch.setattr(
        trainer, "load_geometry_reranker_artifact", reject_reload
    )
    with pytest.raises(ValueError, match="synthetic strict reload failure"):
        save_geometry_reranker_artifact(output, artifact)

    assert output.read_bytes() == original
    assert not list(tmp_path.glob("existing.pth.tmp*"))


def test_atomic_save_rejects_final_component_symlink(tmp_path):
    artifact, _, parent_path, _ = _geometry_artifact_fixture(tmp_path)
    output = tmp_path / "geometry-link.pth"
    output.symlink_to(parent_path)
    original_parent = parent_path.read_bytes()

    with pytest.raises(ValueError, match="symlink"):
        save_geometry_reranker_artifact(output, artifact)

    assert output.is_symlink()
    assert parent_path.read_bytes() == original_parent


@pytest.mark.parametrize("case", [
    "impossible-contingency",
    "selected-threshold-order",
    "parent-threshold-order",
    "oracle-threshold-order",
])
def test_calibration_metrics_reject_impossible_aggregate_counts(tmp_path, case):
    artifact, _, _, _ = _geometry_artifact_fixture(tmp_path)
    metrics = artifact["calibration_metrics"]
    if case == "impossible-contingency":
        metrics.update(
            parent_hits025=1,
            parent_hits050=1,
            parent_acc025=1.0,
            parent_acc050=1.0,
            fixes025=1,
            fixes050=1,
            breaks025=1,
            breaks050=1,
        )
    elif case == "selected-threshold-order":
        metrics.update(
            hits025=0,
            acc025=0.0,
            fixes025=0,
            score=calibration_score(0.0, 1.0),
        )
    elif case == "parent-threshold-order":
        metrics.update(
            parent_hits050=1,
            parent_acc050=1.0,
            fixes050=0,
        )
    else:
        metrics.update(
            hits025=0,
            hits050=0,
            acc025=0.0,
            acc050=0.0,
            fixes025=0,
            fixes050=0,
            geometry_oracle_hits025=0,
            geometry_oracle_acc025=0.0,
            score=calibration_score(0.0, 0.0),
        )

    with pytest.raises(ValueError):
        validate_geometry_artifact(artifact)


def test_authoritative_training_bindings_are_explicit_constants():
    assert AUTHORITATIVE_PARENT_ARTIFACT_SHA256 == (
        "f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b"
    )
    assert AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256 == (
        "411ec7d5d80a7be9596de20b348667d529e6a8f568b8ab0c0e0922b8719f9045"
    )


def _zero_parent_artifact_path(tmp_path, sample_count=4):
    model = QueryReranker(
        input_dim=len(BASE_FEATURE_NAMES), hidden_dim=8, dropout=0.0
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    path = Path(tmp_path) / "zero_parent.pth"
    save_reranker_artifact(
        path,
        model,
        torch.zeros(len(BASE_FEATURE_NAMES)),
        torch.ones(len(BASE_FEATURE_NAMES)),
        _base_manifest(sample_count),
        epoch=0,
        calibration_metrics={"acc025": 0.0, "acc050": 0.0},
        training_args={"seed": 0},
        reranker_weight=0.0,
    )
    return path


def _learnable_rows():
    rows = []
    index = 0
    for scene_index in range(4):
        for repeat in range(3):
            row = _joined_row(
                index,
                "learn_scene_{:02d}".format(scene_index),
                feature_offset=repeat / 100.0,
            )
            geometry = row["geometry"]
            geometry["geometry_features"].zero_()
            geometry["geometry_ious"].zero_()
            # Parent default Top-1 is compact position 1. Geometry feature 0
            # alone distinguishes its failing g0 from successful g1.
            geometry["geometry_features"][1, 0, 0] = -1.0
            geometry["geometry_features"][1, 1, 0] = 1.0
            geometry["geometry_ious"][1, 0] = 0.10
            geometry["geometry_ious"][1, 1] = 0.90
            row["base"]["candidate_ious"] = geometry[
                "geometry_ious"
            ][:, 0].clone()
            rows.append(row)
            index += 1
    return rows


def test_short_geometry_training_learns_signal_and_reload_reproduces(tmp_path):
    rows = _learnable_rows()
    parent_path = _zero_parent_artifact_path(tmp_path, sample_count=len(rows))
    parent = load_parent_reranker_snapshot(parent_path)
    materialize_parent_scores(rows, parent, device="cpu")
    fit_rows, calibration_rows = deterministic_scene_split(
        rows, seed=0, calibration_fraction=0.25
    )
    feature_mean, feature_std = compute_geometry_feature_stats(
        fit_rows, parent, batch_size=4
    )
    torch.manual_seed(5)
    model = QueryReranker(
        input_dim=GEOMETRY_INPUT_DIM, hidden_dim=16, dropout=0.0
    )
    output = tmp_path / "learned_geometry.pth"

    artifact = fit_and_save_geometry_model(
        model,
        fit_rows,
        calibration_rows,
        feature_mean,
        feature_std,
        parent,
        _base_manifest(len(rows)),
        _geometry_manifest(len(rows)),
        output,
        split_seed=0,
        model_seed=5,
        calibration_fraction=0.25,
        lr=0.03,
        weight_decay=0.0,
        batch_size=4,
        max_epochs=20,
        patience=6,
        device="cpu",
    )

    metrics = artifact["calibration_metrics"]
    assert metrics["acc025"] > metrics["parent_acc025"]
    assert metrics["acc050"] > metrics["parent_acc050"]
    assert metrics["acc025"] == metrics["acc050"] == 1.0
    assert artifact["geometry_weight"] > 0.0
    restored, loaded = load_geometry_reranker_artifact(
        output,
        parent_artifact_path=parent_path,
        base_manifest=_base_manifest(len(rows)),
        geometry_manifest=_geometry_manifest(len(rows)),
    )
    restored_by_weight = evaluate_geometry_blends(
        restored,
        calibration_rows,
        loaded["feature_mean"],
        loaded["feature_std"],
        parent,
        batch_size=4,
        device="cpu",
    )
    restored_metrics = {
        key: restored_by_weight[loaded["geometry_weight"]][key]
        for key in loaded["calibration_metrics"]
    }
    assert restored_metrics == loaded["calibration_metrics"]


def test_training_data_loader_uses_strict_train_loaders_and_nested_join(
        tmp_path, monkeypatch):
    import scripts.train_rec_geometry_reranker as trainer

    base_rows = [{"base-sentinel": True}]
    geometry_rows = [{"geometry-sentinel": True}]
    base_manifest = _base_manifest()
    geometry_manifest = _geometry_manifest()
    base_path = (tmp_path / "base").resolve()
    geometry_manifest["base_cache_binding"]["path"] = str(base_path)
    binding = copy.deepcopy(geometry_manifest["base_cache_binding"])
    joined = [{"base": base_rows[0], "geometry": geometry_rows[0]}]
    parent_path = _zero_parent_artifact_path(tmp_path)
    calls = []

    def fake_base(path, split):
        calls.append(("base", Path(path).resolve(), split))
        return base_rows, base_manifest, binding

    def fake_geometry(path, split):
        calls.append(("geometry", Path(path).resolve(), split))
        return geometry_rows, geometry_manifest

    def fake_join(base, geometry, base_meta, geometry_meta):
        calls.append(("join", base, geometry, base_meta, geometry_meta))
        return joined

    monkeypatch.setattr(trainer, "load_bound_candidate_cache", fake_base)
    monkeypatch.setattr(trainer, "load_geometry_cache", fake_geometry)
    monkeypatch.setattr(trainer, "join_base_and_geometry_rows", fake_join)

    loaded = load_geometry_training_data(
        base_path, tmp_path / "geometry", parent_path
    )

    assert loaded[0] is joined
    assert loaded[1] is base_manifest
    assert loaded[2] is geometry_manifest
    assert loaded[3][0]._artifact_sha256 == hashlib.sha256(
        parent_path.read_bytes()
    ).hexdigest()
    assert calls[0] == ("base", base_path, "train")
    assert calls[1][0] == "geometry" and calls[1][2] == "train"
    assert calls[2][0] == "join"


def test_public_trainer_runs_scene_split_stats_and_training(tmp_path, monkeypatch):
    import scripts.train_rec_geometry_reranker as trainer

    rows = _learnable_rows()
    parent_path = _zero_parent_artifact_path(tmp_path, sample_count=len(rows))
    parent = load_parent_reranker_snapshot(parent_path)
    monkeypatch.setattr(
        trainer,
        "load_geometry_training_data",
        lambda *_args: (
            rows, _base_manifest(len(rows)), _geometry_manifest(len(rows)), parent
        ),
    )
    output = tmp_path / "public_geometry.pth"

    artifact = train_geometry_reranker(
        tmp_path / "base",
        tmp_path / "geometry",
        parent_path,
        output,
        split_seed=0,
        model_seed=7,
        hidden_dim=16,
        dropout=0.0,
        lr=0.03,
        weight_decay=0.0,
        batch_size=4,
        max_epochs=10,
        patience=4,
        device="cpu",
        calibration_fraction=0.25,
    )

    assert output.is_file()
    assert artifact["calibration_metrics"]["acc025"] == 1.0
    assert artifact["calibration_metrics"]["acc050"] == 1.0
    assert artifact["training_args"]["split_seed"] == 0
    assert artifact["training_args"]["model_seed"] == 7


def test_public_trainer_materializes_parent_before_scene_split(
        tmp_path, monkeypatch):
    import scripts.train_rec_geometry_reranker as trainer

    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent_path = _zero_parent_artifact_path(tmp_path, sample_count=len(rows))
    parent = load_parent_reranker_snapshot(parent_path)
    events = []
    monkeypatch.setattr(
        trainer,
        "load_geometry_training_data",
        lambda *_args: (
            rows, _base_manifest(len(rows)), _geometry_manifest(len(rows)), parent
        ),
    )

    def record_materialization(received, received_parent, device):
        assert received is rows
        assert received_parent is parent
        assert torch.device(device) == torch.device("cpu")
        events.append("materialize")
        return torch.zeros(len(rows), CANDIDATE_COUNT)

    def stop_at_split(*_args, **_kwargs):
        assert events == ["materialize"]
        raise RuntimeError("split-order-sentinel")

    monkeypatch.setattr(trainer, "materialize_parent_scores", record_materialization)
    monkeypatch.setattr(trainer, "deterministic_scene_split", stop_at_split)

    with pytest.raises(RuntimeError, match="split-order-sentinel"):
        train_geometry_reranker(
            tmp_path / "base",
            tmp_path / "geometry",
            parent_path,
            tmp_path / "geometry.pth",
            device="cpu",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_public_trainer_resolves_implicit_cuda_device_index(
        tmp_path, monkeypatch):
    import scripts.train_rec_geometry_reranker as trainer

    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent_path = _zero_parent_artifact_path(tmp_path, sample_count=len(rows))
    parent = load_parent_reranker_snapshot(parent_path)
    monkeypatch.setattr(
        trainer,
        "load_geometry_training_data",
        lambda *_args: (
            rows, _base_manifest(len(rows)), _geometry_manifest(len(rows)), parent
        ),
    )

    def inspect_device(_rows, _parent, device):
        assert device == torch.device("cuda", torch.cuda.current_device())
        raise RuntimeError("resolved-device-sentinel")

    monkeypatch.setattr(trainer, "materialize_parent_scores", inspect_device)
    with pytest.raises(RuntimeError, match="resolved-device-sentinel"):
        train_geometry_reranker(
            tmp_path / "base",
            tmp_path / "geometry",
            parent_path,
            tmp_path / "geometry.pth",
            device="cuda",
        )


@pytest.mark.parametrize("runtime", ["cpu", "tf32-off"])
def test_public_authoritative_training_rejects_runtime_before_materialization(
        tmp_path, monkeypatch, runtime):
    import scripts.train_rec_geometry_reranker as trainer

    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent_path = _zero_parent_artifact_path(tmp_path, sample_count=len(rows))
    parent = load_parent_reranker_snapshot(parent_path)
    geometry_manifest = _geometry_manifest(len(rows))
    geometry_manifest["base_cache_binding"]["content_sha256"] = (
        AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256
    )
    monkeypatch.setattr(
        trainer,
        "load_geometry_training_data",
        lambda *_args: (
            rows, _base_manifest(len(rows)), geometry_manifest, parent
        ),
    )

    def must_not_materialize(*_args, **_kwargs):
        raise AssertionError("authoritative runtime must fail before materialization")

    monkeypatch.setattr(
        trainer, "materialize_parent_scores", must_not_materialize
    )
    device = "cpu"
    if runtime == "tf32-off":
        device = "cuda:0"
        monkeypatch.setattr(
            trainer, "_resolve_device", lambda _device: torch.device("cuda:0")
        )
        monkeypatch.setattr(
            trainer, "_parent_matmul_allow_tf32", lambda _device: False
        )

    with pytest.raises(ValueError, match="authoritative.*cuda:0.*TF32"):
        train_geometry_reranker(
            tmp_path / "base",
            tmp_path / "geometry",
            parent_path,
            tmp_path / "geometry.pth",
            device=device,
        )


def test_public_authoritative_training_rejects_world_size_before_materializer(
        tmp_path, monkeypatch):
    import scripts.train_rec_geometry_reranker as trainer

    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent_path = _zero_parent_artifact_path(tmp_path, sample_count=len(rows))
    parent = load_parent_reranker_snapshot(parent_path)
    geometry_manifest = _geometry_manifest(len(rows))
    geometry_manifest["base_cache_binding"]["content_sha256"] = (
        AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256
    )
    calls = []

    def fake_load(*_args):
        calls.append("load")
        return rows, _base_manifest(len(rows)), geometry_manifest, parent

    def must_not_materialize(*_args, **_kwargs):
        calls.append("materialize")
        raise AssertionError("world-size guard must precede materialization")

    monkeypatch.setattr(trainer, "load_geometry_training_data", fake_load)
    monkeypatch.setattr(
        trainer, "materialize_parent_scores", must_not_materialize
    )
    monkeypatch.setattr(
        trainer, "_resolve_device", lambda _device: torch.device("cuda:0")
    )
    monkeypatch.setattr(
        trainer, "_parent_matmul_allow_tf32", lambda _device: True
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match="world_size=1"):
        train_geometry_reranker(
            tmp_path / "base",
            tmp_path / "geometry",
            parent_path,
            tmp_path / "geometry.pth",
            device="cuda:0",
        )

    assert "materialize" not in calls


@pytest.mark.parametrize("location", ["parent", "parent-hardlink", "base", "geometry"])
def test_public_trainer_rejects_protected_output_before_loading_data(
        tmp_path, monkeypatch, location):
    import os
    import scripts.train_rec_geometry_reranker as trainer

    parent_path = _zero_parent_artifact_path(tmp_path)
    base_cache = tmp_path / "base"
    geometry_cache = tmp_path / "geometry"
    if location == "parent":
        output = parent_path
    elif location == "parent-hardlink":
        output = tmp_path / "parent-hardlink.pth"
        os.link(str(parent_path), str(output))
    elif location == "base":
        output = base_cache / "artifact.pth"
    else:
        output = geometry_cache / "artifact.pth"

    def must_not_load(*_args, **_kwargs):
        raise AssertionError("protected output must fail before cache loading")

    monkeypatch.setattr(trainer, "load_geometry_training_data", must_not_load)
    original_parent = parent_path.read_bytes()
    with pytest.raises(ValueError, match="output"):
        train_geometry_reranker(
            base_cache,
            geometry_cache,
            parent_path,
            output,
            device="cpu",
        )
    assert parent_path.read_bytes() == original_parent


def test_lower_level_fit_rejects_parent_output_alias_before_training(tmp_path):
    rows = _learnable_rows()
    fit_rows, calibration_rows = deterministic_scene_split(
        rows, seed=0, calibration_fraction=0.25
    )
    parent_path = _zero_parent_artifact_path(tmp_path, sample_count=len(rows))
    parent = load_parent_reranker_snapshot(parent_path)
    original_parent = parent_path.read_bytes()
    model = QueryReranker(
        input_dim=GEOMETRY_INPUT_DIM, hidden_dim=16, dropout=0.0
    )

    with pytest.raises(ValueError, match="output"):
        fit_and_save_geometry_model(
            model,
            fit_rows,
            calibration_rows,
            torch.zeros(GEOMETRY_INPUT_DIM),
            torch.ones(GEOMETRY_INPUT_DIM),
            parent,
            _base_manifest(len(rows)),
            _geometry_manifest(len(rows)),
            parent_path,
            split_seed=0,
            model_seed=0,
            calibration_fraction=0.25,
            batch_size=4,
            max_epochs=1,
            patience=1,
            device="cpu",
        )
    assert parent_path.read_bytes() == original_parent


def test_geometry_training_cli_contract(tmp_path):
    args = parse_args([
        "--base-cache", str(tmp_path / "base"),
        "--geometry-cache", str(tmp_path / "geometry"),
        "--parent-artifact", str(tmp_path / "parent.pth"),
        "--output", str(tmp_path / "geometry.pth"),
        "--split-seed", "2",
        "--model-seed", "3",
        "--hidden-dim", "32",
        "--dropout", "0.2",
        "--lr", "0.002",
        "--weight-decay", "0.01",
        "--batch-size", "8",
        "--max-epochs", "12",
        "--patience", "4",
        "--device", "cpu",
    ])

    assert args.base_cache == str(tmp_path / "base")
    assert args.geometry_cache == str(tmp_path / "geometry")
    assert args.parent_artifact == str(tmp_path / "parent.pth")
    assert args.output == str(tmp_path / "geometry.pth")
    assert args.split_seed == 2
    assert args.model_seed == 3
    assert args.hidden_dim == 32
    assert args.dropout == 0.2
    assert args.lr == 0.002
    assert args.weight_decay == 0.01
    assert args.batch_size == 8
    assert args.max_epochs == 12
    assert args.patience == 4
    assert args.device == "cpu"
