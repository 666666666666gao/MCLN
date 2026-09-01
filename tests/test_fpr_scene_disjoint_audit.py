from types import SimpleNamespace
import hashlib

import pytest
import torch
import main_utils

from main_utils import (
    _FPR_SCENE_DISJOINT_FIXED_CONFIG,
    _canonical_fpr_scene_disjoint_config_receipt,
    _load_checkpoint_payload,
    build_parent_relative_text_verifier_audit_diagnostics,
    build_fpr_scene_disjoint_config_receipt,
    capture_fpr_audit_model_state,
    capture_fpr_audit_output_state,
    fpr_scene_sample_identity_digest,
)
from models.parent_relative_text_verifier import (
    compute_parent_relative_text_verifier_loss,
)
from test_parent_relative_text_verifier import (
    _fixture,
    _force_positive_predictions,
)
from train_dist_mod import (
    FPR_SCENE_DISJOINT_SPLITS,
    build_fpr_scene_disjoint_dataset_views,
    fpr_scene_disjoint_fold,
)


def test_exact_decision_counts_include_fix_break_and_kept_rows():
    module, batch, _, _ = _fixture()
    _force_positive_predictions(module)
    output = module(batch)
    output["selected_position"] = torch.tensor([1, 1])
    output["switch_mask"] = torch.tensor([True, True])
    candidate_ious = torch.tensor([
        [0.10, 0.60, 0.20, 0.05],
        [0.60, 0.40, 0.15, 0.05],
    ])

    result = compute_parent_relative_text_verifier_loss(
        output, candidate_ious
    )
    stat_dict = {
        "parent_relative_text_verifier_{}".format(key): value.item()
        for key, value in result["stats"].items()
        if key.startswith("audit_")
    }
    diagnostics = build_parent_relative_text_verifier_audit_diagnostics(
        stat_dict, expected_sample_count=2
    )

    assert diagnostics["switch_count"] == 2
    assert diagnostics["thresholds"]["025"] == {
        "threshold": 0.25,
        "fix_count": 1,
        "break_count": 0,
        "kept_correct_count": 1,
        "kept_wrong_count": 0,
        "parent_hits": 1,
        "selected_hits": 2,
        "parent_accuracy": 0.5,
        "selected_accuracy": 1.0,
        "net_hits": 1,
        "fix_per_switch": 0.5,
        "transition_precision": 1.0,
    }
    assert diagnostics["thresholds"]["050"]["fix_count"] == 1
    assert diagnostics["thresholds"]["050"]["break_count"] == 1
    assert diagnostics["thresholds"]["050"]["net_hits"] == 0


def test_audit_diagnostics_reject_non_partitioning_counts():
    counts = {
        "audit_sample_count": 2,
        "audit_switch_count": 1,
        "audit_fix025_count": 1,
        "audit_break025_count": 1,
        "audit_kept_correct025_count": 1,
        "audit_kept_wrong025_count": 0,
        "audit_fix050_count": 0,
        "audit_break050_count": 0,
        "audit_kept_correct050_count": 1,
        "audit_kept_wrong050_count": 1,
    }
    stats = {
        "parent_relative_text_verifier_{}".format(key): value
        for key, value in counts.items()
    }
    with pytest.raises(ValueError, match="do not partition"):
        build_parent_relative_text_verifier_audit_diagnostics(stats, 2)


def _fixed_config_args():
    values = {
        name: expected
        for name, expected in _FPR_SCENE_DISJOINT_FIXED_CONFIG
    }
    values.update({
        "dataset": ["nr3d"],
        "test_dataset": "nr3d",
        "legacy_scene_graph_cache": "",
    })
    return SimpleNamespace(**values)


def test_all_folds_share_one_strict_fpr_config_sha(monkeypatch):
    first_args = _fixed_config_args()
    first_args.fpr_scene_disjoint_fold = 0
    first_args.expected_eval_sample_count = 7129
    second_args = _fixed_config_args()
    second_args.fpr_scene_disjoint_fold = 4
    second_args.expected_eval_sample_count = 5915
    preregistered = _canonical_fpr_scene_disjoint_config_receipt(
        first_args
    )["sha256"]
    monkeypatch.setattr(
        main_utils, "FPR_SCENE_DISJOINT_CONFIG_SHA256", preregistered
    )
    first = build_fpr_scene_disjoint_config_receipt(first_args)
    second = build_fpr_scene_disjoint_config_receipt(second_args)
    assert first == second
    assert len(first["sha256"]) == 64

    drifted = _fixed_config_args()
    drifted.parent_relative_text_verifier_top_k = 4
    with pytest.raises(ValueError, match="top_k drifted"):
        build_fpr_scene_disjoint_config_receipt(drifted)

    unlisted_drift = _fixed_config_args()
    unlisted_drift.some_future_behavior_flag = True
    with pytest.raises(ValueError, match="configuration SHA-256 drifted"):
        build_fpr_scene_disjoint_config_receipt(unlisted_drift)


def test_legacy_scene_audit_hash_ignores_false_counterfactual_default():
    args = _fixed_config_args()
    without_new_default = _canonical_fpr_scene_disjoint_config_receipt(
        args
    )["sha256"]
    args.parent_relative_text_verifier_counterfactual_training = False
    with_new_default = _canonical_fpr_scene_disjoint_config_receipt(args)[
        "sha256"
    ]

    assert with_new_default == without_new_default


def test_legacy_scene_audit_rejects_counterfactual_training():
    args = _fixed_config_args()
    args.parent_relative_text_verifier_counterfactual_training = True

    with pytest.raises(ValueError, match="forbid counterfactual Parent"):
        _canonical_fpr_scene_disjoint_config_receipt(args)


def test_scene_sample_identity_digest_rejects_duplicate_coverage():
    assert fpr_scene_sample_identity_digest([2, 0, 1]) == (
        fpr_scene_sample_identity_digest([0, 1, 2])
    )
    with pytest.raises(ValueError, match="must be unique"):
        fpr_scene_sample_identity_digest([0, 1, 1])


def test_scene_audit_checkpoint_load_binds_same_fd_sha_and_epoch(
        monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "protected-e57.pth"
    torch.save({"epoch": 57, "model": {}}, str(checkpoint_path))
    raw = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(
        main_utils, "FPR_SCENE_DISJOINT_E57_SHA256", checkpoint_sha256
    )
    args = SimpleNamespace(
        fpr_scene_disjoint_audit=True,
        restore_e57_lr_to_initial=False,
        fpr_scene_disjoint_checkpoint_sha256=checkpoint_sha256,
        checkpoint_path=str(checkpoint_path),
    )

    checkpoint = _load_checkpoint_payload(args)

    assert checkpoint["epoch"] == 57
    assert args.fpr_scene_disjoint_consumed_checkpoint_sha256 == (
        checkpoint_sha256
    )
    assert args.fpr_scene_disjoint_consumed_checkpoint_epoch == 57


def _scene_ids_for_fold(fold, count):
    result = []
    candidate = 0
    while len(result) < count:
        scene_id = "audit_scene_{:d}_{:05d}".format(fold, candidate)
        if fpr_scene_disjoint_fold(scene_id) == fold:
            result.append(scene_id)
        candidate += 1
    return result


def _synthetic_nr3d_train_dataset():
    annotations = []
    for fold, counts in FPR_SCENE_DISJOINT_SPLITS.items():
        scene_ids = _scene_ids_for_fold(fold, counts["holdout_scenes"])
        per_scene = [1] * len(scene_ids)
        per_scene[0] += counts["holdout_samples"] - len(scene_ids)
        for scene_id, sample_count in zip(scene_ids, per_scene):
            annotations.extend([
                {"dataset": "nr3d", "scan_id": scene_id}
                for _ in range(sample_count)
            ])
    class SyntheticDataset(SimpleNamespace):
        def __len__(self):
            return len(self.annos)

        def __getitem__(self, index):
            return {"source_row": index}

    return SyntheticDataset(
        annos=annotations, split="train", dataset_dict={"nr3d": 1},
        joint_det=False, augment_det=False, augment=True, overfit=False,
    )


def test_scene_disjoint_views_are_complete_exact_and_non_augmented_holdout():
    base = _synthetic_nr3d_train_dataset()
    train, holdout, metadata = build_fpr_scene_disjoint_dataset_views(
        base, 0, dict(FPR_SCENE_DISJOINT_SPLITS[0])
    )

    assert len(train.annos) == 25790
    assert len(holdout.annos) == 7129
    assert metadata["fit_scenes"] == 402
    assert metadata["holdout_scenes"] == 109
    assert train.augment is True
    assert holdout.augment is False
    assert not train.joint_det and not holdout.joint_det
    assert not train.augment_det and not holdout.augment_det
    first_item = train[0]
    assert first_item["source_row"] == 0
    assert first_item["fpr_scene_audit_sample_index"] == 7129
    train_scenes = {record["scan_id"] for record in train.annos}
    holdout_scenes = {record["scan_id"] for record in holdout.annos}
    assert not train_scenes.intersection(holdout_scenes)


class _AuditStateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.structured_slot_builder = torch.nn.Linear(2, 2)
        self.frozen_backbone = torch.nn.Linear(2, 2)


def test_state_and_output_digests_detect_only_intended_changes():
    model = _AuditStateModel()
    before = capture_fpr_audit_model_state(model)
    with torch.no_grad():
        model.structured_slot_builder.weight.add_(1.0)
    after = capture_fpr_audit_model_state(model)

    assert before["frozen"]["sha256"] == after["frozen"]["sha256"]
    assert before["trainable"]["sha256"] != after["trainable"]["sha256"]

    output = {
        "last_center": torch.zeros(1, 2, 3),
        "last_pred_size": torch.ones(1, 2, 3),
        "last_pred_masks": [torch.zeros(2, 4)],
        "sp_last_pred_masks": [torch.ones(2, 4)],
        "adaptive_weights": [torch.full((2, 4), 0.5)],
        "parent_relative_text_verifier_parent_scores": torch.tensor([
            [0.9, 0.8]
        ]),
    }
    first = capture_fpr_audit_output_state(output)
    second = capture_fpr_audit_output_state(output)
    assert first == second
    output["last_center"][0, 0, 0] = 1.0
    third = capture_fpr_audit_output_state(output)
    assert first["combined_sha256"] != third["combined_sha256"]
