from __future__ import print_function

import hashlib
import json
import pathlib
import re
import subprocess
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_ONLY_DATA_MANIFEST = (
    ROOT / "scripts" / "nr3d_fpr_tv_av4_train_only_data_manifest_v2.json"
)
LAUNCHER = ROOT / "scripts" / (
    "run_nr3d_fpr_tv_counterfactual_parent_audit.sh"
)
EXECUTOR = ROOT / "scripts" / "mcln_fpr_tv_av4_audit_static_exec.x86_64"
BUILD_RECEIPT = ROOT / "scripts" / (
    "mcln_fpr_tv_av4_audit_static_exec.build_receipt"
)
SPEC = ROOT / "FPR_TV_COUNTERFACTUAL_PARENT_AUDIT_SPEC_2026-09-01.md"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _train_args(source):
    block = re.search(r"\ntrain_args=\(\n(.*?)\n\)\nfull_command=", source, re.S)
    assert block is not None
    return block.group(1)


def test_static_entry_build_receipt_matches_reviewed_binary():
    receipt = json.loads(BUILD_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["schema"] == (
        "mcln-fpr-tv-av4-counterfactual-parent-static-build-v1"
    )
    assert receipt["artifact_mode"] == "0755"
    assert receipt["artifact_sha256"] == _sha256(EXECUTOR)
    assert receipt["artifact_size"] == EXECUTOR.stat().st_size
    assert receipt["trust_root"] == "/root/mcln_fpr_av4_audit_trust/v1"
    assert receipt["shared_gpu_lock"] == (
        "/root/autodl-tmp/mcln_v99_backbone_gpu0.lock"
    )
    program_headers = subprocess.check_output(
        ["readelf", "-lW", str(EXECUTOR)], text=True
    )
    assert "INTERP" not in program_headers


def test_launcher_is_exact_bounded_counterfactual_parent_audit():
    source = LAUNCHER.read_text(encoding="utf-8")
    args = _train_args(source)
    required = (
        "--dataset \"${DATASET}\" --test_dataset \"${DATASET}\"",
        "--joint_det --butd_cls",
        "--max_train_batches \"${AUDIT_BATCHES}\"",
        "--gradient_accumulation_steps 1",
        "--start_epoch \"${AUDIT_EPOCH}\" --max_epoch \"${AUDIT_EPOCH}\"",
        "--use_source_choice_selector --eval_use_selector_choice_scores",
        "--use_parent_relative_text_verifier",
        "--parent_relative_text_verifier_train_only",
        "--parent_relative_text_verifier_counterfactual_training",
        "--parent_relative_text_verifier_top_k 5",
        "--parent_relative_text_verifier_max_candidates 10",
        "--parent_relative_text_verifier_loss_weight 1.0",
    )
    for fragment in required:
        assert fragment in args
    forbidden = (
        "--eval ",
        "--density_aware_target_box_loss_weight",
        "--use_sacr_source",
        "--use_sacr_score_refiner",
        "--relation_counterfactual_aux",
        "--fpr_scene_disjoint_audit",
    )
    for fragment in forbidden:
        assert fragment not in args
    assert 'readonly AUDIT_BATCHES=100' in source
    assert 'readonly BATCH_SIZE=16' in source
    assert 'long_training_authorized": False' in source
    assert 'exit 20' in source
    assert "-name 'eval_metrics*.json'" in source
    assert "-name '*.pth'" in source


def test_main_bounded_path_records_exact_training_and_sentinels():
    source = (ROOT / "main_utils.py").read_text(encoding="utf-8")
    required = (
        'max_train_batches != 100',
        'args.batch_size != 16',
        'args.start_epoch != 58',
        '"optimizer_step_count"',
        '"sample_count"',
        '"state_integrity"',
        '"output_integrity"',
        '"formal_validation_accessed": False',
        '"long_training_authorized": False',
    )
    for fragment in required:
        assert fragment in source


def test_spec_permanently_excludes_unrelated_routes_and_long_training():
    source = SPEC.read_text(encoding="utf-8")
    for fragment in (
        "fair baseline reproduction",
        "Section/Experiment 7",
        "Section/Experiment 8 or E0--E7 matrix",
        "long_training_authorized=false",
        "strictly above 60.0%",
        "strictly above 68.9%",
    ):
        assert fragment in source


def test_train_only_data_manifest_excludes_validation_sources():
    manifest = json.loads(TRAIN_ONLY_DATA_MANIFEST.read_text())
    assert manifest["schema"] == (
        "mcln-nr3d-fpr-tv-av4-train-only-data-manifest-v2"
    )
    assert "val_v3scans.pkl" not in manifest["sources"]
    assert "superpoints/val" not in manifest["sources"]
    paths = [row["path"] for row in manifest["files"]]
    assert "val_v3scans.pkl" not in paths
    assert not any(path.startswith("superpoints/val/") for path in paths)
    assert manifest["file_count"] == len(paths)
    assert manifest["total_size"] == sum(
        row["size"] for row in manifest["files"]
    )


def test_bounded_counterfactual_loader_does_not_build_test_loader(monkeypatch):
    import main_utils

    class DummyDataset(object):
        pass

    class DummyTrainer(main_utils.BaseTrainTester):
        @staticmethod
        def get_datasets(_args):
            return DummyDataset(), None

    constructed = []
    monkeypatch.setattr(
        main_utils,
        "DistributedSampler",
        lambda dataset, shuffle=True: (dataset, shuffle),
    )
    monkeypatch.setattr(
        main_utils,
        "DataLoader",
        lambda dataset, **_kwargs: constructed.append(dataset) or dataset,
    )
    args = types.SimpleNamespace(
        num_workers=0,
        dataloader_prefetch_factor=2,
        eval=False,
        hard_example_replay_manifest="",
        hard_example_replay_manifest_sha256="",
        batch_size=16,
        rng_seed=0,
        persistent_train_workers=False,
        fpr_scene_disjoint_audit=False,
        parent_relative_text_verifier_counterfactual_training=True,
        max_train_batches=100,
    )
    trainer = object.__new__(DummyTrainer)
    train_loader, test_loader = trainer.get_loaders(args)
    assert len(constructed) == 1
    assert train_loader is constructed[0]
    assert test_loader is None


def test_bounded_counterfactual_main_logging_accepts_missing_test_loader():
    import main_utils

    args = types.SimpleNamespace(
        parent_relative_text_verifier_counterfactual_training=True,
        max_train_batches=100,
    )
    assert main_utils._optional_test_dataset_size(args, None) is None

    args.parent_relative_text_verifier_counterfactual_training = False
    try:
        main_utils._optional_test_dataset_size(args, None)
    except ValueError as error:
        assert "bounded train-only audit" in str(error)
    else:
        raise AssertionError("non-audit path accepted a missing test loader")


def test_bounded_counterfactual_dataset_skips_validation_construction(
        monkeypatch):
    import train_dist_mod

    calls = []

    class DummyDataset(object):
        def __init__(self, **kwargs):
            calls.append(kwargs["split"])

    monkeypatch.setattr(train_dist_mod, "Joint3DDataset", DummyDataset)
    args = types.SimpleNamespace(
        dataset=["nr3d"],
        test_dataset="nr3d",
        joint_det=True,
        density_aware_target_box_scene_disjoint_audit=False,
        fpr_scene_disjoint_audit=False,
        eval=False,
        debug=False,
        eval_train=False,
        use_color=True,
        use_height=False,
        data_root="/data",
        detect_intermediate=True,
        use_multiview=False,
        butd=False,
        butd_gt=False,
        butd_cls=True,
        augment_det=False,
        skip_missing_superpoints=True,
        use_sacr_source=False,
        use_sacr_score_refiner=False,
        use_parent_relative_text_verifier=True,
        legacy_scene_graph_cache="",
        legacy_scene_graph_cache_strict=False,
        legacy_scene_graph_cache_expected_target_selection="",
        legacy_scene_graph_cache_expected_sha256="",
        parent_relative_text_verifier_counterfactual_training=True,
        max_train_batches=100,
    )
    train_dataset, test_dataset = train_dist_mod.TrainTester.get_datasets(args)
    assert calls == ["train"]
    assert train_dataset is not None
    assert test_dataset is None
