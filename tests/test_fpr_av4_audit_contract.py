from __future__ import print_function

import hashlib
import json
import pathlib
import re
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
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
