import hashlib
import importlib.util
import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_RUNNER = _ROOT / "scripts" / "run_density_target_box_scene_audit.py"
_LAUNCHER = (
    _ROOT / "scripts" / "run_nr3d_density_aware_target_box_scene_audit.sh"
)
_MANIFEST = (
    _ROOT / "scripts" / "density_target_box_scene_runtime_manifest_v1.json"
)
_STATIC_EXEC = (
    _ROOT / "scripts" / "mcln_density_target_box_scene_audit_static_exec.x86_64"
)
_STATIC_BUILD_RECEIPT = (
    _ROOT / "scripts" / "mcln_density_target_box_scene_audit_static_exec.build_receipt"
)
_MANIFEST_SHA256 = (
    "04977c404fb759722d56e8bbeadb383a7113f4cec8e6d7dbde24d35f3f48c354"
)
_STATIC_EXEC_SHA256 = (
    "d63392f280a6563e6cd8439a44aa5da8eb68c59d71c7a5574aa2763915e02775"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "density_scene_runner_under_test", str(_RUNNER)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _value(arguments, option):
    index = arguments.index(option)
    return arguments[index + 1]


def test_scene_runtime_manifest_and_launcher_pins_are_closed():
    assert _sha256(_MANIFEST) == _MANIFEST_SHA256
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema"] == (
        "mcln-density-aware-target-box-scene-reviewed-runtime-v1"
    )
    assert manifest["file_count"] == 371
    assert len(manifest["files"]) == 371
    assert manifest["total_size"] == sum(
        record["size"] for record in manifest["files"].values()
    )
    for relative in (
            "main_utils.py", "train_dist_mod.py", "models/losses.py",
            "models/density_aware_target_box_audit.py",
            "scripts/decide_density_target_box_scene_audit.py",
            "scripts/run_density_target_box_scene_audit.py",
            "DENSITY_AWARE_TARGET_BOX_SCENE_AUDIT_SPEC_2026-09-01.md"):
        record = manifest["files"][relative]
        path = _ROOT / relative
        assert path.stat().st_size == record["size"]
        assert _sha256(path) == record["sha256"]
    launcher = _LAUNCHER.read_text(encoding="utf-8")
    assert _MANIFEST_SHA256 in launcher
    assert manifest["files"][
        "scripts/run_density_target_box_scene_audit.py"
    ]["sha256"] in launcher


def test_scene_static_executor_has_frozen_build_provenance():
    receipt = json.loads(_STATIC_BUILD_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["schema"] == (
        "mcln-density-target-box-scene-static-build-v1"
    )
    assert receipt["artifact_sha256"] == _STATIC_EXEC_SHA256
    assert receipt["artifact_size"] == _STATIC_EXEC.stat().st_size
    assert receipt["artifact_mode"] == "0755"
    assert _sha256(_STATIC_EXEC) == _STATIC_EXEC_SHA256
    assert receipt["reviewed_source_sha256"] == _sha256(
        _ROOT / receipt["source_path"]
    )
    assert receipt["trust_root"] == "/root/mcln_density_scene_audit_trust/v1"
    assert receipt["shared_gpu_lock"] == (
        "/root/autodl-tmp/mcln_v99_backbone_gpu0.lock"
    )
    assert receipt["elf_static"] is True
    assert receipt["has_interp_segment"] is False
    launcher = _LAUNCHER.read_text(encoding="utf-8")
    assert _STATIC_EXEC_SHA256 in launcher


def test_scene_roles_share_contract_except_frozen_weight_and_budget():
    runner = _load_runner()
    code = Path("/reviewed/code")
    inputs = Path("/reviewed/inputs")
    output = Path("/runtime/role")
    parent = runner._common_train_args(code, inputs, output, "parent")
    control = runner._common_train_args(code, inputs, output, "control")
    method = runner._common_train_args(code, inputs, output, "method")

    for arguments in (parent, control, method):
        assert "--eval" not in arguments
        assert "--checkpoint_metric_retention" not in arguments
        assert _value(arguments, "--batch_size") == "16"
        assert _value(arguments, "--gradient_accumulation_steps") == "1"
        assert _value(arguments, "--max_epoch") == "58"
        assert _value(arguments, "--expected_eval_sample_count") == "6329"
        assert _value(
            arguments, "--density_aware_target_box_scene_disjoint_fold"
        ) == "2"
        assert _value(arguments, "--checkpoint_path") == str(
            inputs / "protected_e57.pth"
        )

    assert _value(parent, "--max_train_batches") == "0"
    assert _value(control, "--max_train_batches") == "100"
    assert _value(method, "--max_train_batches") == "100"
    assert _value(
        parent, "--density_aware_target_box_loss_weight"
    ) == "0.0"
    assert _value(
        control, "--density_aware_target_box_loss_weight"
    ) == "0.0"
    assert _value(
        method, "--density_aware_target_box_loss_weight"
    ) == "1.0"
