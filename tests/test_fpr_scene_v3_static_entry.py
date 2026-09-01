import hashlib
import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_LAUNCHER = (
    _ROOT / "scripts" / "run_nr3d_fpr_tv_scene_disjoint_v3_fold3.sh"
)
_STATIC_SOURCE = _ROOT / "scripts" / "mcln_fpr_audit_static_exec.c"
_STATIC_EXEC = (
    _ROOT / "scripts" / "mcln_fpr_tv_scene_v3_fold3_static_exec.x86_64"
)
_BUILD_RECEIPT = (
    _ROOT / "scripts" / "mcln_fpr_tv_scene_v3_fold3_static_exec.build_receipt"
)
_STATIC_EXEC_SHA256 = (
    "b42c9d3461c56b2d63c7671a5b91ad1412d10e0c97d24d9876b794bc7a20e22c"
)
_STATIC_SOURCE_SHA256 = (
    "0bf6cfcfb015a91474579ba0c0f186c49c6a38695601d904d3216724cc67dcdc"
)
_TRUST_ROOT = "/root/mcln_fpr_scene_v3_trust/v1"
_SHARED_GPU_LOCK = "/root/autodl-tmp/mcln_v99_backbone_gpu0.lock"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v3_static_executor_has_frozen_build_provenance():
    receipt = json.loads(_BUILD_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["schema"] == "mcln-fpr-tv-scene-v3-fold3-static-build-v1"
    assert receipt["artifact_sha256"] == _STATIC_EXEC_SHA256
    assert receipt["artifact_size"] == _STATIC_EXEC.stat().st_size
    assert receipt["artifact_mode"] == "0755"
    assert _sha256(_STATIC_EXEC) == _STATIC_EXEC_SHA256
    assert receipt["reviewed_source_sha256"] == _STATIC_SOURCE_SHA256
    assert _sha256(_STATIC_SOURCE) == _STATIC_SOURCE_SHA256
    assert receipt["trust_root"] == _TRUST_ROOT
    assert receipt["shared_gpu_lock"] == _SHARED_GPU_LOCK
    assert receipt["elf_static"] is True
    assert receipt["has_interp_segment"] is False
    assert receipt["trusted_launcher_deployment_name"] == (
        "run_nr3d_fpr_tv_density_audit.sh"
    )


def test_v3_launcher_requires_static_identity_and_uses_parent_gpu_lock():
    launcher = _LAUNCHER.read_text(encoding="utf-8")
    assert 'readonly TRUST_ROOT="{}"'.format(_TRUST_ROOT) in launcher
    assert _STATIC_EXEC_SHA256 in launcher
    assert _STATIC_SOURCE_SHA256 in launcher
    assert '[[ "${MCLN_FPR_TRUSTED_CLEAN_ENV:-}" == "1" ]]' in launcher
    assert 'readonly consumed_launcher_fd="/proc/$$/fd/3"' in launcher
    assert 'readonly FOLD="3"' in launcher
    assert 'if (($# != 0)); then' in launcher
    assert "flock -n" not in launcher
    assert '"trusted_execution": {' in launcher
    assert '"static_executor_sha256": static_exec_sha256' in launcher
    assert '"shared_gpu_lock": shared_gpu_lock' in launcher
