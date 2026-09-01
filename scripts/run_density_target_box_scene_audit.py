"""One-shot formal runner for the density target-box scene audit.

This file is consumed only through the reviewed static launcher.  It creates
one immutable code/input snapshot, runs parent/control/method serially, and
publishes an audit-only paired decision.  It never launches formal validation
or saves a checkpoint.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile


SCHEMA = "mcln-density-target-box-scene-formal-runner-v1"
SOURCE_ROOT = pathlib.Path(
    "/root/autodl-tmp/mcln_density_target_box_scene_review_20260901"
)
DATA_ROOT = pathlib.Path("/root/autodl-tmp/DATA_ROOT")
OUTPUT_ROOT = (
    DATA_ROOT / "output/network_v99_baseline_gt/nr3d"
)
PYTHON_BIN = pathlib.Path("/root/miniconda3/envs/bdetr/bin/python")
SOURCE_CHECKPOINT = (
    OUTPUT_ROOT
    / "control/tier_hard_query_e57_e58_e62_patience2"
    / "startup_recovery_v2/formal_input_snapshot_v2/protected_e57.pth"
)
SOURCE_CHECKPOINT_SHA256 = (
    "fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
)
GROUPFREE_CHECKPOINT = DATA_ROOT / "gf_detector_l6o256.pth"
GROUPFREE_SHA256 = (
    "9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
)
DATA_MANIFEST = (
    OUTPUT_ROOT / "control/fpr_tv_audit/nr3d_train_input_manifest_v1.json"
)
DATA_MANIFEST_SHA256 = (
    "ce0e287856363fce2c6cb119617798ff98470aaba331d8239fbc32ffcdc93259"
)
RUNTIME_MANIFEST = (
    SOURCE_ROOT / "scripts/density_target_box_scene_runtime_manifest_v1.json"
)
# Replaced only after the reviewed runtime manifest is generated.
RUNTIME_MANIFEST_SHA256 = os.environ.get(
    "MCLN_DENSITY_SCENE_RUNTIME_MANIFEST_SHA256", ""
)
SNAPSHOT_EXECUTOR_RELATIVE = "scripts/mcln_density_audit_snapshot_exec.py"
SNAPSHOT_EXECUTOR_SHA256 = (
    "839b6d8479b94e610288723219b0149203dde89ab0a85a6dd3bd9d4776d04c88"
)
DECISION_SCRIPT_RELATIVE = "scripts/decide_density_target_box_scene_audit.py"
RUNNER_RELATIVE = "scripts/run_density_target_box_scene_audit.py"
SPEC_RELATIVE = "DENSITY_AWARE_TARGET_BOX_SCENE_AUDIT_SPEC_2026-09-01.md"
EXP = "nr3d_v99_density_target_box_scene_fold2_e57_e58_b100_pair"
AUDIT_ROOT = OUTPUT_ROOT / "audit" / (EXP + "_one_shot")
LOCK_FILE = pathlib.Path("/root/autodl-tmp/mcln_v99_backbone_gpu0.lock")
SNAPSHOT_OWNER_UID = 65532
SNAPSHOT_OWNER_GID = 65532
MIN_FREE_GIB = 6
EXPECTED_SOURCES = [
    "train_v3scans.pkl",
    "val_v3scans.pkl",
    "refer_it_3d/nr3d.csv",
    "roberta-base",
    "superpoints/train",
    "superpoints/val",
    "group_free_pred_bboxes/group_free_pred_bboxes_train",
]
ROLE_PORTS = {"parent": 5527, "control": 5528, "method": 5529}


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _read_opened_regular(path, expected_sha=None, expected_mode=None):
    path = pathlib.Path(path)
    if path.is_symlink():
        raise ValueError("symlink input rejected: {}".format(path))
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("input is not regular: {}".format(path))
        chunks = []
        while True:
            chunk = os.read(descriptor, 16 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
            before.st_dev, before.st_ino, before.st_size,
            stat.S_IMODE(before.st_mode)
    ) != (
            after.st_dev, after.st_ino, after.st_size,
            stat.S_IMODE(after.st_mode)
    ):
        raise ValueError("input changed while reading: {}".format(path))
    payload = b"".join(chunks)
    digest = _sha256_bytes(payload)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("SHA-256 mismatch for {}".format(path))
    if expected_mode is not None and (
            format(stat.S_IMODE(after.st_mode), "04o") != expected_mode):
        raise ValueError("mode mismatch for {}".format(path))
    return payload, digest, after


def _json_from_raw(raw, label):
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("invalid {} JSON: {}".format(label, exc))
    if not isinstance(value, dict):
        raise ValueError("{} must be a JSON object".format(label))
    return value


def _atomic_json_no_overwrite(path, payload):
    path = pathlib.Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(str(path))
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, str(path))
        directory_descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checked_relative(value):
    if not isinstance(value, str) or not value:
        raise ValueError("runtime path must be a non-empty string")
    normalized = os.path.normpath(value).replace(os.sep, "/")
    if (
            normalized != value or os.path.isabs(value)
            or value == ".." or value.startswith("../")):
        raise ValueError("unsafe runtime path: {}".format(value))
    return value


def _require_real_descendant(root, relative):
    root = pathlib.Path(root).resolve()
    current = root
    for component in relative.split("/"):
        current = current / component
        information = os.lstat(str(current))
        if stat.S_ISLNK(information.st_mode):
            raise ValueError("symlink component rejected: {}".format(current))
    resolved = current.resolve()
    if os.path.commonpath([str(root), str(resolved)]) != str(root):
        raise ValueError("path escaped root: {}".format(current))
    return current


def _load_runtime_manifest():
    raw, digest, _info = _read_opened_regular(
        RUNTIME_MANIFEST, RUNTIME_MANIFEST_SHA256, "0644"
    )
    manifest = _json_from_raw(raw, "runtime manifest")
    if manifest.get("schema") != (
            "mcln-density-aware-target-box-scene-reviewed-runtime-v1"):
        raise ValueError("runtime manifest schema drifted")
    if manifest.get("source_root") != str(SOURCE_ROOT.resolve()):
        raise ValueError("runtime manifest source_root drifted")
    records = manifest.get("files")
    if not isinstance(records, dict) or len(records) != manifest.get(
            "file_count"):
        raise ValueError("runtime manifest record count drifted")
    if manifest.get("total_size") != sum(
            record.get("size", -1) for record in records.values()):
        raise ValueError("runtime manifest total size drifted")
    for required in (
            "train_dist_mod.py", "main_utils.py", "models/losses.py",
            "models/density_aware_target_box.py",
            "models/density_aware_target_box_audit.py",
            SNAPSHOT_EXECUTOR_RELATIVE, DECISION_SCRIPT_RELATIVE,
            RUNNER_RELATIVE, SPEC_RELATIVE):
        if required not in records:
            raise ValueError("runtime manifest omits {}".format(required))
    if records[SNAPSHOT_EXECUTOR_RELATIVE].get(
            "sha256") != SNAPSHOT_EXECUTOR_SHA256:
        raise ValueError("snapshot executor identity drifted")
    return manifest, raw, digest


def _verify_source_runtime(manifest):
    root = SOURCE_ROOT.resolve()
    for relative, record in sorted(manifest["files"].items()):
        relative = _checked_relative(relative)
        source = _require_real_descendant(root, relative)
        raw, _digest, information = _read_opened_regular(
            source, record.get("sha256"), record.get("mode")
        )
        if len(raw) != record.get("size") or information.st_size != len(raw):
            raise ValueError("runtime size mismatch: {}".format(relative))


def _copy_runtime_snapshot(manifest, destination):
    destination = pathlib.Path(destination)
    if not destination.is_dir() or list(destination.iterdir()):
        raise ValueError("code snapshot destination must be an empty directory")
    root = SOURCE_ROOT.resolve()
    for relative, record in sorted(manifest["files"].items()):
        relative = _checked_relative(relative)
        source = _require_real_descendant(root, relative)
        raw, _digest, _information = _read_opened_regular(
            source, record.get("sha256"), record.get("mode")
        )
        target = destination / pathlib.Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chown(str(target), SNAPSHOT_OWNER_UID, SNAPSHOT_OWNER_GID)
        os.chmod(str(target), 0o444)
    for current, directories, _files in os.walk(str(destination), topdown=False):
        for name in directories:
            directory = pathlib.Path(current) / name
            os.chown(str(directory), SNAPSHOT_OWNER_UID, SNAPSHOT_OWNER_GID)
            os.chmod(str(directory), 0o555)
    os.chown(str(destination), SNAPSHOT_OWNER_UID, SNAPSHOT_OWNER_GID)
    os.chmod(str(destination), 0o555)


def _verify_runtime_snapshot(manifest, root):
    root = pathlib.Path(root)
    observed = set()
    for current, directories, files in os.walk(str(root)):
        current_path = pathlib.Path(current)
        current_info = os.lstat(str(current_path))
        if (
                current_info.st_uid != SNAPSHOT_OWNER_UID
                or current_info.st_gid != SNAPSHOT_OWNER_GID
                or stat.S_IMODE(current_info.st_mode) != 0o555):
            raise ValueError("snapshot directory metadata drifted")
        for name in directories:
            if (current_path / name).is_symlink():
                raise ValueError("snapshot directory symlink rejected")
        for name in files:
            path = current_path / name
            if path.is_symlink():
                raise ValueError("snapshot file symlink rejected")
            relative = str(path.relative_to(root)).replace(os.sep, "/")
            observed.add(relative)
    if observed != set(manifest["files"]):
        raise ValueError("snapshot file inventory drifted")
    for relative, record in sorted(manifest["files"].items()):
        raw, _digest, information = _read_opened_regular(
            root / pathlib.Path(relative), record.get("sha256"), "0444"
        )
        if (
                len(raw) != record.get("size")
                or information.st_uid != SNAPSHOT_OWNER_UID
                or information.st_gid != SNAPSHOT_OWNER_GID):
            raise ValueError("snapshot file metadata drifted: {}".format(
                relative
            ))


def _verify_dataset_manifest():
    root = DATA_ROOT.resolve()
    raw, _digest, _info = _read_opened_regular(
        DATA_MANIFEST, DATA_MANIFEST_SHA256
    )
    manifest = _json_from_raw(raw, "data manifest")
    if manifest.get("schema") != "mcln-nr3d-fpr-tv-audit-data-manifest-v1":
        raise ValueError("data manifest schema drifted")
    if manifest.get("data_root") != str(root):
        raise ValueError("data manifest root drifted")
    if manifest.get("sources") != EXPECTED_SOURCES:
        raise ValueError("data manifest source closure drifted")
    paths = []
    for relative_source in EXPECTED_SOURCES:
        source = _require_real_descendant(root, relative_source)
        if source.is_dir():
            for current, directories, files in os.walk(str(source)):
                directories.sort()
                files.sort()
                current_path = pathlib.Path(current)
                for name in directories:
                    if (current_path / name).is_symlink():
                        raise ValueError("dataset directory symlink rejected")
                paths.extend(current_path / name for name in files)
        elif source.is_file():
            paths.append(source)
        else:
            raise ValueError("dataset input is missing: {}".format(source))
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("data manifest files are missing")
    observed_names = [str(path.relative_to(root)) for path in paths]
    if [row.get("path") for row in rows] != observed_names:
        raise ValueError("data file inventory drifted")
    if len(rows) != manifest.get("file_count"):
        raise ValueError("data manifest file count drifted")
    if manifest.get("total_size") != sum(
            row.get("size", -1) for row in rows):
        raise ValueError("data manifest total size drifted")
    for path, row in zip(paths, rows):
        raw_file, _digest, information = _read_opened_regular(
            path, row.get("sha256")
        )
        if (
                len(raw_file) != row.get("size")
                or stat.S_IMODE(information.st_mode) != row.get("mode")):
            raise ValueError("dataset metadata drifted: {}".format(path))
    return raw


def _verify_checkpoint_contract():
    import torch

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(SOURCE_CHECKPOINT), flags)
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != SOURCE_CHECKPOINT_SHA256:
            raise ValueError("protected E57 SHA drifted before load")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            checkpoint = torch.load(handle, map_location="cpu")
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            second.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if second.hexdigest() != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("protected E57 changed during load")
    if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev, after.st_ino, after.st_size):
        raise ValueError("protected E57 inode changed during load")
    config = checkpoint.get("config")
    config = vars(config) if hasattr(config, "__dict__") else dict(config or {})
    optimizer = checkpoint.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    expected_lrs = [1e-5, 1e-4, 1e-5, 1.25e-5]
    actual_lrs = [group.get("lr") for group in groups]
    if checkpoint.get("epoch") != 57:
        raise ValueError("protected checkpoint is not E57")
    if config.get("batch_size") != 16:
        raise ValueError("protected checkpoint is not B16")
    if config.get("gradient_accumulation_steps", 1) != 1:
        raise ValueError("protected checkpoint is not accumulation one")
    if config.get("joint_det") is not True or config.get("butd_cls") is not True:
        raise ValueError("protected checkpoint is not joint_det+butd_cls")
    if config.get("use_source_choice_selector") is not True:
        raise ValueError("protected checkpoint is not V99 selector")
    if config.get("source_choice_selector_sources") != (
            "default,default_rank_blend_contrastive010"):
        raise ValueError("protected V99 sources drifted")
    if len(groups) != 4 or len(optimizer.get("state", {})) != 716:
        raise ValueError("protected optimizer topology drifted")
    for actual, expected in zip(actual_lrs, expected_lrs):
        if actual is None or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("protected current learning rate drifted")


def _copy_independent_input(source, destination, expected_sha):
    raw, _digest, source_info = _read_opened_regular(source, expected_sha)
    destination = pathlib.Path(destination)
    descriptor = os.open(
        str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chown(str(destination), SNAPSHOT_OWNER_UID, SNAPSHOT_OWNER_GID)
    os.chmod(str(destination), 0o444)
    _raw, _digest, destination_info = _read_opened_regular(
        destination, expected_sha, "0444"
    )
    if (source_info.st_dev, source_info.st_ino) == (
            destination_info.st_dev, destination_info.st_ino):
        raise ValueError("input snapshot did not receive an independent inode")
    if (
            destination_info.st_uid != SNAPSHOT_OWNER_UID
            or destination_info.st_gid != SNAPSHOT_OWNER_GID):
        raise ValueError("input snapshot owner drifted")


def _resource_check():
    usage = shutil.disk_usage(str(DATA_ROOT))
    if usage.free < MIN_FREE_GIB * 1024 ** 3:
        raise RuntimeError("fewer than {} GiB remain".format(MIN_FREE_GIB))
    result = subprocess.run(
        [
            "/usr/bin/nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits", "-i", "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    used = int(result.stdout.strip())
    if used >= 500:
        raise RuntimeError("GPU0 is busy ({} MiB)".format(used))
    if not LOCK_FILE.is_file() or LOCK_FILE.is_symlink():
        raise RuntimeError("shared GPU lock path is missing or unsafe")


def _verify_static_environment():
    required = {
        "MCLN_FPR_TRUSTED_CLEAN_ENV": "1",
        "MCLN_FPR_LAUNCHER_FD": "3",
    }
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise RuntimeError("static trust environment {} drifted".format(name))
    for name in (
            "MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256",
            "MCLN_FPR_STATIC_EXEC_SHA256", "MCLN_FPR_STATIC_SOURCE_SHA256",
            "MCLN_DENSITY_SCENE_RUNTIME_MANIFEST_SHA256"):
        value = os.environ.get(name, "")
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise RuntimeError("static trust digest {} is invalid".format(name))


def _common_train_args(code_snapshot, input_root, role_root, role):
    weight = "1.0" if role == "method" else "0.0"
    max_batches = "0" if role == "parent" else "100"
    return [
        "--num_target", "256", "--sampling", "kps",
        "--num_encoder_layers", "3", "--num_decoder_layers", "6",
        "--self_position_embedding", "loc_learned",
        "--query_points_obj_topk", "4", "--use_color",
        "--weight_decay", "0.0005", "--data_root", str(DATA_ROOT) + "/",
        "--val_freq", "1", "--batch_size", "16", "--num_workers", "4",
        "--dataloader_prefetch_factor", "2", "--persistent_train_workers",
        "--save_freq", "1", "--print_freq", "20", "--clip_norm", "0.1",
        "--rng_seed", "0", "--lr_backbone", "1e-3", "--lr", "1e-4",
        "--lr_decay_epochs", "150", "--warmup-epoch", "-1",
        "--dataset", "nr3d", "--test_dataset", "nr3d",
        "--joint_det", "--butd_cls",
        "--density_aware_target_box_loss_weight", weight,
        "--density_aware_target_box_checkpoint_sha256",
        SOURCE_CHECKPOINT_SHA256,
        "--density_aware_target_box_scene_disjoint_audit",
        "--density_aware_target_box_scene_disjoint_role", role,
        "--density_aware_target_box_scene_disjoint_fold", "2",
        "--density_aware_target_box_scene_disjoint_expected_fit_scenes", "408",
        "--density_aware_target_box_scene_disjoint_expected_holdout_scenes", "103",
        "--density_aware_target_box_scene_disjoint_expected_fit_samples", "26590",
        "--density_aware_target_box_scene_disjoint_expected_holdout_samples", "6329",
        "--max_train_batches", max_batches,
        "--gradient_accumulation_steps", "1", "--local_rank", "0",
        "--detect_intermediate", "--use_soft_token_loss",
        "--use_contrastive_align", "--log_dir", str(role_root),
        "--pp_checkpoint", str(input_root / "gf_detector_l6o256.pth"),
        "--pp_checkpoint_sha256", GROUPFREE_SHA256,
        "--self_attend", "--skip_missing_superpoints",
        "--checkpoint_path", str(input_root / "protected_e57.pth"),
        "--resume_lr_scale", "1.0", "--start_epoch", "1",
        "--max_epoch", "58", "--model", "MCLN",
        "--exp", EXP + "_" + role,
        "--use_source_choice_selector", "--eval_use_selector_choice_scores",
        "--source_choice_selector_sources",
        "default,default_rank_blend_contrastive010",
        "--source_choice_selector_default_source", "default",
        "--source_choice_selector_hidden_dim", "288",
        "--source_choice_selector_lr", "1.25e-4",
        "--source_choice_selector_loss_weight", "0.5",
        "--source_choice_selector_choice_target",
        "precision_gain_default_sourcewise_focal_bce",
        "--source_choice_selector_min_iou_gap", "0.03",
        "--expected_eval_sample_count", "6329",
    ]


def _role_command(code_snapshot, input_root, runtime_output, role):
    role_root = runtime_output / role
    role_home = runtime_output / (role + "_home")
    role_root.mkdir()
    role_home.mkdir()
    for name in ("hf", "xdg", "torch"):
        (role_home / name).mkdir()
    executor = code_snapshot / SNAPSHOT_EXECUTOR_RELATIVE
    train_entry = code_snapshot / "train_dist_mod.py"
    train_args = _common_train_args(
        code_snapshot, input_root, role_root, role
    )
    command = [
        str(PYTHON_BIN), "-I", "-S", str(executor),
        "--code-root", str(code_snapshot),
        "--write-root", str(role_root),
        "--allow-write", str(role_home), "--",
        str(PYTHON_BIN), str(train_entry),
    ] + train_args
    environment = {
        "HOME": str(role_home), "USER": "root", "LOGNAME": "root",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PATH": "/root/miniconda3/envs/bdetr/bin:/usr/local/cuda/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "{}:{}".format(code_snapshot, code_snapshot / "pointnet2"),
        "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": str(role_home / "hf"),
        "TRANSFORMERS_CACHE": str(role_home / "hf"),
        "XDG_CACHE_HOME": str(role_home / "xdg"),
        "TORCH_HOME": str(role_home / "torch"),
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1", "LOCAL_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(ROLE_PORTS[role]),
    }
    return role_root, command, environment


def _run_role(code_snapshot, input_root, runtime_output, role):
    role_root, command, environment = _role_command(
        code_snapshot, input_root, runtime_output, role
    )
    command_path = runtime_output / (role + "_command.json")
    _atomic_json_no_overwrite(command_path, {
        "schema": "mcln-density-target-box-scene-command-v1",
        "role": role,
        "argv": command,
        "environment": environment,
    })
    log_path = role_root / "formal_stdout_stderr.log"
    with open(str(log_path), "xb") as log_handle:
        result = subprocess.run(
            command,
            cwd=str(code_snapshot),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        log_handle.flush()
        os.fsync(log_handle.fileno())
    if result.returncode != 0:
        raise RuntimeError("{} role exited {}".format(role, result.returncode))
    expected_epoch = 57 if role == "parent" else 58
    expected_name = (
        "density_target_box_scene_audit_{}_epoch_{}.json".format(
            role, expected_epoch
        )
    )
    receipts = [path for path in role_root.rglob(expected_name) if path.is_file()]
    if len(receipts) != 1:
        raise RuntimeError("{} role produced {} receipts".format(
            role, len(receipts)
        ))
    generated = [path for path in role_root.rglob("*.pth") if path.is_file()]
    if generated:
        raise RuntimeError("{} role generated weights".format(role))
    raw, digest, _info = _read_opened_regular(receipts[0])
    payload = _json_from_raw(raw, role + " receipt")
    if payload.get("role") != role:
        raise RuntimeError("{} receipt role drifted".format(role))
    os.chmod(str(receipts[0]), 0o444)
    return {
        "path": str(receipts[0]), "sha256": digest,
        "command_path": str(command_path),
        "command_sha256": _read_opened_regular(command_path)[1],
    }


def _run_snapshot_verify(code_snapshot, runtime_output, runtime_home):
    probe_root = runtime_output / "snapshot_verify"
    probe_root.mkdir()
    command = [
        str(PYTHON_BIN), "-I", "-S",
        str(code_snapshot / SNAPSHOT_EXECUTOR_RELATIVE),
        "--code-root", str(code_snapshot),
        "--write-root", str(probe_root),
        "--allow-write", str(runtime_home), "--verify-only",
    ]
    environment = {
        "HOME": str(runtime_home), "USER": "root", "LOGNAME": "root",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PATH": "/root/miniconda3/envs/bdetr/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    }
    subprocess.run(
        command, cwd=str(code_snapshot), env=environment,
        stdin=subprocess.DEVNULL, check=True, close_fds=True,
    )


def _create_pre_provenance(audit_root, code_snapshot, input_root, manifest,
                           manifest_digest, records):
    runner_record = manifest["files"][RUNNER_RELATIVE]
    payload = {
        "schema": "mcln-density-target-box-scene-provenance-v1",
        "audit_only": True,
        "long_training_authorized": False,
        "audit_root": str(audit_root),
        "code_snapshot": str(code_snapshot),
        "input_snapshot": str(input_root),
        "runtime_manifest_sha256": manifest_digest,
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "groupfree_sha256": GROUPFREE_SHA256,
        "runner_sha256": runner_record["sha256"],
        "launcher_sha256": os.environ[
            "MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256"
        ],
        "static_executor_sha256": os.environ["MCLN_FPR_STATIC_EXEC_SHA256"],
        "static_source_sha256": os.environ["MCLN_FPR_STATIC_SOURCE_SHA256"],
        "roles": records,
    }
    path = audit_root / "pre_audit_provenance.json"
    _atomic_json_no_overwrite(path, payload)
    return path


def _publish_decision(code_snapshot, audit_root, receipt_records,
                      provenance_path):
    decision_path = audit_root / "paired_decision.json"
    command = [
        str(PYTHON_BIN), "-I", "-S",
        str(code_snapshot / DECISION_SCRIPT_RELATIVE),
        "--parent", receipt_records["parent"]["path"],
        "--control", receipt_records["control"]["path"],
        "--method", receipt_records["method"]["path"],
        "--provenance", str(provenance_path),
        "--output", str(decision_path),
    ]
    environment = {
        "HOME": str(audit_root), "USER": "root", "LOGNAME": "root",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PATH": "/root/miniconda3/envs/bdetr/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    }
    subprocess.run(
        command, cwd=str(code_snapshot), env=environment,
        stdin=subprocess.DEVNULL, check=True, close_fds=True,
    )
    raw, digest, _info = _read_opened_regular(decision_path, expected_mode="0444")
    decision = _json_from_raw(raw, "paired decision")
    if decision.get("long_training_authorized") is not False:
        raise RuntimeError("paired decision authorized long training")
    return decision_path, digest, decision


def _preflight(manifest):
    _verify_source_runtime(manifest)
    _verify_dataset_manifest()
    _verify_checkpoint_contract()
    _read_opened_regular(GROUPFREE_CHECKPOINT, GROUPFREE_SHA256)
    _resource_check()
    if AUDIT_ROOT.exists() or AUDIT_ROOT.is_symlink():
        raise RuntimeError("one-shot scene audit root is already consumed")
    print("scene_audit_preflight=pass audit_only=true long_training_authorized=false")


def _backbone(manifest, manifest_raw, manifest_digest):
    _preflight(manifest)
    AUDIT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(str(AUDIT_ROOT), 0o700)
    code_snapshot = AUDIT_ROOT / "code_snapshot"
    input_root = AUDIT_ROOT / "input_snapshot"
    runtime_output = AUDIT_ROOT / "runtime_output"
    runtime_home = AUDIT_ROOT / "runtime_home"
    for path in (code_snapshot, input_root, runtime_output, runtime_home):
        os.mkdir(str(path), 0o700)
    for name in ("hf", "xdg", "torch"):
        (runtime_home / name).mkdir()

    _copy_runtime_snapshot(manifest, code_snapshot)
    _verify_runtime_snapshot(manifest, code_snapshot)
    _copy_independent_input(
        SOURCE_CHECKPOINT, input_root / "protected_e57.pth",
        SOURCE_CHECKPOINT_SHA256,
    )
    _copy_independent_input(
        GROUPFREE_CHECKPOINT, input_root / "gf_detector_l6o256.pth",
        GROUPFREE_SHA256,
    )
    _copy_independent_input(
        DATA_MANIFEST, input_root / DATA_MANIFEST.name,
        DATA_MANIFEST_SHA256,
    )
    manifest_snapshot = input_root / RUNTIME_MANIFEST.name
    descriptor = os.open(
        str(manifest_snapshot), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    try:
        offset = 0
        while offset < len(manifest_raw):
            offset += os.write(descriptor, manifest_raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chown(str(manifest_snapshot), SNAPSHOT_OWNER_UID, SNAPSHOT_OWNER_GID)
    os.chmod(str(manifest_snapshot), 0o444)
    os.chown(str(input_root), SNAPSHOT_OWNER_UID, SNAPSHOT_OWNER_GID)
    os.chmod(str(input_root), 0o555)
    _run_snapshot_verify(code_snapshot, runtime_output, runtime_home)

    receipt_records = {}
    for role in ("parent", "control", "method"):
        receipt_records[role] = _run_role(
            code_snapshot, input_root, runtime_output, role
        )
        _verify_runtime_snapshot(manifest, code_snapshot)
        _read_opened_regular(
            input_root / "protected_e57.pth", SOURCE_CHECKPOINT_SHA256, "0444"
        )
        _read_opened_regular(
            input_root / "gf_detector_l6o256.pth", GROUPFREE_SHA256, "0444"
        )

    generated_weights = [
        path for path in runtime_output.rglob("*.pth") if path.is_file()
    ]
    if generated_weights:
        raise RuntimeError("scene audit generated checkpoint files")
    _verify_source_runtime(manifest)
    _verify_dataset_manifest()
    _verify_checkpoint_contract()
    _read_opened_regular(GROUPFREE_CHECKPOINT, GROUPFREE_SHA256)
    _verify_runtime_snapshot(manifest, code_snapshot)

    provenance_path = _create_pre_provenance(
        AUDIT_ROOT, code_snapshot, input_root, manifest, manifest_digest,
        receipt_records,
    )
    decision_path, decision_sha, decision = _publish_decision(
        code_snapshot, AUDIT_ROOT, receipt_records, provenance_path
    )
    final = {
        "schema": SCHEMA,
        "audit_root": str(AUDIT_ROOT),
        "density_gate_passed": bool(decision.get("density_gate_passed")),
        "decision": str(decision_path),
        "decision_sha256": decision_sha,
        "audit_only": True,
        "long_training_authorized": False,
        "generated_weight_count": 0,
        "formal_validation_sample_count": 0,
    }
    _atomic_json_no_overwrite(AUDIT_ROOT / "formal_completion.json", final)
    print(json.dumps(final, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "backbone"), required=True)
    args = parser.parse_args()
    _verify_static_environment()
    manifest, manifest_raw, manifest_digest = _load_runtime_manifest()
    if args.mode == "preflight":
        _preflight(manifest)
    else:
        _backbone(manifest, manifest_raw, manifest_digest)


if __name__ == "__main__":
    main()
