#!/usr/bin/env python3
from __future__ import print_function

import hashlib
import json
import os
import pathlib
import re
import stat
import sys


SCHEMA = "mcln-fpr-tv-av4-failed-attempt-evidence-v1"
EXPECTED_FAILURE_STAGE = (
    "first_batch_post_backward_pre_optimizer_step_gradient_audit"
)
RUNTIME_MANIFEST_RELATIVE_PATH = (
    "input_snapshot/fpr_tv_counterfactual_parent_runtime_manifest_v2.json"
)


def _read_regular(path, expected_sha256, expected_size=None,
                  expected_mode=None, expected_owner=None):
    path = pathlib.Path(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("evidence is not a regular file: " + str(path))
        chunks = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise ValueError("evidence changed while reading: " + str(path))
    if digest.hexdigest() != expected_sha256:
        raise ValueError("evidence SHA changed: " + str(path))
    if expected_size is not None and before.st_size != int(expected_size):
        raise ValueError("evidence size changed: " + str(path))
    if expected_mode is not None:
        actual_mode = "{:04o}".format(stat.S_IMODE(before.st_mode))
        if actual_mode != expected_mode:
            raise ValueError("evidence mode changed: " + str(path))
    if expected_owner is not None:
        actual_owner = "{}:{}".format(before.st_uid, before.st_gid)
        if actual_owner != expected_owner:
            raise ValueError("evidence owner changed: " + str(path))
    return b"".join(chunks)


def _load_json(raw, label):
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("invalid {} JSON: {}".format(label, error))
    if not isinstance(result, dict):
        raise ValueError(label + " must be a JSON object")
    return result


def _walk_regular_files(root, skipped_top_level=()):
    root = pathlib.Path(root)
    files = {}
    directories = set()
    for current, directory_names, file_names in os.walk(str(root)):
        current_path = pathlib.Path(current)
        relative_directory = current_path.relative_to(root).as_posix()
        relative_directory = (
            "." if relative_directory == "." else relative_directory
        )
        if relative_directory == ".":
            directory_names[:] = [
                name for name in directory_names
                if name not in set(skipped_top_level)
            ]
        directories.add(relative_directory)
        for directory_name in directory_names:
            candidate = current_path / directory_name
            if candidate.is_symlink():
                raise ValueError("evidence contains a directory symlink")
        for file_name in file_names:
            candidate = current_path / file_name
            if candidate.is_symlink():
                raise ValueError("evidence contains a file symlink")
            relative = candidate.relative_to(root).as_posix()
            files[relative] = candidate
    return files, directories


def _verify_code_snapshot(root, runtime_manifest_raw):
    root = pathlib.Path(root)
    manifest = _load_json(runtime_manifest_raw, "runtime manifest")
    records = manifest.get("files")
    if (
            manifest.get("schema")
            != "mcln-fpr-tv-counterfactual-parent-reviewed-runtime-v2"
            or not isinstance(records, dict)
            or manifest.get("file_count") != len(records)
            or manifest.get("total_size")
            != sum(record.get("size", -1) for record in records.values())):
        raise ValueError("runtime manifest contract changed")
    files, _ = _walk_regular_files(root)
    if set(files) != set(records):
        raise ValueError("failed code-snapshot file set changed")
    for relative, record in records.items():
        if not isinstance(record, dict):
            raise ValueError("invalid runtime record: " + relative)
        _read_regular(
            files[relative],
            record.get("sha256"),
            expected_size=record.get("size"),
            expected_mode="0444",
            expected_owner="65532:65532",
        )
    return records


def _verify_zero_step_control_flow(main_raw, launch_raw):
    main_text = main_raw.decode("utf-8")
    launch_text = launch_raw.decode("utf-8")
    method_start = main_text.index("    def train_one_epoch(")
    method_end = main_text.index("    # BRIEF eval", method_start)
    method_text = main_text[method_start:method_end]
    backward_position = method_text.index(").backward()")
    error_position = method_text.index(
        '"actual Parent score-gradient audit is missing"'
    )
    optimizer_position = method_text.index("optimizer.step()")
    if not backward_position < error_position < optimizer_position:
        raise ValueError("failed main no longer proves pre-optimizer failure")
    if "retain_grad()" in method_text[:error_position]:
        raise ValueError("failed main unexpectedly retained score gradients")
    required_log_fragments = (
        "  0%|          | 0/2806 [00:10<?, ?it/s]",
        "Traceback (most recent call last):",
        "main_utils.py\", line 7534, in train_one_epoch",
        "actual Parent score-gradient audit is missing",
        "ValueError: actual Parent score-gradient audit is missing",
    )
    for fragment in required_log_fragments:
        if fragment not in launch_text:
            raise ValueError("failed launch log lacks marker: " + fragment)
    if re.search(r"(?<![0-9])(?:[1-9][0-9]*)/2806", launch_text):
        raise ValueError("failed launch contains positive progress")
    for forbidden in (
            "Train: [58]", "train_audit_receipt_epoch_58.json",
            "bounded_audit_receipt=validated", "optimizer_step_count"):
        if forbidden in launch_text:
            raise ValueError("failed launch unexpectedly progressed: " + forbidden)


def _verify_failed_runtime_selection(config, command_text):
    """Bind the frozen failure to the verifier-only counterfactual branch."""
    if not isinstance(config, dict):
        raise ValueError("failed config must be an object")
    required_config = {
        "use_parent_relative_text_verifier": True,
        "parent_relative_text_verifier_train_only": True,
        "parent_relative_text_verifier_counterfactual_training": True,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise ValueError("failed verifier-only config changed: " + key)
    if not isinstance(command_text, str):
        raise ValueError("failed command must be text")
    for fragment in (
            "--use_parent_relative_text_verifier",
            "--parent_relative_text_verifier_train_only",
            "--parent_relative_text_verifier_counterfactual_training"):
        if fragment not in command_text.split():
            raise ValueError(
                "failed verifier-only command changed: " + fragment
            )


def _verify_train_mode_selected_before_forward(main_text):
    """Prove the verifier-only mode setter ran before sentinel and train forward."""
    train_start = main_text.index("    def train_one_epoch(")
    train_end = main_text.index("    # BRIEF eval", train_start)
    train_text = main_text[train_start:train_end]
    mode_call = "self._set_source_moe_train_mode(model, args)"
    call_positions = [
        match.start() for match in re.finditer(re.escape(mode_call), train_text)
    ]
    sentinel_position = train_text.index(
        "self._capture_fpr_audit_sentinel("
    )
    forward_position = train_text.index("end_points = model(inputs)")
    if (
            len(call_positions) < 2
            or not call_positions[0] < sentinel_position
            or not sentinel_position < call_positions[1] < forward_position):
        raise ValueError(
            "failed verifier-only train mode was not selected before forward"
        )


def _verify_first_batch_failure_is_input_independent(
        main_raw, mcln_raw, verifier_raw):
    """Prove the frozen bug must fail on the first yielded training batch."""
    main_text = main_raw.decode("utf-8")
    mcln_text = mcln_raw.decode("utf-8")
    verifier_text = verifier_raw.decode("utf-8")

    mode_start = main_text.index("    def _set_source_moe_train_mode(")
    mode_end = main_text.index("    def train_one_epoch(", mode_start)
    mode_text = main_text[mode_start:mode_end]
    required_mode_fragments = (
        "model.eval()",
        "if parent_relative_text_verifier_only:",
        '"structured_slot_builder", "sacr_head",',
        '"parent_relative_text_verifier"',
        "module.train()",
    )
    if any(fragment not in mode_text for fragment in required_mode_fragments):
        raise ValueError("failed train-mode contract changed")
    if "unwrapped.training = True" in mode_text:
        raise ValueError("failed train mode unexpectedly enabled MCLN root")
    _verify_train_mode_selected_before_forward(main_text)

    required_mcln_fragments = (
        "verifier_parent_scores = parent_scores",
        "if (self.training",
        "and self.parent_relative_text_verifier_counterfactual_training):",
        "parent_scores.detach().requires_grad_(True)",
        "verifier_batch[\"default_scores\"].retain_grad()",
    )
    if any(fragment not in mcln_text for fragment in required_mcln_fragments):
        raise ValueError("failed MCLN score-axis contract changed")

    build_start = verifier_text.index(
        "def build_parent_relative_text_verifier_batch("
    )
    build_end = verifier_text.index(
        "def build_counterfactual_parent_views(", build_start
    )
    build_text = verifier_text[build_start:build_end]
    if (
            '\"default_scores\": _gather_query_values(' not in build_text
            or "retain_grad()" in build_text):
        raise ValueError("failed compact score-axis construction changed")

    # The fixed config enables the audit flag.  The frozen train-mode code
    # leaves MCLN.training false, so its only detach/requires_grad/retain branch
    # is unreachable.  The compact score is a gather result and is never
    # retained; the unconditional post-backward grad check therefore raises on
    # the first yielded batch, before the first optimizer.step().


def verify_failed_attempt(evidence_path, expected_evidence_sha256,
                          expected_audit_root=None):
    evidence_raw = _read_regular(
        evidence_path, expected_evidence_sha256,
        expected_mode="0644", expected_owner="0:0",
    )
    evidence = _load_json(evidence_raw, "failure evidence")
    if (
            evidence.get("schema") != SCHEMA
            or evidence.get("failure_stage") != EXPECTED_FAILURE_STAGE
            or evidence.get("optimizer_steps") != 0
            or evidence.get("receipts") != 0
            or evidence.get("decisions") != 0
            or evidence.get("weights") != 0
            or evidence.get("formal_validation_accessed") is not False):
        raise ValueError("failed-attempt evidence contract changed")
    root = pathlib.Path(evidence.get("audit_root", ""))
    if (
            expected_audit_root is not None
            and str(root) != str(expected_audit_root)):
        raise ValueError("failed-attempt root differs from launcher contract")
    if root.is_symlink() or root.resolve() != root or not root.is_dir():
        raise ValueError("failed-attempt root identity changed")
    root_info = root.stat()
    if (
            root_info.st_uid != 0
            or root_info.st_gid != 0
            or stat.S_IMODE(root_info.st_mode) != 0o700):
        raise ValueError("failed-attempt root metadata changed")

    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("failed-attempt artifact map is missing")
    raw_by_relative = {}
    for relative, record in artifacts.items():
        if not isinstance(record, dict):
            raise ValueError("invalid failed-attempt record: " + relative)
        raw_by_relative[relative] = _read_regular(
            root / relative,
            record.get("sha256"),
            expected_size=record.get("size"),
            expected_mode=record.get("mode"),
            expected_owner=record.get("owner"),
        )

    input_files, _ = _walk_regular_files(root / "input_snapshot")
    expected_input_files = {
        relative.split("/", 1)[1]
        for relative in artifacts
        if relative.startswith("input_snapshot/")
    }
    if (
            len(expected_input_files)
            != evidence.get("input_snapshot_file_count")
            or set(input_files) != expected_input_files):
        raise ValueError("failed input-snapshot file set changed")
    runtime_manifest_raw = raw_by_relative[RUNTIME_MANIFEST_RELATIVE_PATH]
    runtime_records = _verify_code_snapshot(
        root / "code_snapshot", runtime_manifest_raw
    )

    non_snapshot_files, _ = _walk_regular_files(
        root, skipped_top_level=("code_snapshot", "input_snapshot")
    )
    expected_non_snapshot_files = {
        relative for relative in artifacts
        if not relative.startswith("code_snapshot/")
        and not relative.startswith("input_snapshot/")
    }
    if set(non_snapshot_files) != expected_non_snapshot_files:
        raise ValueError("failed non-snapshot file set changed")

    config_relative = (
        "runtime_output/nr3d/{}/{}/config.json".format(
            evidence["experiment"], evidence["runtime_timestamp"]
        )
    )
    config = _load_json(raw_by_relative[config_relative], "failed config")
    expected_config = {
        "exp": evidence["experiment"],
        "start_epoch": 58,
        "max_epoch": 58,
        "max_train_batches": 100,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "parent_relative_text_verifier_counterfactual_training": True,
        "parent_relative_text_verifier_detach_inputs": False,
        "eval": False,
        "eval_train": False,
        "local_rank": 0,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError("failed config changed: " + key)

    pre = _load_json(
        raw_by_relative["pre_audit_provenance.json"], "failed provenance"
    )
    if (
            pre.get("schema")
            != "mcln-fpr-tv-counterfactual-parent-pre-audit-v1"
            or pre.get("code_snapshot_root")
            != str(root / "code_snapshot")
            or pre.get("observed_sha256", {}).get("launcher")
            != evidence.get("original_launcher_sha256")
            or pre.get("observed_sha256", {}).get("main_utils")
            != artifacts["code_snapshot/main_utils.py"]["sha256"]):
        raise ValueError("failed pre-audit provenance changed")

    command_text = raw_by_relative["train_command.txt"].decode("utf-8")
    _verify_failed_runtime_selection(config, command_text)
    for fragment in (
            "--max_train_batches 100", "--start_epoch 58", "--max_epoch 58",
            "--parent_relative_text_verifier_counterfactual_training"):
        if fragment not in command_text:
            raise ValueError("failed command changed: " + fragment)
    if " --eval " in command_text or command_text.rstrip().endswith(" --eval"):
        raise ValueError("failed command unexpectedly requested evaluation")

    _verify_zero_step_control_flow(
        raw_by_relative["code_snapshot/main_utils.py"],
        raw_by_relative["runtime_output/launch.log"],
    )
    mcln_record = runtime_records["models/mcln.py"]
    verifier_record = runtime_records[
        "models/parent_relative_text_verifier.py"
    ]
    _verify_first_batch_failure_is_input_independent(
        raw_by_relative["code_snapshot/main_utils.py"],
        _read_regular(
            root / "code_snapshot/models/mcln.py",
            mcln_record.get("sha256"),
            expected_size=mcln_record.get("size"),
            expected_mode="0444",
            expected_owner="65532:65532",
        ),
        _read_regular(
            root / "code_snapshot/models/parent_relative_text_verifier.py",
            verifier_record.get("sha256"),
            expected_size=verifier_record.get("size"),
            expected_mode="0444",
            expected_owner="65532:65532",
        ),
    )
    needle_root = str(root).encode("utf-8")
    needle_exp = evidence["experiment"].encode("utf-8")
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command_line = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if needle_root in command_line or needle_exp in command_line:
            raise ValueError(
                "failed-attempt process is still alive: " + entry.name
            )
    return evidence


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3:
        raise SystemExit(
            "usage: verify_nr3d_fpr_tv_av4_failed_attempt.py "
            "EVIDENCE EXPECTED_SHA256 EXPECTED_AUDIT_ROOT"
        )
    expected_sha256 = argv[1]
    if (
            len(expected_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in expected_sha256)):
        raise SystemExit("expected evidence SHA256 must be lowercase hex")
    try:
        evidence = verify_failed_attempt(argv[0], expected_sha256, argv[2])
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(str(error))
    print(
        "failed_attempt_verified={} optimizer_steps=0 receipts=0 "
        "decisions=0 weights=0".format(evidence["failure_stage"])
    )


if __name__ == "__main__":
    main()
