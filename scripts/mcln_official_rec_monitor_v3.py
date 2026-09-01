#!/usr/bin/env python3
import glob
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import time

import torch


STATE_SCHEMA = "mcln-official-rec-monitor-v1"
RECEIPT_SCHEMA = "mcln-retrain-metrics-v1"
RECEIPT_SCOPE = "formal_exact_sample_count"
DATASET_FORMAL_SAMPLE_COUNTS = {
    "nr3d": 7899,
    "sr3d": 17726,
}


def _atomic_json(path, payload):
    directory = os.path.dirname(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(directory)
    finally:
        if temporary is not None and os.path.lexists(temporary):
            os.unlink(temporary)


def _fsync_directory(directory):
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _acquire_single_instance_lock(control_dir):
    import fcntl

    lock_path = os.path.join(control_dir, "official_rec_monitor.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    handle = os.fdopen(descriptor, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        handle.close()
        raise RuntimeError("another official REC monitor holds the lock")
    return handle


def _exact_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{} must be an integer".format(name))
    return value


def _finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be numeric".format(name))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    return value


def _parse_metric_group(group, expected_sample_count, name):
    sample_count = _exact_int(
        group["sample_count"], "{}.sample_count".format(name)
    )
    if sample_count != expected_sample_count:
        raise ValueError("{} has the wrong sample count".format(name))
    hits025 = _exact_int(group["hits025"], "{}.hits025".format(name))
    hits050 = _exact_int(group["hits050"], "{}.hits050".format(name))
    if not 0 <= hits025 <= sample_count:
        raise ValueError("{}.hits025 is outside the sample range".format(name))
    if not 0 <= hits050 <= hits025:
        raise ValueError(
            "{}.hits050 must be between zero and hits025".format(name)
        )
    rec025 = _finite_float(group["acc025"], "{}.acc025".format(name))
    rec050 = _finite_float(group["acc050"], "{}.acc050".format(name))
    expected_rec025 = (
        float(hits025) / float(sample_count) if sample_count else 0.0
    )
    expected_rec050 = (
        float(hits050) / float(sample_count) if sample_count else 0.0
    )
    if abs(rec025 - expected_rec025) > 1e-12:
        raise ValueError("{}.acc025 does not match hits/count".format(name))
    if abs(rec050 - expected_rec050) > 1e-12:
        raise ValueError("{}.acc050 does not match hits/count".format(name))
    return hits025, hits050, rec025, rec050


def _parse_formal_receipt(path, expected_sample_count):
    with open(path, "r", encoding="utf-8") as handle:
        receipt = json.load(handle)
        receipt_stat = os.fstat(handle.fileno())
    match = re.search(r"epoch_(\d+)\.json$", path)
    if match is None:
        raise ValueError("receipt filename does not contain an epoch")
    epoch = int(match.group(1))
    sample_count = _exact_int(receipt["sample_count"], "sample_count")
    position_subgroups = receipt["position_subgroups"]
    if set(position_subgroups) != {"unique", "multiple"}:
        raise ValueError("position_subgroups must be exactly unique/multiple")
    position = position_subgroups["multiple"]
    position_sample_count = _exact_int(
        position["sample_count"], "multiple.sample_count"
    )
    if (sample_count != expected_sample_count
            or position_sample_count != expected_sample_count):
        return None
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("receipt schema is not the reviewed metrics schema")
    unique = position_subgroups["unique"]
    unique_count = _exact_int(unique["sample_count"], "unique.sample_count")
    if unique_count + position_sample_count != sample_count:
        raise ValueError("position subgroup counts do not partition sample_count")
    _parse_metric_group(unique, 0, "unique")
    hits025, hits050, rec025, rec050 = _parse_metric_group(
        position, expected_sample_count, "multiple"
    )

    return {
        "epoch": epoch,
        "mtime_ns": receipt_stat.st_mtime_ns,
        "receipt": path,
        "sample_count": sample_count,
        "rec025": rec025,
        "rec050": rec050,
        "hits025": hits025,
        "hits050": hits050,
    }


def load_rows(root, expected_sample_count):
    rows = []
    ignored_nonformal_count = 0
    invalid_receipt_count = 0
    pattern = os.path.join(root, "**", "eval_metrics_epoch_*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            row = _parse_formal_receipt(path, expected_sample_count)
            if row is None:
                ignored_nonformal_count += 1
            else:
                rows.append(row)
        except (KeyError, OSError, TypeError, ValueError, AttributeError,
                json.JSONDecodeError):
            invalid_receipt_count += 1
    return rows, ignored_nonformal_count, invalid_receipt_count


def _preserved_checkpoint_name(epoch, rec025):
    return "official_best_rec025_epoch_{}_{}.pth".format(
        epoch, "{:.8f}".format(rec025).replace(".", "p")
    )


def _hash_open_file(handle):
    digest = hashlib.sha256()
    while True:
        block = handle.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _validated_checkpoint_identity(path, expected_epoch):
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("checkpoint is not a regular file")
        digest_before = _hash_open_file(handle)
        handle.seek(0)
        checkpoint = torch.load(handle, map_location="cpu")
        checkpoint_epoch = _exact_int(
            checkpoint.get("epoch"), "checkpoint.epoch"
        )
        if checkpoint_epoch != expected_epoch:
            raise ValueError("checkpoint epoch does not match expected epoch")
        handle.seek(0)
        digest_after = _hash_open_file(handle)
        after = os.fstat(handle.fileno())
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or digest_before != digest_after:
        raise ValueError("checkpoint changed while being validated")
    return digest_before


def _copy_valid_checkpoint(row, control_dir):
    directory = os.path.dirname(row["receipt"])
    candidates = [
        os.path.join(directory, "ckpt_epoch_{}.pth".format(row["epoch"])),
        os.path.join(directory, "ckpt_epoch_last.pth"),
    ]
    for candidate in candidates:
        source_descriptor = None
        temporary = None
        try:
            source_descriptor = os.open(
                candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            source_stat_before = os.fstat(source_descriptor)
            if not stat.S_ISREG(source_stat_before.st_mode):
                raise ValueError("checkpoint source is not a regular file")
            destination = os.path.join(
                control_dir,
                _preserved_checkpoint_name(row["epoch"], row["rec025"]),
            )
            temporary_descriptor, temporary = tempfile.mkstemp(
                prefix=os.path.basename(destination) + ".",
                suffix=".tmp",
                dir=control_dir,
            )
            with os.fdopen(source_descriptor, "rb") as source_handle:
                source_descriptor = None
                with os.fdopen(temporary_descriptor, "w+b") as output_handle:
                    copied_digest = hashlib.sha256()
                    while True:
                        block = source_handle.read(1024 * 1024)
                        if not block:
                            break
                        output_handle.write(block)
                        copied_digest.update(block)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                    source_stat_after = os.fstat(source_handle.fileno())
                    source_identity_before = (
                        source_stat_before.st_dev,
                        source_stat_before.st_ino,
                        source_stat_before.st_size,
                        source_stat_before.st_mtime_ns,
                    )
                    source_identity_after = (
                        source_stat_after.st_dev,
                        source_stat_after.st_ino,
                        source_stat_after.st_size,
                        source_stat_after.st_mtime_ns,
                    )
                    if source_identity_before != source_identity_after:
                        raise ValueError("checkpoint changed while being copied")
                    if output_handle.tell() != source_stat_before.st_size:
                        raise ValueError("checkpoint copy size is incomplete")
                    output_handle.seek(0)
                    checkpoint = torch.load(output_handle, map_location="cpu")
                    checkpoint_epoch = _exact_int(
                        checkpoint.get("epoch"), "checkpoint.epoch"
                    )
                    if checkpoint_epoch != row["epoch"]:
                        raise ValueError("checkpoint epoch does not match receipt")
                    os.fchmod(output_handle.fileno(), 0o444)
                    os.fsync(output_handle.fileno())
                    published_stat = os.fstat(output_handle.fileno())
            os.replace(temporary, destination)
            temporary = None
            _fsync_directory(control_dir)
            destination_stat = os.stat(destination, follow_symlinks=False)
            if (
                    destination_stat.st_dev != published_stat.st_dev
                    or destination_stat.st_ino != published_stat.st_ino
                    or destination_stat.st_size != published_stat.st_size):
                raise RuntimeError("published checkpoint identity changed")
            return destination, copied_digest.hexdigest()
        except Exception:
            pass
        finally:
            if source_descriptor is not None:
                os.close(source_descriptor)
            if temporary is not None and os.path.lexists(temporary):
                os.unlink(temporary)
    return None


def _validated_preserved_state(
        previous,
        formal_rows,
        control_dir,
        expected_sample_count,
        initial_best,
        initial_epoch,
        initial_checkpoint,
        initial_checkpoint_sha256):
    initial = {
        "preserved_best_rec025": initial_best,
        "preserved_best_epoch": initial_epoch,
        "preserved_checkpoint": initial_checkpoint,
        "preserved_checkpoint_sha256": initial_checkpoint_sha256,
    }
    if (
            previous.get("schema") != STATE_SCHEMA
            or previous.get("receipt_scope") != RECEIPT_SCOPE
            or previous.get("expected_sample_count") != expected_sample_count):
        return initial
    try:
        previous_best = _finite_float(
            previous["preserved_best_rec025"], "preserved_best_rec025"
        )
        previous_epoch = _exact_int(
            previous["preserved_best_epoch"], "preserved_best_epoch"
        )
        previous_checkpoint = previous["preserved_checkpoint"]
        previous_checkpoint_sha256 = previous[
            "preserved_checkpoint_sha256"
        ]
    except (KeyError, TypeError, ValueError):
        return initial
    if previous_best < initial_best - 1e-12:
        return initial
    if abs(previous_best - initial_best) <= 1e-12:
        return initial
    if not any(
            row["epoch"] == previous_epoch
            and abs(row["rec025"] - previous_best) <= 1e-12
            for row in formal_rows):
        return initial
    if (
            not isinstance(previous_checkpoint, str)
            or not isinstance(previous_checkpoint_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", previous_checkpoint_sha256)):
        return initial
    expected_checkpoint = os.path.join(
        control_dir,
        _preserved_checkpoint_name(previous_epoch, previous_best),
    )
    if os.path.abspath(previous_checkpoint) != os.path.abspath(
            expected_checkpoint):
        return initial
    if os.path.islink(previous_checkpoint) or not os.path.isfile(
            previous_checkpoint):
        return initial
    try:
        actual_checkpoint_sha256 = _validated_checkpoint_identity(
            previous_checkpoint, previous_epoch
        )
    except Exception:
        return initial
    if actual_checkpoint_sha256 != previous_checkpoint_sha256:
        return initial
    return {
        "preserved_best_rec025": previous_best,
        "preserved_best_epoch": previous_epoch,
        "preserved_checkpoint": previous_checkpoint,
        "preserved_checkpoint_sha256": previous_checkpoint_sha256,
    }


def _load_previous_state(state_path, dataset):
    if not os.path.isfile(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("dataset") == dataset:
            return previous
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return {}


def _update_state_once(state, root, control_dir, expected_sample_count):
    rows, ignored_count, invalid_count = load_rows(
        root, expected_sample_count
    )
    state["ignored_nonformal_receipt_count"] = ignored_count
    state["invalid_receipt_count"] = invalid_count
    if not rows:
        state["formal_receipt_count"] = 0
        state["last_receipt"] = None
        for key in (
                "metric_best_receipt_epoch",
                "metric_best_receipt_rec025",
                "metric_best_receipt",
                "latest_epoch",
                "latest_rec025",
                "latest_rec050"):
            state.pop(key, None)
        state["target_reached"] = (
            float(state["preserved_best_rec025"])
            > float(state["target_rec025"])
        )
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return state, None

    latest = max(
        rows, key=lambda item: (item["mtime_ns"], item["receipt"])
    )
    metric_best = max(
        rows, key=lambda item: (item["rec025"], item["mtime_ns"])
    )
    changed_receipt = None
    if latest["receipt"] != state.get("last_receipt"):
        changed_receipt = latest
        state["last_receipt"] = latest["receipt"]

    if metric_best["rec025"] > float(state["preserved_best_rec025"]):
        published = _copy_valid_checkpoint(metric_best, control_dir)
        if published:
            destination, destination_sha256 = published
            previous_checkpoint = state.get("preserved_checkpoint")
            if (
                    previous_checkpoint
                    and os.path.realpath(
                        os.path.dirname(previous_checkpoint)
                    ) == os.path.realpath(control_dir)
                    and previous_checkpoint != destination
                    and not os.path.islink(previous_checkpoint)
                    and os.path.isfile(previous_checkpoint)):
                state["checkpoint_cleanup_after_state_write"] = (
                    previous_checkpoint
                )
            state["preserved_best_rec025"] = metric_best["rec025"]
            state["preserved_best_epoch"] = metric_best["epoch"]
            state["preserved_checkpoint"] = destination
            state["preserved_checkpoint_sha256"] = destination_sha256
            state["new_preserved_best"] = {
                "epoch": metric_best["epoch"],
                "rec025": metric_best["rec025"],
            }

    state["formal_receipt_count"] = len(rows)
    state["metric_best_receipt_epoch"] = metric_best["epoch"]
    state["metric_best_receipt_rec025"] = metric_best["rec025"]
    state["metric_best_receipt"] = metric_best["receipt"]
    state["latest_epoch"] = latest["epoch"]
    state["latest_rec025"] = latest["rec025"]
    state["latest_rec050"] = latest["rec050"]
    state["target_reached"] = (
        float(state["preserved_best_rec025"]) > float(state["target_rec025"])
    )
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return state, changed_receipt


def main():
    if len(sys.argv) != 9:
        raise SystemExit(
            "usage: monitor.py DATASET ROOT CONTROL_DIR TARGET "
            "INITIAL_BEST INITIAL_EPOCH INITIAL_CHECKPOINT "
            "EXPECTED_SAMPLE_COUNT"
        )
    dataset, root, control_dir = sys.argv[1:4]
    target = float(sys.argv[4])
    initial_best = float(sys.argv[5])
    initial_epoch = int(sys.argv[6])
    initial_checkpoint = sys.argv[7]
    expected_sample_count = int(sys.argv[8])
    if dataset not in DATASET_FORMAL_SAMPLE_COUNTS:
        raise SystemExit("dataset must be nr3d or sr3d")
    required_sample_count = DATASET_FORMAL_SAMPLE_COUNTS[dataset]
    if expected_sample_count != required_sample_count:
        raise SystemExit(
            "expected sample count for {} must be {}".format(
                dataset, required_sample_count
            )
        )
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise SystemExit("target must be finite and in [0, 1]")
    if not math.isfinite(initial_best) or not 0.0 <= initial_best <= 1.0:
        raise SystemExit("initial best must be finite and in [0, 1]")
    if initial_epoch < 0:
        raise SystemExit("initial epoch must be non-negative")
    os.makedirs(control_dir, exist_ok=True)
    if (
            os.path.islink(initial_checkpoint)
            or not os.path.isfile(initial_checkpoint)
            or os.path.realpath(os.path.dirname(initial_checkpoint))
            != os.path.realpath(control_dir)):
        raise SystemExit(
            "initial checkpoint must be a regular file in control_dir"
        )
    monitor_lock = _acquire_single_instance_lock(control_dir)
    try:
        initial_checkpoint_sha256 = _validated_checkpoint_identity(
            initial_checkpoint, initial_epoch
        )
    except Exception as error:
        monitor_lock.close()
        raise SystemExit(
            "initial checkpoint validation failed: {}".format(error)
        )

    state_path = os.path.join(control_dir, "official_rec_monitor_state.json")
    state = {
        "schema": STATE_SCHEMA,
        "dataset": dataset,
        "target_rec025": target,
        "preserved_best_rec025": initial_best,
        "preserved_best_epoch": initial_epoch,
        "preserved_checkpoint": initial_checkpoint,
        "preserved_checkpoint_sha256": initial_checkpoint_sha256,
        "last_receipt": None,
    }
    previous_state = _load_previous_state(state_path, dataset)
    state.update(previous_state)
    formal_rows, _, _ = load_rows(root, expected_sample_count)
    state.update(
        _validated_preserved_state(
            previous_state,
            formal_rows,
            control_dir,
            expected_sample_count,
            initial_best,
            initial_epoch,
            initial_checkpoint,
            initial_checkpoint_sha256,
        )
    )
    # The reviewed launch contract always overrides stale scope and target
    # fields. A newer preserved checkpoint survives only after the validation
    # above binds it to an exact formal receipt and the control directory.
    state["schema"] = STATE_SCHEMA
    state["dataset"] = dataset
    state["target_rec025"] = target
    state["expected_sample_count"] = expected_sample_count
    state["receipt_scope"] = RECEIPT_SCOPE
    state.pop("new_preserved_best", None)

    try:
        while True:
            state, changed_receipt = _update_state_once(
                state, root, control_dir, expected_sample_count
            )
            if changed_receipt is not None:
                print(
                    time.strftime("[%F %T %Z]"),
                    dataset,
                    "latest formal",
                    "epoch={}".format(changed_receipt["epoch"]),
                    "samples={}".format(changed_receipt["sample_count"]),
                    "rec025={:.8f}".format(changed_receipt["rec025"]),
                    "rec050={:.8f}".format(changed_receipt["rec050"]),
                    flush=True,
                )
            new_best = state.pop("new_preserved_best", None)
            if new_best is not None:
                print(
                    time.strftime("[%F %T %Z]"),
                    dataset,
                    "preserved new official REC@0.25 best",
                    "epoch={}".format(new_best["epoch"]),
                    "value={:.8f}".format(new_best["rec025"]),
                    flush=True,
                )
            checkpoint_cleanup = state.pop(
                "checkpoint_cleanup_after_state_write", None
            )
            _atomic_json(state_path, state)
            if (
                    checkpoint_cleanup
                    and os.path.realpath(
                        os.path.dirname(checkpoint_cleanup)
                    ) == os.path.realpath(control_dir)
                    and not os.path.islink(checkpoint_cleanup)
                    and os.path.isfile(checkpoint_cleanup)):
                os.unlink(checkpoint_cleanup)
                _fsync_directory(control_dir)
            time.sleep(60)
    finally:
        monitor_lock.close()


if __name__ == "__main__":
    main()
