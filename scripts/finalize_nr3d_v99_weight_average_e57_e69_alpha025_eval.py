#!/usr/bin/env python3
from __future__ import print_function

import argparse
import hashlib
import json
import math
import os


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_with_sha(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def find_unique_receipt(run_root):
    matches = []
    for directory, _, names in os.walk(run_root):
        if "eval_metrics_epoch_57.json" in names:
            matches.append(os.path.join(directory, "eval_metrics_epoch_57.json"))
    if len(matches) != 1:
        raise SystemExit("expected exactly one epoch-57 receipt, found {}".format(len(matches)))
    return os.path.realpath(matches[0])


def publish_json_no_overwrite(path, payload):
    directory = os.path.dirname(path)
    temporary = path + ".tmp.{}".format(os.getpid())
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--official-monitor-state", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--expected-sample-count", type=int, default=7899)
    parser.add_argument("--base-hits025", type=int, default=4463)
    parser.add_argument("--target-hits025", type=int, default=4724)
    args = parser.parse_args()

    run_root = os.path.realpath(args.run_root)
    decision_path = os.path.join(run_root, "decision.json")
    if os.path.lexists(decision_path):
        raise SystemExit("decision already exists")

    receipt_path = find_unique_receipt(run_root)
    manifest_path = os.path.join(run_root, "weight_average_manifest.json")
    checkpoint_path = os.path.join(run_root, "candidate_e57_075_e69_025.pth")
    claim_path = os.path.join(run_root, "one_shot_claim.json")
    command_path = os.path.join(run_root, "eval_command.txt")
    pre_eval_path = os.path.join(run_root, "pre_eval_provenance.json")

    receipt, receipt_sha = load_json_with_sha(receipt_path)
    manifest, manifest_sha = load_json_with_sha(manifest_path)
    claim, claim_sha = load_json_with_sha(claim_path)
    pre_eval, pre_eval_sha = load_json_with_sha(pre_eval_path)
    monitor, monitor_sha = load_json_with_sha(args.official_monitor_state)

    current_code = {
        name: sha256_file(path)
        for name, path in claim["code_paths"].items()
    }
    if current_code != claim.get("code_sha256"):
        raise SystemExit("code or GroupFree drifted during evaluation")
    if current_code != pre_eval.get("code_sha256"):
        raise SystemExit("post-eval code hashes differ from pre-eval hashes")

    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != manifest.get("output_sha256"):
        raise SystemExit("candidate checkpoint SHA differs from manifest")
    if checkpoint_sha != pre_eval.get("candidate_checkpoint_sha256"):
        raise SystemExit("candidate checkpoint differs from pre-eval provenance")
    if manifest_sha != pre_eval.get("manifest_sha256"):
        raise SystemExit("weight-average manifest drifted")
    if claim_sha != pre_eval.get("claim_sha256"):
        raise SystemExit("one-shot claim drifted")
    command_sha = sha256_file(command_path)
    if command_sha != pre_eval.get("eval_command_sha256"):
        raise SystemExit("evaluation command drifted")

    if receipt.get("schema") != "mcln-retrain-metrics-v1":
        raise SystemExit("evaluation receipt schema mismatch")
    rec = receipt.get("position_subgroups", {}).get("multiple", {})
    mask = receipt.get("mask", {}).get("position_subgroups", {}).get("multiple", {})
    if (
        receipt.get("sample_count") != args.expected_sample_count
        or rec.get("sample_count") != args.expected_sample_count
        or mask.get("sample_count") != args.expected_sample_count
    ):
        raise SystemExit("evaluation sample-count mismatch")

    hits025 = int(rec.get("hits025", -1))
    hits050 = int(rec.get("hits050", -1))
    acc025 = float(rec.get("acc025", float("nan")))
    acc050 = float(rec.get("acc050", float("nan")))
    mask_hits025 = int(mask.get("hits025", -1))
    mask_hits050 = int(mask.get("hits050", -1))
    mask_acc025 = float(mask.get("acc025", float("nan")))
    mask_acc050 = float(mask.get("acc050", float("nan")))
    mask_miou = float(receipt.get("mask", {}).get("miou", float("nan")))
    for accuracy, hits, label in (
        (acc025, hits025, "REC@0.25"),
        (acc050, hits050, "REC@0.50"),
        (mask_acc025, mask_hits025, "Mask@0.25"),
        (mask_acc050, mask_hits050, "Mask@0.50"),
    ):
        if not math.isclose(
            accuracy,
            hits / float(args.expected_sample_count),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit("{} hits/accuracy mismatch".format(label))
    if not math.isfinite(mask_miou):
        raise SystemExit("Mask mIoU is not finite")

    official_checkpoint = os.path.realpath(args.official_checkpoint)
    official_checkpoint_sha = sha256_file(official_checkpoint)
    if official_checkpoint_sha != checkpoint_sha:
        raise SystemExit("official preserved checkpoint is not the evaluated candidate")
    if monitor.get("last_receipt") != receipt_path:
        raise SystemExit("official monitor has not consumed the unique receipt")
    if not math.isclose(
        float(monitor.get("preserved_best_rec025", float("nan"))),
        acc025,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit("official monitor has not preserved the candidate metric")
    if os.path.realpath(monitor.get("preserved_checkpoint", "")) != official_checkpoint:
        raise SystemExit("official monitor checkpoint path mismatch")

    finalizer_path = os.path.realpath(__file__)
    decision = {
        "schema": "mcln-nr3d-v99-e57-e69-weight-average-eval-v1",
        "sample_count": args.expected_sample_count,
        "hits025": hits025,
        "hits050": hits050,
        "acc025": acc025,
        "acc050": acc050,
        "mask_hits025": mask_hits025,
        "mask_hits050": mask_hits050,
        "mask_acc025": mask_acc025,
        "mask_acc050": mask_acc050,
        "mask_miou": mask_miou,
        "base_hits025": args.base_hits025,
        "strict_target_hits025": args.target_hits025,
        "improves_protected_best": hits025 > args.base_hits025,
        "strict_target_reached": hits025 >= args.target_hits025,
        "checkpoint": os.path.realpath(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "receipt": receipt_path,
        "receipt_sha256": receipt_sha,
        "weight_average_manifest": os.path.realpath(manifest_path),
        "weight_average_manifest_sha256": manifest_sha,
        "one_shot_claim": os.path.realpath(claim_path),
        "one_shot_claim_sha256": claim_sha,
        "eval_command": os.path.realpath(command_path),
        "eval_command_sha256": command_sha,
        "pre_eval_provenance": os.path.realpath(pre_eval_path),
        "pre_eval_provenance_sha256": pre_eval_sha,
        "code_sha256": current_code,
        "official_monitor_state": os.path.realpath(args.official_monitor_state),
        "official_monitor_state_sha256": monitor_sha,
        "official_checkpoint": official_checkpoint,
        "official_checkpoint_sha256": official_checkpoint_sha,
        "finalizer": finalizer_path,
        "finalizer_sha256": sha256_file(finalizer_path),
        "recovered_after_launcher_receipt_path_bug": True,
        "evaluation_rerun": False,
    }
    publish_json_no_overwrite(decision_path, decision)
    os.chmod(decision_path, 0o444)
    os.chmod(receipt_path, 0o444)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
