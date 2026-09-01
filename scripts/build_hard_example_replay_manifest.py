from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


SCHEMA = "mcln-hard-example-replay-v1"
CRITERIA = {
    "default_top1_iou_lte": 0.25,
    "default_topk": 5,
    "topk_oracle_iou_gt": 0.25,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path, payload):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("output already exists: {}".format(path))
    temporary = path.with_name(path.name + ".tmp")
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        os.chmod(str(path), 0o444)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--joint-dataset-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache = Path(args.candidate_cache).expanduser().resolve()
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("candidate cache manifest is missing")
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("split") != "train":
        raise ValueError("hard-example replay requires a train cache")
    dataset = cache_manifest.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("candidate cache dataset is invalid")
    sample_count = cache_manifest.get("sample_count")
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("candidate cache sample_count is invalid")
    if args.joint_dataset_size < sample_count:
        raise ValueError("joint dataset is smaller than candidate cache")

    snapshot_digest = hashlib.sha256()
    hard_examples = []
    expected_index = 0
    top1_hits025 = 0
    top1_hits050 = 0
    top5_hits025 = 0
    top5_hits050 = 0
    for shard_name in cache_manifest.get("shards", []):
        shard_path = cache / shard_name
        shard_sha256 = sha256_file(shard_path)
        snapshot_digest.update(
            ("{}\t{}\t{}\n".format(
                shard_name, shard_path.stat().st_size, shard_sha256
            )).encode("utf-8")
        )
        payload = torch.load(str(shard_path), map_location="cpu")
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("candidate cache shard lacks rows")
        for row in rows:
            dataset_index = row.get("dataset_index")
            if dataset_index != expected_index:
                raise ValueError("candidate cache indices are not contiguous")
            expected_index += 1
            valid = row["valid_mask"].bool()
            query_indices = row["query_indices"].long()
            candidate_ious = row["candidate_ious"].float()
            scores = row["default_scores"].float().clone()
            scores[~valid] = -float("inf")
            top_query = int(row["default_top1_query_index"])
            top_matches = valid & query_indices.eq(top_query)
            if int(top_matches.sum().item()) != 1:
                raise ValueError("default Top-1 query mapping is not unique")
            top_slot = int(torch.nonzero(top_matches, as_tuple=False)[0].item())
            top1_iou = float(candidate_ious[top_slot].item())
            order = torch.argsort(scores, descending=True)
            top5_slots = order[: CRITERIA["default_topk"]]
            usable_top5 = top5_slots[valid[top5_slots]]
            if not len(usable_top5):
                raise ValueError("candidate row has no valid default candidate")
            top5_ious = candidate_ious[usable_top5]
            best_offset = int(torch.argmax(top5_ious).item())
            best_slot = int(usable_top5[best_offset].item())
            top5_iou = float(candidate_ious[best_slot].item())
            top1_hits025 += int(top1_iou > 0.25)
            top1_hits050 += int(top1_iou > 0.50)
            top5_hits025 += int(top5_iou > 0.25)
            top5_hits050 += int(top5_iou > 0.50)
            if top1_iou <= 0.25 and top5_iou > 0.25:
                hard_examples.append({
                    "best_top5_iou": top5_iou,
                    "best_top5_query_index": int(query_indices[best_slot]),
                    "dataset_index": dataset_index,
                    "required_score_gap": float(
                        scores[top_slot].item() - scores[best_slot].item()
                    ),
                    "scan_id": str(row["scan_id"]),
                    "target_id": row["target_id"],
                    "top1_iou": top1_iou,
                    "top1_query_index": top_query,
                })
    if expected_index != sample_count:
        raise ValueError("candidate cache row count changed")
    if not hard_examples:
        raise ValueError("hard-example replay set is empty")

    payload = {
        "base_dataset_size": sample_count,
        "candidate_cache": {
            "checkpoint_sha256": cache_manifest.get("checkpoint_sha256"),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "path": str(cache),
            "shard_snapshot_sha256": snapshot_digest.hexdigest(),
        },
        "criteria": CRITERIA,
        "dataset": dataset,
        "diagnostics": {
            "top1_hits025": top1_hits025,
            "top1_hits050": top1_hits050,
            "top5_hits025": top5_hits025,
            "top5_hits050": top5_hits050,
        },
        "hard_count": len(hard_examples),
        "hard_examples": hard_examples,
        "joint_dataset_size": args.joint_dataset_size,
        "repeat_count": 1,
        "schema": SCHEMA,
    }
    atomic_json_write(args.output, payload)
    print(json.dumps({
        "hard_count": len(hard_examples),
        "output": str(Path(args.output).expanduser().resolve()),
        "sample_count": sample_count,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
