#!/usr/bin/env python
"""Seal the user-retained V109 Pareto system without copying weights."""

from __future__ import print_function

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v133_receipt_utils import atomic_write_new_json


OUTPUT_ROOT = Path("/root/autodl-tmp/DATA_ROOT/output")
DEFAULT_OUTPUT = (
    OUTPUT_ROOT / "protected_weights" / "v109_permanent_retention.json"
)
ARTIFACTS = {
    "v108_parent": (
        OUTPUT_ROOT / "rec_reranker/e71_top16_meshsp/v108_artifacts/"
        "parent_h256_seed0.pth"
    ),
    "v108_geometry": (
        OUTPUT_ROOT / "rec_reranker/e71_top16_meshsp/v108_artifacts/"
        "geometry_h256_seed0.pth"
    ),
    "v109_policy": (
        OUTPUT_ROOT / "rec_reranker/e71_top16_meshsp/v109_artifacts/"
        "nested_policy_h128_seed0_fullfit.pth"
    ),
    "v109_artifact_receipt": (
        OUTPUT_ROOT / "rec_reranker/e71_top16_meshsp/v109_artifacts/"
        "artifact_receipt.json"
    ),
    "v109_official_claim": (
        OUTPUT_ROOT / "rec_reranker/e71_top16_meshsp/v109_artifacts/"
        "v109_meshsp_official_once_after_train_runtime_parity.claim.json"
    ),
    "v109_official_result": (
        OUTPUT_ROOT / "v109_meshsp_official_20260814/official_result.json"
    ),
}
EXPECTED_SHA256 = {
    "v108_parent": "7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f",
    "v108_geometry": "20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972",
    "v109_policy": "20db69ddc27680a035384277bc48cd44109215e3d7d1158cdc4a4f21ff7c785b",
    "v109_artifact_receipt": "19f7676241b1558beb53c67f770cf8c3a3d149d3e0ee21a61579e383d53b7115",
    "v109_official_claim": "3da2e573115e4985c185fb81f85c9aba407836791e15cb40c1be5000ee8178b0",
    "v109_official_result": "9afe5160359e56f867d1f500cd906b7b2133af124b49a353ed0d50b8ab8778ba",
}
EXPECTED_MIOU = 0.459224


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(path):
    path = Path(path).expanduser().absolute()
    entry = os.lstat(str(path))
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("protected input must be a regular file: {}".format(path))
    if stat.S_IMODE(entry.st_mode) != 0o444:
        raise ValueError("protected input must have mode 0444: {}".format(path))
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256(path),
        "size": int(entry.st_size),
        "mode": "0444",
        "device": int(entry.st_dev),
        "inode": int(entry.st_ino),
    }


def validate_result(result):
    if result.get("schema") != "rec-v109-meshsp-official-validation-result-v1":
        raise ValueError("unexpected V109 result schema")
    if (
        result.get("sample_count") != 9508
        or result.get("returncode") != 0
        or result.get("validation_data_accessed") is not True
        or result.get("inference_uses_ground_truth") is not False
    ):
        raise ValueError("V109 official result is incomplete")
    metrics = result["metrics"]
    expected_overall = {
        "rec_hits025": 5551,
        "rec_hits050": 4834,
        "mask_hits025": 5689,
        "mask_hits050": 4974,
    }
    for name, expected in expected_overall.items():
        if metrics.get(name) != expected:
            raise ValueError("V109 metric changed: {}".format(name))
    expected_subgroups = {
        "position_subgroups": {
            "unique": (1260, 1151, 1419),
            "multiple": (4291, 3683, 8089),
        },
        "mask_subgroups": {
            "unique": (1280, 1137, 1419),
            "multiple": (4409, 3837, 8089),
        },
    }
    for family, groups in expected_subgroups.items():
        for group, expected in groups.items():
            actual = metrics[family][group]
            if (actual["hits25"], actual["hits50"], actual["total"]) != expected:
                raise ValueError("V109 metric changed: {}.{}".format(family, group))
    if abs(metrics["mask_miou"] - EXPECTED_MIOU) > 1e-6:
        raise ValueError("V109 mask mIoU changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    artifacts = {name: snapshot(path) for name, path in ARTIFACTS.items()}
    for name, record in artifacts.items():
        if record["sha256"] != EXPECTED_SHA256[name]:
            raise ValueError("{} SHA-256 changed".format(name))
    with Path(artifacts["v109_official_result"]["path"]).open(
        "r", encoding="utf-8"
    ) as handle:
        result = json.load(handle)
    validate_result(result)
    payload = {
        "schema": "mcln.v109-permanent-retention.v1",
        "created_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "decision": "retain_per_explicit_user_instruction",
        "archive_is_metadata_only": True,
        "weight_copy_count": 0,
        "protection_scope": ["v108_parent", "v108_geometry", "v109_policy"],
        "official_metrics": result["metrics"],
        "artifacts": artifacts,
    }
    output = atomic_write_new_json(payload, args.output)
    print(json.dumps({
        "path": str(output),
        "sha256": sha256(output),
        "mode": format(stat.S_IMODE(os.stat(str(output)).st_mode), "04o"),
        "weight_copy_count": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
