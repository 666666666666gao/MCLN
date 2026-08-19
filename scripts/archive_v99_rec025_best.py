#!/usr/bin/env python
"""Seal the V99 REC@0.25-best system without copying its large weights."""

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


DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/DATA_ROOT/output/v99_rec025_best_archive/"
    "v99_rec025_best_archive.json"
)
DEFAULT_ARTIFACTS = {
    "epoch71_backbone": Path(
        "/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/"
        "mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth"
    ),
    "parent_reranker": Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/artifacts/"
        "reranker_h256_d010_lr1e3_seed0_final_contract.pth"
    ),
    "geometry_reranker": Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
        "geometry_artifacts/selected_geometry_reranker.pth"
    ),
    "v99_contextual_hierarchy": Path(
        "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
        "v99_artifacts/pareto_contextual_h128_seed0_fullfit.pth"
    ),
    "official_result": Path(
        "/root/autodl-tmp/DATA_ROOT/output/v99_meshsp_official_20260814/"
        "v99_meshsp_official_result.json"
    ),
}
EXPECTED_SHA256 = {
    "epoch71_backbone": (
        "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208"
    ),
    "parent_reranker": (
        "f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b"
    ),
    "geometry_reranker": (
        "835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f"
    ),
    "v99_contextual_hierarchy": (
        "9752990c393fa6e45173a9dd129c4de4bb740924094dcbbec2f3121cbf39d1f2"
    ),
    "official_result": (
        "311097c8a0fc1eceab3c95983937071e67fd8082ac46d1af5d3701ada4eb491c"
    ),
}
SOURCE_FILES = (
    "models/rec_hierarchical_reranker.py",
    "models/rec_pareto_contextual_hierarchy.py",
    "scripts/build_v99_pareto_contextual_artifact.py",
    "scripts/run_v99_pareto_contextual_hierarchical.py",
    "scripts/run_frozen_v99_pareto_contextual_official.py",
    "scripts/audit_v99_fullfit_replay.py",
    "scripts/audit_v99_runtime_parity_train.py",
    "train_dist_mod.py",
)
EXPECTED_HITS = {
    "rec": {
        "overall": {"acc_025": (5572, 9508), "acc_050": (4797, 9508)},
        "unique": {"acc_025": (1261, 1419), "acc_050": (1143, 1419)},
        "multiple": {"acc_025": (4311, 8089), "acc_050": (3654, 8089)},
    },
    "mask": {
        "overall": {"acc_025": (5690, 9508), "acc_050": (4976, 9508)},
        "unique": {"acc_025": (1280, 1419), "acc_050": (1137, 1419)},
        "multiple": {"acc_025": (4410, 8089), "acc_050": (3839, 8089)},
    },
}
EXPECTED_MASK_MIOU = 0.4593026020554575


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path, *, require_readonly):
    path = Path(path).expanduser().absolute()
    entry = os.lstat(str(path))
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("archive input must be a regular non-symlink file: {}".format(path))
    mode = stat.S_IMODE(entry.st_mode)
    if require_readonly and mode != 0o444:
        raise ValueError("archive input must have mode 0444: {}".format(path))
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256(path),
        "size": int(entry.st_size),
        "mode": format(mode, "04o"),
        "device": int(entry.st_dev),
        "inode": int(entry.st_ino),
        "hardlink_count": int(entry.st_nlink),
    }


def _validate_result(result):
    if result.get("schema") != "mcln.scanrefer.validation-receipt.v1":
        raise ValueError("unexpected V99 result schema")
    if result.get("sample_count") != 9508 or result.get("metrics_valid") is not True:
        raise ValueError("V99 official result is incomplete")
    for family, groups in EXPECTED_HITS.items():
        for group, thresholds in groups.items():
            for threshold, expected in thresholds.items():
                metric = result[family][group][threshold]
                actual = (metric["hits"], metric["total"])
                if actual != expected:
                    raise ValueError(
                        "V99 metric changed: {}.{}.{}={} expected {}".format(
                            family, group, threshold, actual, expected
                        )
                    )
                if abs(metric["rate"] - expected[0] / float(expected[1])) > 1e-15:
                    raise ValueError("V99 metric rate is inconsistent with hits")
    if abs(result["mask"]["overall"]["miou"] - EXPECTED_MASK_MIOU) > 1e-15:
        raise ValueError("V99 mask mIoU changed")
    if result["process_exit"].get("failure_stage") != "post_metric_export":
        raise ValueError("V99 post-metric exit provenance changed")


def build_archive(args):
    artifacts = {}
    for name, default_path in DEFAULT_ARTIFACTS.items():
        path = Path(getattr(args, name) or default_path)
        artifacts[name] = _snapshot(path, require_readonly=True)
        if artifacts[name]["sha256"] != EXPECTED_SHA256[name]:
            raise ValueError("{} SHA-256 changed".format(name))

    with Path(artifacts["official_result"]["path"]).open(
        "r", encoding="utf-8"
    ) as handle:
        result = json.load(handle)
    _validate_result(result)

    repo_root = Path(args.repo_root).resolve(strict=True)
    source_manifest = {}
    for relative in SOURCE_FILES:
        source_manifest[relative] = _snapshot(
            repo_root / relative, require_readonly=False
        )

    return {
        "schema": "mcln.v99-rec025-best-archive.v1",
        "created_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "decision": "archived_after_v135_failed_to_improve_rec025",
        "archive_is_metadata_only": True,
        "weight_copy_count": 0,
        "best_system": {
            "name": "V99 + mesh-derived official superpoints",
            "dataset": "ScanRefer val",
            "sample_count": 9508,
            "rec": result["rec"],
            "mask": result["mask"],
            "goal": result["goal_status"],
        },
        "innovation_contract": {
            "candidate_scope": "frozen Top-16 REC query candidates and mask variants",
            "architecture": (
                "one-layer permutation-equivariant contextual query-set hierarchy "
                "with separate query and variant heads"
            ),
            "training_target": "bounded_iou_plus_2hit025_plus_hit050_soft_listwise",
            "selection": {
                "threshold_heads": ["predicted_hit025", "predicted_hit050"],
                "requires_strict_positive_gain": ["delta025", "delta050"],
                "aggregate_gain": "2 * delta025 + delta050",
                "aggregate_margin": 0.13312220573425293,
            },
            "fit_protocol": {
                "scene_disjoint_folds": 5,
                "fit_rows": 33040,
                "fit_scenes": 506,
                "epochs": 12,
                "seed": 0,
                "hidden_dim": 128,
                "dropout": 0.1,
                "weight_decay": 0.001,
                "validation_used_for_training_or_selection": False,
            },
            "oof_evidence": {
                "switches": 5186,
                "delta_hits025": 175,
                "delta_hits050": 474,
                "scene_bootstrap_lower_95_hits025": 132,
                "scene_bootstrap_lower_95_hits050": 385,
                "all_folds_positive": True,
                "receipt_sha256": (
                    "db42ef5853fb36fba9bdc53afb719bff9eb5a3f9e772475a4c76c363db01572d"
                ),
            },
        },
        "claim_boundary": {
            "single_complete_system": True,
            "not_metric_wise_splicing": True,
            "official_exit_code": result["process_exit"]["return_code"],
            "official_failure_stage": result["process_exit"]["failure_stage"],
            "all_predictions_and_metrics_completed": True,
            "mesh_superpoint_fix_is_data_pipeline_correction": True,
            "v99_artifact_was_not_retuned_on_validation": True,
        },
        "artifacts": artifacts,
        "source_manifest_at_archive_time": source_manifest,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    for name in DEFAULT_ARTIFACTS:
        parser.add_argument("--" + name.replace("_", "-"), dest=name)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_archive(args)
    output = atomic_write_new_json(payload, args.output)
    print(json.dumps({
        "archive": str(output),
        "sha256": _sha256(output),
        "mode": format(stat.S_IMODE(os.stat(str(output)).st_mode), "04o"),
        "weight_copy_count": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
