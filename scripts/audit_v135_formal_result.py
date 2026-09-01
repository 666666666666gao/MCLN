#!/usr/bin/env python
"""Audit all V135 formal epochs and make the V99 comparison explicit."""

from __future__ import print_function

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v133_receipt_utils import atomic_write_new_json


SCHEMA = "mcln-v135-formal-result-audit-v2"
METRIC_SCHEMA = "mcln-retrain-metrics-v1"
DIAGNOSTIC_SCHEMA = "mcln-source-choice-diagnostics-v1"
RETENTION_SCHEMA = "mcln-checkpoint-retention-v1"
CONTRACT_SCHEMA = "mcln-v135-relation-counterfactual-contract-v1"
ADMISSION_SCHEMA = "mcln-v135-formal-admission-v2"
V99_ARCHIVE_SCHEMA = "mcln.v99-rec025-best-archive.v1"
V109_RETENTION_SCHEMA = "mcln.v109-permanent-retention.v1"
EXPECTED_SAMPLE_COUNT = 9508
EXPECTED_EPOCHS = (1, 2, 3, 4)
GOAL_HITS = {"rec025": 5610, "rec050": 4659}
RETENTION_METRICS = (
    "rec_acc025",
    "rec_acc050",
    "mask_acc025",
    "mask_acc050",
    "mask_miou",
)
DEFAULT_V99_ARCHIVE = Path(
    "/root/autodl-tmp/DATA_ROOT/output/v99_rec025_best_archive/"
    "v99_rec025_best_archive.json"
)
DEFAULT_V109_RETENTION = Path(
    "/root/autodl-tmp/DATA_ROOT/output/protected_weights/"
    "v109_permanent_retention.json"
)
EXPECTED_V99_ARCHIVE_SHA256 = (
    "6c5a98cd5734bb6916a1af250b71c0e4c19725378fddbdc7611796252967afdb"
)
EXPECTED_V109_RETENTION_SHA256 = (
    "28b721bad9c7474c891877c5f8d4afb9cc684f491428f9da076948e4c7421b7e"
)
EXPECTED_CONTRACT_SHA256 = (
    "eb3d9d6fc80ac447a676fffd71d86f9bddf3391dae3e620442fe7308a8c65542"
)
EXPECTED_ADMISSION_SHA256 = (
    "c612ac28611b6345ed70a01c45e3cf338d8f64cbd0f033e84723885c71266dd8"
)
EXPECTED_TRAINABLE_PREFIXES = (
    "structured_slot_builder.rel_attn.,"
    "structured_slot_builder.anchor_attn.,"
    "sacr_head.anchor_mlp.,"
    "sacr_head.relation_mlp.,"
    "sacr_head.geo_encoder."
)
EXPECTED_TRAINABLE_PARAMETERS = 421249
EXPECTED_RUN_DIR = Path(
    "/root/autodl-tmp/DATA_ROOT/output/network_v135_relation_counterfactual/"
    "v135_relation_counterfactual_formal_e1_e4_b8x1/scanrefer/"
    "v135_relation_counterfactual_formal_e1_e4_b8x1/1786818966"
)
EXPECTED_LAUNCH_LOG = Path(
    "/root/autodl-tmp/DATA_ROOT/output/network_v135_relation_counterfactual/"
    "launch/v135_relation_counterfactual_formal_e1_e4_b8x1_"
    "20260816_023602.log"
)
EXPECTED_PROPOSAL_CHECKPOINT = Path(
    "/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth"
)
EXPECTED_PROPOSAL_CHECKPOINT_SHA256 = (
    "9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
)
FROZEN_CONFIG = {
    "num_target": 256,
    "sampling": "kps",
    "num_encoder_layers": 3,
    "num_decoder_layers": 6,
    "self_position_embedding": "loc_learned",
    "self_attend": True,
    "model": "MCLN",
    "query_points_obj_topk": 4,
    "use_contrastive_align": True,
    "use_soft_token_loss": True,
    "detect_intermediate": True,
    "batch_size": 8,
    "dataset": ["scanrefer"],
    "test_dataset": "scanrefer",
    "joint_det": False,
    "butd": True,
    "butd_gt": False,
    "butd_cls": False,
    "augment_det": True,
    "use_height": False,
    "use_color": True,
    "use_multiview": False,
    "num_workers": 2,
    "dataloader_prefetch_factor": 1,
    "persistent_train_workers": False,
    "max_epoch": 4,
    "start_epoch": 1,
    "rng_seed": 0,
    "debug": False,
    "debug_train_holdout": False,
    "eval": False,
    "eval_train": False,
    "expected_eval_sample_count": EXPECTED_SAMPLE_COUNT,
    "checkpoint_metric_retention": True,
    "checkpoint_start_epoch": None,
    "save_freq": 1,
    "val_freq": 1,
    "ap_iou_thresholds": [0.25, 0.5],
    "use_source_choice_selector": True,
    "source_choice_selector_train_only": False,
    "source_choice_selector_lr": 0.001,
    "source_choice_selector_hidden_dim": 288,
    "source_choice_selector_loss_weight": 0.0,
    "source_choice_selector_sources": (
        "default,default_rank_blend_contrastive010"
    ),
    "source_choice_selector_default_source": "default",
    "source_choice_selector_choice_target": (
        "precision_gain_default_sourcewise_focal_bce"
    ),
    "source_choice_selector_min_iou_gap": 0.03,
    "eval_use_selector_choice_scores": True,
    "use_source_moe": False,
    "use_decoder_query_adapter": False,
    "use_query_mask_fusion_calibrator": False,
    "use_egqs_mask_refiner": False,
    "use_joint_query_quality_reranker": False,
    "use_sacr_source": False,
    "use_sacr_score_refiner": True,
    "sacr_score_refiner_train_only": True,
    "sacr_score_use_parent_relative_abstention": False,
    "sacr_score_use_relation_counterfactual": True,
    "sacr_score_refiner_lr": 0.0003,
    "sacr_score_refiner_loss_weight": 1.0,
    "sacr_score_temperature": 0.1,
    "sacr_score_mask_weight": 0.25,
    "sacr_score_max_delta": 0.25,
    "sacr_score_promotion_margin": 0.01,
    "sacr_score_mask_tolerance": 0.02,
    "sacr_counterfactual_parent_top_k": 16,
    "sacr_counterfactual_target_tolerance": 0.05,
    "sacr_counterfactual_attribute_tolerance": 0.05,
    "sacr_counterfactual_geometry_threshold": 0.08,
    "sacr_counterfactual_iou_gap": 0.10,
    "sacr_counterfactual_correct_iou_threshold": 0.25,
    "sacr_counterfactual_pair_margin": 0.25,
    "sacr_counterfactual_max_negatives": 4,
    "sacr_counterfactual_relation_scale": 4.0,
    "sacr_counterfactual_deployment_threshold": 0.05,
    "sacr_hidden_dim": 288,
    "sacr_max_pairs": 3,
    "sacr_top_m_targets": 32,
    "sacr_top_k_anchors": 16,
    "sacr_geo_dim": 16,
    "sacr_min_parse_confidence": 0.0,
    "sacr_score_contract_audit": False,
    "sacr_residual_scale_init": 0.1,
    "optimizer": "adamW",
    "weight_decay": 0.0005,
    "lr": 2e-5,
    "lr_backbone": 2e-4,
    "text_encoder_lr": 3e-6,
    "lr_scheduler": "step",
    "lr_decay_epochs": [8, 10],
    "lr_decay_rate": 0.1,
    "clip_norm": 0.1,
    "frozen": False,
    "small_lr": False,
    "pp_checkpoint": str(EXPECTED_PROPOSAL_CHECKPOINT),
    "exp": "v135_relation_counterfactual_formal_e1_e4_b8x1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def regular_snapshot(path, required_mode=None):
    path = Path(path).expanduser().absolute()
    entry = os.lstat(str(path))
    require(not stat.S_ISLNK(entry.st_mode),
            "audit input must not be a symlink: {}".format(path))
    require(stat.S_ISREG(entry.st_mode),
            "audit input must be a regular file: {}".format(path))
    mode = stat.S_IMODE(entry.st_mode)
    if required_mode is not None:
        require(mode == required_mode,
                "audit input must have mode {:04o}: {}".format(
                    required_mode, path
                ))
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256(path),
        "size": int(entry.st_size),
        "mode": format(mode, "04o"),
    }


def readonly_snapshot(path):
    return regular_snapshot(path, required_mode=0o444)


def require_regular_file(path, label):
    entry = os.lstat(str(path))
    require(not stat.S_ISLNK(entry.st_mode) and stat.S_ISREG(entry.st_mode),
            "{} must be a non-symlink regular file: {}".format(label, path))
    return entry


def numbered_files(run_dir, prefix):
    files = {}
    pattern = re.compile(r"^{}_epoch_(\d+)\.json$".format(prefix))
    for path in Path(run_dir).glob("{}_epoch_*.json".format(prefix)):
        match = pattern.match(path.name)
        require(match is not None, "unexpected epoch filename: {}".format(path))
        epoch = int(match.group(1))
        require(epoch not in files, "duplicate epoch receipt")
        files[epoch] = path.resolve()
    return files


def validate_config(run_dir, admission):
    config = read_json(Path(run_dir) / "config.json")
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in FROZEN_CONFIG.items() if config.get(key) != value
    }
    require(not mismatches, "V135 formal config changed: {}".format(mismatches))
    require(Path(config["log_dir"]).resolve() == Path(run_dir).resolve(),
            "V135 formal log_dir differs from run directory")
    parent = admission["parent_checkpoint"]
    checkpoint_path = Path(config["checkpoint_path"]).resolve()
    require(checkpoint_path == Path(parent["path"]).resolve(),
            "V135 formal parent checkpoint path changed")
    checkpoint_snapshot = regular_snapshot(checkpoint_path)
    require(checkpoint_snapshot["sha256"] == parent["sha256"],
            "V135 formal parent checkpoint SHA-256 changed")
    proposal_snapshot = regular_snapshot(config["pp_checkpoint"])
    require(proposal_snapshot["sha256"]
            == EXPECTED_PROPOSAL_CHECKPOINT_SHA256,
            "V135 proposal checkpoint SHA-256 changed")
    return config


def validate_provenance(args):
    contract_path = Path(args.contract_receipt).resolve()
    admission_path = Path(args.formal_admission).resolve()
    v99_path = Path(args.v99_archive).resolve()
    v109_path = Path(args.v109_retention).resolve()
    contract_snapshot = readonly_snapshot(contract_path)
    admission_snapshot = readonly_snapshot(admission_path)
    v99_snapshot = readonly_snapshot(v99_path)
    v109_snapshot = readonly_snapshot(v109_path)
    require(contract_snapshot["sha256"] == EXPECTED_CONTRACT_SHA256,
            "V135 contract receipt SHA-256 changed")
    require(admission_snapshot["sha256"] == EXPECTED_ADMISSION_SHA256,
            "V135 formal admission SHA-256 changed")
    require(v99_snapshot["sha256"] == EXPECTED_V99_ARCHIVE_SHA256,
            "V99 archive SHA-256 changed")
    require(v109_snapshot["sha256"] == EXPECTED_V109_RETENTION_SHA256,
            "V109 permanent-retention receipt SHA-256 changed")
    contract = read_json(contract_path)
    require(contract.get("schema") == CONTRACT_SCHEMA
            and contract.get("verdict") == "pass",
            "V135 contract did not pass")
    for relative, expected in contract["source_sha256"].items():
        require(sha256(ROOT / relative) == expected,
                "contract source changed: {}".format(relative))
    admission = read_json(admission_path)
    require(admission.get("schema") == ADMISSION_SCHEMA,
            "unexpected V135 admission schema")
    require(admission.get("verdict") == "formal_admission_pass",
            "V135 admission did not pass")
    require(admission.get("scope") == "safety_and_execution_only",
            "V135 admission scope changed")
    require(admission.get("efficacy_status")
            == "inconclusive_no_rec025_oracle_headroom",
            "V135 smoke was misrepresented as efficacy evidence")
    require(admission["contract_receipt"]["sha256"]
            == contract_snapshot["sha256"],
            "V135 admission contract binding changed")
    v99 = read_json(v99_path)
    require(v99.get("schema") == V99_ARCHIVE_SCHEMA,
            "unexpected V99 archive schema")
    require(v99["best_system"]["sample_count"] == EXPECTED_SAMPLE_COUNT,
            "V99 archive sample count changed")
    v109 = read_json(v109_path)
    require(v109.get("schema") == V109_RETENTION_SCHEMA,
            "unexpected V109 permanent-retention schema")
    require(v109.get("decision")
            == "retain_per_explicit_user_instruction",
            "V109 permanent-retention decision changed")
    require(v109.get("weight_copy_count") == 0,
            "V109 permanent-retention receipt is not metadata-only")
    require(set(v109.get("protection_scope", ()))
            == {"v108_parent", "v108_geometry", "v109_policy"},
            "V109 protected weight scope changed")
    for name, recorded in v109["artifacts"].items():
        current = readonly_snapshot(recorded["path"])
        for field in ("path", "sha256", "size", "mode"):
            require(current[field] == recorded[field],
                    "V109 protected artifact changed: {} {}".format(
                        name, field
                    ))
    return {
        "contract": contract_snapshot,
        "formal_admission": admission_snapshot,
        "v99_archive": v99_snapshot,
        "v109_permanent_retention": v109_snapshot,
    }, v99["best_system"], admission


def validate_metric(metric):
    require(metric.get("schema") == METRIC_SCHEMA,
            "unexpected V135 metric schema")
    require(metric.get("sample_count") == EXPECTED_SAMPLE_COUNT,
            "V135 metric sample count changed")
    learned = metric["position"]["learned_selector"]
    for name in ("hits025", "hits050"):
        value = learned[name]
        require(isinstance(value, int) and not isinstance(value, bool),
                "V135 REC hit count is invalid")
        require(0 <= value <= EXPECTED_SAMPLE_COUNT,
                "V135 REC hit count is out of range")
    mask = metric["mask"]
    for name in ("hits025", "hits050"):
        value = mask[name]
        require(isinstance(value, int) and not isinstance(value, bool),
                "V135 mask hit count is invalid")
        require(0 <= value <= EXPECTED_SAMPLE_COUNT,
                "V135 mask hit count is out of range")
    require(math.isfinite(mask["miou"]) and 0 <= mask["miou"] <= 1,
            "V135 mask mIoU is invalid")
    for family, groups in (
        ("position_subgroups", metric["position_subgroups"]),
        ("mask.position_subgroups", mask["position_subgroups"]),
    ):
        require(set(groups) == {"unique", "multiple"},
                "V135 subgroup names changed: {}".format(family))
        require(sum(group["sample_count"] for group in groups.values())
                == EXPECTED_SAMPLE_COUNT,
                "V135 subgroup sample count differs: {}".format(family))
        for hit_key in ("hits025", "hits050"):
            overall = learned if family == "position_subgroups" else mask
            require(sum(group[hit_key] for group in groups.values())
                    == overall[hit_key],
                    "V135 subgroup hits differ: {} {}".format(
                        family, hit_key
                    ))


def validate_diagnostic(metric, diagnostic):
    require(diagnostic.get("schema") == DIAGNOSTIC_SCHEMA,
            "unexpected V135 diagnostic schema")
    require(diagnostic.get("sample_count") == EXPECTED_SAMPLE_COUNT,
            "V135 diagnostic sample count changed")
    learned = metric["position"]["learned_selector"]
    parent = diagnostic["sacr_parent"]
    effects = diagnostic["sacr_parent_effects"]
    fixed = metric["position"]["fixed_default"]
    for hit_key in ("hits025", "hits050"):
        require(parent[hit_key] == fixed[hit_key],
                "V135 diagnostic parent differs from fixed source")
    for suffix, hit_key in (("025", "hits025"), ("050", "hits050")):
        effect = effects[suffix]
        require(
            learned[hit_key] == parent[hit_key]
            + effect["sacr_parent_fix"] - effect["sacr_parent_break"],
            "V135 fix/break accounting differs from REC",
        )
        headroom = diagnostic["sacr_feasible_oracle_headroom"][hit_key]
        oracle = diagnostic["sacr_feasible_oracle"][hit_key]
        require(oracle == parent[hit_key] + headroom,
                "V135 feasible-oracle accounting differs")


def parse_training_contract(launch_text):
    trainable_line = (
        EXPECTED_TRAINABLE_PREFIXES
        + "_train_only: trainable parameters {}".format(
            EXPECTED_TRAINABLE_PARAMETERS
        )
    )
    require(trainable_line in launch_text,
            "V135 formal trainable prefix/count contract changed")
    tracked = (
        "sacr_score_hard_negative_row_ratio",
        "sacr_score_selected_negative_count_mean",
        "sacr_score_parent_drift_abs_max",
        "sacr_score_residual_abs_max",
    )
    values = {}
    number = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    for name in tracked:
        parsed = [
            float(value) for value in re.findall(
                re.escape(name) + r"\s+" + number, launch_text
            )
        ]
        require(parsed, "V135 training log lacks {}".format(name))
        require(all(math.isfinite(value) and value >= 0 for value in parsed),
                "V135 training log has invalid {}".format(name))
        values[name] = parsed
    require(max(values["sacr_score_hard_negative_row_ratio"]) > 0,
            "V135 formal training mined no hard-negative rows")
    require(max(values["sacr_score_selected_negative_count_mean"]) > 0,
            "V135 formal training selected no hard negatives")
    require(max(values["sacr_score_parent_drift_abs_max"]) <= 1e-12,
            "V135 formal training changed the parent score")
    require(max(values["sacr_score_residual_abs_max"]) <= 0.250001,
            "V135 formal training exceeded the residual bound")
    return {
        "logged_steps": len(
            values["sacr_score_hard_negative_row_ratio"]
        ),
        "hard_negative_row_ratio_max": max(
            values["sacr_score_hard_negative_row_ratio"]
        ),
        "selected_negative_count_mean_max": max(
            values["sacr_score_selected_negative_count_mean"]
        ),
        "parent_drift_abs_max": max(
            values["sacr_score_parent_drift_abs_max"]
        ),
        "residual_abs_max": max(values["sacr_score_residual_abs_max"]),
    }


def normalized_epoch(epoch, metric, diagnostic, v99):
    learned = metric["position"]["learned_selector"]
    mask = metric["mask"]
    effects = diagnostic["sacr_parent_effects"]
    oracle = diagnostic["sacr_feasible_oracle"]
    return {
        "epoch": epoch,
        "rec": {
            "hits025": learned["hits025"],
            "hits050": learned["hits050"],
            "acc025": learned["hits025"] / float(EXPECTED_SAMPLE_COUNT),
            "acc050": learned["hits050"] / float(EXPECTED_SAMPLE_COUNT),
            "delta_v99_hits025": (
                learned["hits025"]
                - v99["rec"]["overall"]["acc_025"]["hits"]
            ),
            "delta_v99_hits050": (
                learned["hits050"]
                - v99["rec"]["overall"]["acc_050"]["hits"]
            ),
            "unique": metric["position_subgroups"]["unique"],
            "multiple": metric["position_subgroups"]["multiple"],
        },
        "mask": {
            "hits025": mask["hits025"],
            "hits050": mask["hits050"],
            "acc025": mask["hits025"] / float(EXPECTED_SAMPLE_COUNT),
            "acc050": mask["hits050"] / float(EXPECTED_SAMPLE_COUNT),
            "miou": mask["miou"],
            "delta_v99_hits025": (
                mask["hits025"]
                - v99["mask"]["overall"]["acc_025"]["hits"]
            ),
            "delta_v99_hits050": (
                mask["hits050"]
                - v99["mask"]["overall"]["acc_050"]["hits"]
            ),
            "delta_v99_miou": (
                mask["miou"] - v99["mask"]["overall"]["miou"]
            ),
            "unique": mask["position_subgroups"]["unique"],
            "multiple": mask["position_subgroups"]["multiple"],
        },
        "effects": effects,
        "feasible_oracle": oracle,
        "feasible_oracle_headroom": (
            diagnostic["sacr_feasible_oracle_headroom"]
        ),
    }


def expected_retention_values(row):
    return {
        "rec_acc025": row["rec"]["acc025"],
        "rec_acc050": row["rec"]["acc050"],
        "mask_acc025": row["mask"]["acc025"],
        "mask_acc050": row["mask"]["acc050"],
        "mask_miou": row["mask"]["miou"],
    }


def validate_retention(run_dir, rows, complete):
    manifest_path = Path(run_dir) / "checkpoint_retention.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == RETENTION_SCHEMA,
            "unexpected checkpoint-retention schema")
    by_epoch = {row["epoch"]: row for row in rows}
    for epoch, row in by_epoch.items():
        record = manifest["records"].get(str(epoch))
        require(record is not None,
                "checkpoint retention is missing epoch {}".format(epoch))
        expected = expected_retention_values(row)
        for name, value in expected.items():
            require(abs(record[name] - value) <= 1e-12,
                    "checkpoint retention metric differs: epoch {} {}".format(
                        epoch, name
                    ))
    expected_best = {}
    for name in RETENTION_METRICS:
        value, epoch = max(
            (expected_retention_values(row)[name], -row["epoch"])
            for row in rows
        )
        expected_best[name] = {"epoch": -epoch, "value": value}
    require(manifest["best"] == expected_best,
            "checkpoint retention best map differs")
    expected_latest_epoch = max(by_epoch)
    require(manifest["latest_epoch"] == expected_latest_epoch,
            "V135 latest checkpoint and metric epochs differ")
    if complete:
        require(expected_latest_epoch == EXPECTED_EPOCHS[-1],
                "completed V135 latest epoch is not 4")
    for name, best in expected_best.items():
        alias = Path(run_dir) / "ckpt_best_{}.pth".format(name)
        source = Path(run_dir) / "ckpt_epoch_{}.pth".format(best["epoch"])
        require_regular_file(alias, "best checkpoint alias")
        require_regular_file(source, "best checkpoint source")
        require(os.path.samefile(str(alias), str(source)),
                "best checkpoint alias has the wrong inode: {}".format(name))
    latest = Path(run_dir) / "ckpt_epoch_last.pth"
    latest_source = Path(run_dir) / "ckpt_epoch_{}.pth".format(
        expected_latest_epoch
    )
    require_regular_file(latest, "latest checkpoint alias")
    require_regular_file(latest_source, "latest checkpoint source")
    require(os.path.samefile(str(latest), str(latest_source)),
            "latest checkpoint alias has the wrong inode")
    expected_paths = {
        "ckpt_epoch_last.pth",
        "ckpt_epoch_{}.pth".format(expected_latest_epoch),
    }
    expected_paths.update(
        "ckpt_best_{}.pth".format(name) for name in RETENTION_METRICS
    )
    expected_paths.update(
        "ckpt_epoch_{}.pth".format(best["epoch"])
        for best in expected_best.values()
    )
    actual_checkpoint_paths = list(Path(run_dir).glob("ckpt*.pth"))
    actual_paths = {path.name for path in actual_checkpoint_paths}
    require(actual_paths == expected_paths,
            "V135 checkpoint cleanup differs: expected={} actual={}".format(
                sorted(expected_paths), sorted(actual_paths)
            ))
    for path in actual_checkpoint_paths:
        require_regular_file(path, "retained checkpoint")
    retained_epochs = sorted(
        {best["epoch"] for best in expected_best.values()}
        | {expected_latest_epoch}
    )
    expected_inodes = {
        os.stat(str(Path(run_dir) / name)).st_ino for name in expected_paths
    }
    require(len(expected_inodes) == len(retained_epochs),
            "V135 checkpoint aliases are not inode-deduplicated")
    checkpoint_sources = {
        str(epoch): regular_snapshot(
            Path(run_dir) / "ckpt_epoch_{}.pth".format(epoch)
        )
        for epoch in retained_epochs
    }
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256(manifest_path),
        "latest_epoch": manifest["latest_epoch"],
        "best": expected_best,
        "checkpoint_paths": sorted(actual_paths),
        "physical_checkpoint_count": len(expected_inodes),
        "checkpoint_sources": checkpoint_sources,
    }


def audit(args):
    run_dir = Path(args.run_dir).resolve()
    require(run_dir == EXPECTED_RUN_DIR,
            "V135 auditor is bound to formal run {}".format(
                EXPECTED_RUN_DIR
            ))
    require(run_dir.is_dir(), "V135 run directory is missing")
    provenance, v99, admission = validate_provenance(args)
    validate_config(run_dir, admission)
    launch_log = Path(args.launch_log).resolve()
    require(launch_log == EXPECTED_LAUNCH_LOG,
            "V135 formal launch log path changed")
    require(launch_log.is_file(), "V135 formal launch log is missing")
    launch_text = launch_log.read_text(encoding="utf-8", errors="replace")
    require(
        "starting v135_relation_counterfactual_formal_e1_e4_b8x1 on GPU 0"
        in launch_text,
        "V135 formal launch identity is missing",
    )
    require(
        "Full config saved to {}/config.json".format(run_dir)
        in launch_text,
        "V135 formal launch log is not bound to the audited run directory",
    )
    require("length of training dataset: 36665" in launch_text
            and "length of testing dataset: 9508" in launch_text,
            "V135 formal dataset sizes changed")
    training_contract = parse_training_contract(launch_text)
    metric_files = numbered_files(run_dir, "eval_metrics")
    diagnostic_files = numbered_files(run_dir, "source_choice_diagnostics")
    require(metric_files, "V135 has no formal metric receipts")
    require(set(metric_files) == set(diagnostic_files),
            "V135 metric and diagnostic epochs differ")
    epochs = tuple(sorted(metric_files))
    if args.allow_partial:
        require(epochs == tuple(range(1, max(epochs) + 1)),
                "partial V135 epochs are not contiguous from one")
    else:
        require(epochs == EXPECTED_EPOCHS,
                "completed V135 must contain exactly four epochs")
    rows = []
    evidence = {}
    for epoch in epochs:
        metric = read_json(metric_files[epoch])
        diagnostic = read_json(diagnostic_files[epoch])
        validate_metric(metric)
        validate_diagnostic(metric, diagnostic)
        rows.append(normalized_epoch(epoch, metric, diagnostic, v99))
        evidence[str(epoch)] = {
            "metrics": {
                "path": str(metric_files[epoch]),
                "sha256": sha256(metric_files[epoch]),
            },
            "diagnostics": {
                "path": str(diagnostic_files[epoch]),
                "sha256": sha256(diagnostic_files[epoch]),
            },
        }
    complete = epochs == EXPECTED_EPOCHS
    retention = validate_retention(run_dir, rows, complete)
    best_row = max(
        rows,
        key=lambda row: (
            row["rec"]["hits025"], row["rec"]["hits050"],
            row["mask"]["hits025"], row["mask"]["hits050"],
            row["mask"]["miou"], -row["epoch"],
        ),
    )
    v99_hits025 = v99["rec"]["overall"]["acc_025"]["hits"]
    v99_hits050 = v99["rec"]["overall"]["acc_050"]["hits"]
    improves_v99_rec025 = best_row["rec"]["hits025"] > v99_hits025
    reaches_joint_goal = any(
        row["rec"]["hits025"] >= GOAL_HITS["rec025"]
        and row["rec"]["hits050"] >= GOAL_HITS["rec050"]
        for row in rows
    )
    payload = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "status": "complete" if complete else "partial",
        "run_dir": str(run_dir),
        "config_sha256": sha256(run_dir / "config.json"),
        "provenance": provenance,
        "training_contract": training_contract,
        "epoch_evidence": evidence,
        "epochs": rows,
        "retention": retention,
        "v99_reference": {
            "rec_hits025": v99_hits025,
            "rec_hits050": v99_hits050,
        },
        "best_epoch_by_rec025": best_row["epoch"],
        "best_rec_hits025": best_row["rec"]["hits025"],
        "best_rec_hits050_same_epoch": best_row["rec"]["hits050"],
        "improves_v99_rec025": improves_v99_rec025,
        "reaches_joint_goal_same_epoch": reaches_joint_goal,
        "decision": (
            "retain_v135_and_review"
            if improves_v99_rec025
            else "freeze_v99_and_start_pure_scanrefer_single_stage"
        ) if complete else "continue_frozen_v135_formal",
    }
    if complete:
        launch_snapshot = {
            "path": str(launch_log),
            "sha256": sha256(launch_log),
        }
        require("finished v135_relation_counterfactual_formal" in launch_text,
                "V135 formal launch did not finish cleanly")
        require("Traceback" not in launch_text
                and "CUDA out of memory" not in launch_text,
                "V135 formal launch contains an execution failure")
        payload["launch_log"] = launch_snapshot
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--formal-admission", required=True)
    parser.add_argument("--contract-receipt", required=True)
    parser.add_argument("--v99-archive", default=str(DEFAULT_V99_ARCHIVE))
    parser.add_argument(
        "--v109-retention", default=str(DEFAULT_V109_RETENTION)
    )
    parser.add_argument("--launch-log", required=True)
    parser.add_argument("--output")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.allow_partial:
        require(args.output is None,
                "partial audit prints only and cannot publish a receipt")
    else:
        require(args.output is not None,
                "completed audit requires --output")
    payload = audit(args)
    if args.allow_partial:
        print(json.dumps(payload, sort_keys=True))
    else:
        output = atomic_write_new_json(payload, args.output)
        print(json.dumps({
            "output": str(output),
            "sha256": sha256(output),
            "mode": format(stat.S_IMODE(os.stat(str(output)).st_mode), "04o"),
            "decision": payload["decision"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
