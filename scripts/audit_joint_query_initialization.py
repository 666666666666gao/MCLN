#!/usr/bin/env python3
"""Audit joint-query initialization against a protected MCLN checkpoint."""

import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
for import_path in (ROOT_DIR, ROOT_DIR / "pointnet2"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from train_dist_mod import TrainTester


PROFILES = {
    "v41": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "expected_missing": 20,
        "expected_parameters": 153531,
    },
    "v42": {
        "mask_calibration": True,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "expected_missing": 22,
        "expected_parameters": 153919,
    },
    "v43": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": False,
        "expected_missing": 22,
        "expected_parameters": 155219,
    },
    "v46": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": True,
        "expected_missing": 22,
        "expected_parameters": 158339,
    },
    "v48": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": False,
        "spatial_mask_refiner": True,
        "spatial_mask_hidden_dim": 32,
        "max_spatial_mask_delta": 2.0,
        "expected_missing": 34,
        "expected_parameters": 176979,
    },
    "v49": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": False,
        "spatial_mask_refiner": True,
        "spatial_mask_hidden_dim": 32,
        "max_spatial_mask_delta": 2.0,
        "adaptive_source_mixing": True,
        "source_count": 3,
        "shared_source_index": 0,
        "max_source_mix_delta": 1.0,
        "source_mix_temperature": 0.5,
        "expected_missing": 45,
        "expected_parameters": 229460,
    },
    "v50_sacr": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": False,
        "spatial_mask_refiner": True,
        "spatial_mask_hidden_dim": 32,
        "max_spatial_mask_delta": 2.0,
        "adaptive_source_mixing": True,
        "source_count": 4,
        "shared_source_index": 0,
        "max_source_mix_delta": 1.0,
        "source_mix_temperature": 0.5,
        "joint_source_names": (
            "default,contrastive_text,mask_text,sacr_structured"
        ),
        "use_sacr_source": True,
        "expected_missing": 66,
        "expected_parameters": 1150390,
    },
    "v51_rapf_source_reliability": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": False,
        "spatial_mask_refiner": True,
        "spatial_mask_hidden_dim": 32,
        "max_spatial_mask_delta": 2.0,
        "adaptive_source_mixing": True,
        "source_distribution_reliability": True,
        "source_count": 4,
        "shared_source_index": 0,
        "max_source_mix_delta": 1.0,
        "source_mix_temperature": 0.5,
        "joint_source_names": (
            "default,contrastive_text,mask_text,sacr_structured"
        ),
        "use_sacr_source": True,
        "expected_missing": 66,
        "expected_parameters": 1151158,
    },
}


def _normalized_state(state):
    return {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in state.items()
    }


def _atomic_write_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def audit_initialization(checkpoint_path, profile):
    if profile not in PROFILES:
        raise ValueError("unknown joint-query initialization profile")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if (not isinstance(checkpoint, dict)
            or not isinstance(checkpoint.get("model"), dict)
            or checkpoint.get("config") is None):
        raise ValueError("protected checkpoint is incomplete")

    contract = PROFILES[profile]
    args = copy.deepcopy(checkpoint["config"])
    overrides = {
        "use_joint_query_quality_reranker": True,
        "joint_query_quality_hidden_dim": 128,
        "joint_query_quality_heads": 4,
        "joint_query_quality_layers": 1,
        "joint_query_quality_dropout": 0.1,
        "joint_query_quality_max_delta": 1.25,
        "joint_query_quality_mask_weight": 0.25,
        "joint_query_quality_score_weight": 1.0,
        "joint_query_quality_use_mask_calibration": contract[
            "mask_calibration"
        ],
        "joint_query_quality_max_mask_alpha_delta": 1.0,
        "joint_query_quality_max_mask_logit_bias": 2.0,
        "joint_query_quality_use_source_mask_evidence": contract[
            "source_mask_evidence"
        ],
        "joint_query_quality_use_gate_evidence": contract[
            "gate_evidence"
        ],
        "joint_query_quality_use_spatial_mask_refiner": contract.get(
            "spatial_mask_refiner", False
        ),
        "joint_query_quality_spatial_mask_hidden_dim": contract.get(
            "spatial_mask_hidden_dim", 32
        ),
        "joint_query_quality_max_spatial_mask_delta": contract.get(
            "max_spatial_mask_delta", 2.0
        ),
        "joint_query_quality_use_adaptive_source_mixing": contract.get(
            "adaptive_source_mixing", False
        ),
        "joint_query_quality_use_source_distribution_reliability": (
            contract.get("source_distribution_reliability", False)
        ),
        "joint_query_quality_source_names": contract.get(
            "joint_source_names", ""
        ),
        "joint_query_quality_max_source_mix_delta": contract.get(
            "max_source_mix_delta", 1.0
        ),
        "joint_query_quality_source_mix_temperature": contract.get(
            "source_mix_temperature", 0.5
        ),
        "use_sacr_source": contract.get("use_sacr_source", False),
        "sacr_hidden_dim": 288,
        "sacr_max_pairs": 3,
        "sacr_top_m_targets": 32,
        "sacr_top_k_anchors": 16,
        "sacr_geo_dim": 16,
        "sacr_min_parse_confidence": 0.0,
        "sacr_residual_scale_init": 0.1,
    }
    for name, value in overrides.items():
        setattr(args, name, value)

    model = TrainTester.get_model(args)
    source = _normalized_state(checkpoint["model"])
    target = model.state_dict()
    incompatible = model.load_state_dict(source, strict=False)
    loaded = model.state_dict()

    source_names = set(source)
    target_names = set(target)
    common = sorted(source_names & target_names)
    missing = sorted(target_names - source_names)
    unexpected = sorted(source_names - target_names)
    changed = sorted(
        name for name in common if not torch.equal(source[name], loaded[name])
    )
    shape_mismatches = sorted(
        name for name in common if source[name].shape != target[name].shape
    )
    joint = model.joint_query_quality_reranker
    new_parameter_numel = sum(
        parameter.numel() for parameter in joint.parameters()
    )
    expected_prefixes = ["joint_query_quality_reranker."]
    if contract.get("use_sacr_source", False):
        expected_prefixes.extend([
            "structured_slot_builder.",
            "sacr_head.",
            "sacr_residual_scale",
        ])
        new_parameter_numel += sum(
            parameter.numel()
            for parameter in model.structured_slot_builder.parameters()
        )
        new_parameter_numel += sum(
            parameter.numel() for parameter in model.sacr_head.parameters()
        )
        new_parameter_numel += model.sacr_residual_scale.numel()
    expected_missing = {
        name for name in target
        if any(name.startswith(prefix) for prefix in expected_prefixes)
    }
    zero_initialized_heads = all(
        int(torch.count_nonzero(value).item()) == 0
        for name, value in joint.state_dict().items()
        if name.startswith((
            "quality_head.", "residual_head.", "mask_calibration_head.",
            "spatial_mask_refiner.query_projection.2.",
            "adaptive_source_mixer.source_router.2.",
            "adaptive_source_mixer.strength_head.2.",
        ))
    )
    passed = (
        len(missing) == contract["expected_missing"]
        and set(missing) == expected_missing
        and not unexpected
        and not changed
        and not shape_mismatches
        and set(incompatible.missing_keys) == expected_missing
        and not incompatible.unexpected_keys
        and new_parameter_numel == contract["expected_parameters"]
        and zero_initialized_heads
    )
    result = {
        "schema": "mcln-{}-protected-initialization-audit-v1".format(profile),
        "profile": profile,
        "checkpoint": str(checkpoint_path),
        "source_state_count": len(source),
        "target_state_count": len(target),
        "common_tensor_count": len(common),
        "changed_common_tensor_count": len(changed),
        "changed_common_tensors": changed,
        "missing_tensor_count": len(missing),
        "missing_tensors": missing,
        "unexpected_tensor_count": len(unexpected),
        "unexpected_tensors": unexpected,
        "shape_mismatch_count": len(shape_mismatches),
        "shape_mismatches": shape_mismatches,
        "joint_query_quality_state_count": len(joint.state_dict()),
        "joint_query_quality_parameter_numel": sum(
            parameter.numel() for parameter in joint.parameters()
        ),
        "new_module_parameter_numel": new_parameter_numel,
        "mask_calibration": contract["mask_calibration"],
        "source_mask_evidence": contract["source_mask_evidence"],
        "gate_evidence": contract["gate_evidence"],
        "spatial_mask_refiner": contract.get("spatial_mask_refiner", False),
        "adaptive_source_mixing": contract.get(
            "adaptive_source_mixing", False
        ),
        "source_distribution_reliability": contract.get(
            "source_distribution_reliability", False
        ),
        "use_sacr_source": contract.get("use_sacr_source", False),
        "zero_initialized_output_heads": zero_initialized_heads,
        "pass": passed,
    }
    if not passed:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = audit_initialization(args.checkpoint, args.profile)
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
