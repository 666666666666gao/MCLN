#!/usr/bin/env python3
"""Audit one SourceMoE validation receipt and its candidate-set oracle."""

import argparse
import json
import math
from pathlib import Path


METRIC_KEYS = (
    "rec_acc025",
    "rec_acc050",
    "mask_acc025",
    "mask_acc050",
    "mask_miou",
)


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite_rate(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be numeric".format(label))
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("{} must be in [0,1]".format(label))
    return value


def _validate_count(value, sample_count, label):
    if (not isinstance(value, int) or isinstance(value, bool)
            or not 0 <= value <= sample_count):
        raise ValueError("{} is invalid".format(label))
    return value


def _validate_iou_sum(value, sample_count, mean_iou, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be numeric".format(label))
    value = float(value)
    if (not math.isfinite(value) or value < 0.0
            or value > float(sample_count)):
        raise ValueError("{} is out of range".format(label))
    if not math.isclose(
            value / float(sample_count), mean_iou,
            rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("{} disagrees with mean IoU".format(label))
    return value


def _validated_position_subgroups(
        groups, sample_count, overall, label, required=False):
    if groups is None:
        if required:
            raise ValueError("evaluation receipt lacks {}".format(label))
        return None
    if not isinstance(groups, dict):
        raise ValueError("{} must be a dictionary".format(label))
    exported = {}
    for name in ("unique", "multiple"):
        group = groups.get(name)
        if not isinstance(group, dict):
            raise ValueError("{} {} is missing".format(label, name))
        group_count = group.get("sample_count")
        if (not isinstance(group_count, int) or isinstance(group_count, bool)
                or group_count < 0 or group_count > sample_count):
            raise ValueError(
                "{} {} sample_count is invalid".format(label, name)
            )
        hits025 = _validate_count(
            group.get("hits025"), group_count, name + " hits025"
        )
        hits050 = _validate_count(
            group.get("hits050"), group_count, name + " hits050"
        )
        if hits050 > hits025:
            raise ValueError(name + " hits050 exceeds hits025")
        expected_rates = {
            "acc025": hits025 / float(group_count) if group_count else 0.0,
            "acc050": hits050 / float(group_count) if group_count else 0.0,
        }
        for key, expected in expected_rates.items():
            rate = _finite_rate(group.get(key), name + " " + key)
            if not math.isclose(
                    rate, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    "{} {} {} disagrees with hits".format(
                        label, name, key
                    )
                )
        exported[name] = {
            "sample_count": group_count,
            "hits025": hits025,
            "hits050": hits050,
            "acc025": expected_rates["acc025"],
            "acc050": expected_rates["acc050"],
        }
    if sum(group["sample_count"] for group in exported.values()) != sample_count:
        raise ValueError("unique and multiple counts do not partition samples")
    for suffix in ("025", "050"):
        if sum(
                group["hits" + suffix] for group in exported.values()
        ) != overall["hits" + suffix]:
            raise ValueError(
                "unique and multiple hits{} do not partition learned hits"
                .format(suffix)
            )
    return exported


def _position_subgroups_from_receipt(
        receipt, sample_count, learned, required=False):
    return _validated_position_subgroups(
        receipt.get("position_subgroups"),
        sample_count,
        learned,
        "position subgroups",
        required=required,
    )


def metrics_from_receipt(
        receipt, expected_sample_count=9508,
        require_position_subgroups=False):
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != "mcln-retrain-metrics-v1"):
        raise ValueError("evaluation receipt schema is invalid")
    sample_count = receipt.get("sample_count")
    if sample_count != expected_sample_count:
        raise ValueError(
            "expected {} samples but metrics contain {}".format(
                expected_sample_count, sample_count
            )
        )
    try:
        learned = receipt["position"]["learned_selector"]
        fixed = receipt["position"]["fixed_default"]
        mask = receipt["mask"]
    except (KeyError, TypeError) as error:
        raise ValueError("evaluation receipt is incomplete") from error
    for source_name, source in (("learned", learned), ("fixed", fixed)):
        hits025 = _validate_count(
            source.get("hits025"), sample_count, source_name + " hits025"
        )
        hits050 = _validate_count(
            source.get("hits050"), sample_count, source_name + " hits050"
        )
        if hits050 > hits025:
            raise ValueError(source_name + " hits050 exceeds hits025")
    mask025 = _validate_count(
        mask.get("hits025"), sample_count, "mask hits025"
    )
    mask050 = _validate_count(
        mask.get("hits050"), sample_count, "mask hits050"
    )
    if mask050 > mask025:
        raise ValueError("mask hits050 exceeds hits025")
    mask_miou = _finite_rate(mask.get("miou"), "mask miou")
    _validate_iou_sum(
        mask.get("iou_sum"), sample_count, mask_miou, "mask iou_sum"
    )
    position_subgroups = _position_subgroups_from_receipt(
        receipt,
        sample_count,
        learned,
        required=require_position_subgroups,
    )
    mask_position_subgroups = _validated_position_subgroups(
        mask.get("position_subgroups"),
        sample_count,
        {"hits025": mask025, "hits050": mask050},
        "mask position subgroups",
        required=require_position_subgroups,
    )
    result = {
        "sample_count": sample_count,
        "fixed_acc025": fixed["hits025"] / float(sample_count),
        "fixed_acc050": fixed["hits050"] / float(sample_count),
        "rec_acc025": learned["hits025"] / float(sample_count),
        "rec_acc050": learned["hits050"] / float(sample_count),
        "mask_acc025": mask025 / float(sample_count),
        "mask_acc050": mask050 / float(sample_count),
        "mask_miou": mask_miou,
    }
    if position_subgroups is not None:
        result["position_subgroups"] = position_subgroups
    if mask_position_subgroups is not None:
        result["mask_position_subgroups"] = mask_position_subgroups
    return result


def oracle_from_receipt(receipt, expected_sample_count=9508):
    if (not isinstance(receipt, dict) or receipt.get("schema")
            != "mcln-source-choice-diagnostics-v1"):
        raise ValueError("source-choice diagnostics schema is invalid")
    sample_count = receipt.get("sample_count")
    if sample_count != expected_sample_count:
        raise ValueError(
            "expected {} samples but diagnostics contain {}".format(
                expected_sample_count, sample_count
            )
        )
    try:
        oracle = receipt["gate_candidate_oracle"]
        headroom = receipt["gate_oracle_headroom"]
    except (KeyError, TypeError) as error:
        raise ValueError("source-choice diagnostics are incomplete") from error
    oracle025 = _validate_count(
        oracle.get("hits025"), sample_count, "oracle hits025"
    )
    oracle050 = _validate_count(
        oracle.get("hits050"), sample_count, "oracle hits050"
    )
    if oracle050 > oracle025:
        raise ValueError("oracle hits050 exceeds hits025")
    headroom025 = _validate_count(
        headroom.get("hits025"), sample_count, "headroom hits025"
    )
    headroom050 = _validate_count(
        headroom.get("hits050"), sample_count, "headroom hits050"
    )
    oracle_miou = _finite_rate(oracle.get("miou"), "oracle miou")
    _validate_iou_sum(
        oracle.get("iou_sum"), sample_count, oracle_miou,
        "oracle iou_sum",
    )
    for suffix, hits in (("025", headroom025), ("050", headroom050)):
        rate = _finite_rate(
            headroom.get("rate" + suffix), "headroom rate" + suffix
        )
        if not math.isclose(
                rate, hits / float(sample_count),
                rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "headroom rate{} disagrees with hits".format(suffix)
            )
    return {
        "oracle_acc025": oracle025 / float(sample_count),
        "oracle_acc050": oracle050 / float(sample_count),
        "oracle_miou": oracle_miou,
        "headroom025": headroom025 / float(sample_count),
        "headroom050": headroom050 / float(sample_count),
    }


def audit(metrics_receipt, diagnostics_receipt, baseline_receipt=None,
          target_acc025=0.59, target_acc050=0.49,
          expected_sample_count=9508):
    metrics = metrics_from_receipt(
        metrics_receipt, expected_sample_count=expected_sample_count
    )
    oracle = oracle_from_receipt(
        diagnostics_receipt, expected_sample_count=expected_sample_count
    )
    target_acc025 = _finite_rate(target_acc025, "target acc025")
    target_acc050 = _finite_rate(target_acc050, "target acc050")
    learned_pass = (
        metrics["rec_acc025"] >= target_acc025
        and metrics["rec_acc050"] >= target_acc050
    )
    oracle_pass = (
        oracle["oracle_acc025"] >= target_acc025
        and oracle["oracle_acc050"] >= target_acc050
    )
    result = {
        "schema": "mcln-source-moe-oracle-audit-v1",
        "targets": {
            "rec_acc025": target_acc025,
            "rec_acc050": target_acc050,
        },
        "metrics": metrics,
        "candidate_oracle": oracle,
        "rec_target_pass": learned_pass,
        "candidate_oracle_target_pass": oracle_pass,
    }
    mask_guard_pass = None
    if baseline_receipt is not None:
        baseline = metrics_from_receipt(
            baseline_receipt, expected_sample_count=expected_sample_count
        )
        result["baseline_metrics"] = baseline
        result["deltas_vs_baseline"] = {
            key: metrics[key] - baseline[key] for key in METRIC_KEYS
        }
        mask_guard_pass = all(
            result["deltas_vs_baseline"][key] >= 0.0
            for key in ("mask_acc025", "mask_acc050", "mask_miou")
        )
    learned_target_pass = learned_pass and mask_guard_pass is not False
    if learned_pass and mask_guard_pass is False:
        decision = "repair_mask_tradeoff"
    elif learned_target_pass:
        decision = "learned_target_reached"
    elif oracle_pass:
        decision = "train_contextual_gate"
    else:
        decision = "improve_candidate_generation"
    result["mask_guard_pass"] = mask_guard_pass
    result["learned_target_pass"] = learned_target_pass
    result["decision"] = decision
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--target-acc025", type=float, default=0.59)
    parser.add_argument("--target-acc050", type=float, default=0.49)
    parser.add_argument("--expected-sample-count", type=int, default=9508)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    result = audit(
        _load_json(args.metrics),
        _load_json(args.diagnostics),
        baseline_receipt=(
            _load_json(args.baseline) if args.baseline else None
        ),
        target_acc025=args.target_acc025,
        target_acc050=args.target_acc050,
        expected_sample_count=args.expected_sample_count,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(output)
    print(text, end="")


if __name__ == "__main__":
    main()
