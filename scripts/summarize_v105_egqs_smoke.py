#!/usr/bin/env python3
"""Summarize the frozen V105 single-GPU ablation gate."""

import argparse
import json
import math
import re
from pathlib import Path


SCHEMA = "mcln-v105-egqs-smoke-summary-v1"
DIAGNOSTIC_PATTERN = re.compile(
    r"(egqs_mask_refiner_[a-z_]+)=([+-]?(?:[0-9.]+)(?:e[+-]?[0-9]+)?)",
    re.IGNORECASE,
)


def _atomic_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)


def _finite_number(value, label):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("{} must be numeric".format(label))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(label))
    return result


def _last_diagnostics(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    result = {}
    for name, value in DIAGNOSTIC_PATTERN.findall(text):
        result[name] = float(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--expected-sample-count", type=int, default=128)
    parser.add_argument("--expected-rec025", type=int, default=64)
    parser.add_argument("--expected-rec050", type=int, default=57)
    parser.add_argument("--miou-margin", type=float, default=0.0003)
    parser.add_argument(
        "--record", nargs=4, action="append", metavar=(
            "NAME", "RUN_DIR", "LOG", "COMPONENTS"
        ), required=True,
    )
    args = parser.parse_args()
    if args.epoch <= 0 or args.expected_sample_count <= 0:
        raise ValueError("epoch and sample count must be positive")
    if len(args.record) != 4:
        raise ValueError("V105 gate requires exactly four records")

    records = {}
    failures = []
    for name, run_dir, log_path, components in args.record:
        if name in records or components in (
                row["components"] for row in records.values()):
            raise ValueError("record names and components must be unique")
        metrics_path = Path(run_dir) / "eval_metrics_epoch_{}.json".format(
            args.epoch
        )
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        position = metrics["position"]["learned_selector"]
        mask = metrics["mask"]
        diagnostics = _last_diagnostics(log_path)
        record = {
            "components": components,
            "run_dir": str(Path(run_dir)),
            "launcher_log": str(Path(log_path)),
            "sample_count": int(metrics["sample_count"]),
            "rec_hits025": int(position["hits025"]),
            "rec_hits050": int(position["hits050"]),
            "mask_hits025": int(mask["hits025"]),
            "mask_hits050": int(mask["hits050"]),
            "mask_miou": _finite_number(mask["miou"], "mask miou"),
            "diagnostics": diagnostics,
        }
        records[name] = record
        if record["sample_count"] != args.expected_sample_count:
            failures.append("{} sample count".format(name))
        if (record["rec_hits025"], record["rec_hits050"]) != (
                args.expected_rec025, args.expected_rec050):
            failures.append("{} REC identity".format(name))
        for diagnostic in (
                "egqs_mask_refiner_residual_abs_mean",
                "egqs_mask_refiner_superpoint_std_mean",
                "egqs_mask_refiner_query_std_mean"):
            if _finite_number(
                    diagnostics.get(diagnostic, 0.0), diagnostic
            ) <= 0.0:
                failures.append("{} {}".format(name, diagnostic))

    by_component = {
        record["components"]: record for record in records.values()
    }
    if set(by_component) != {"content", "evidence", "geometry", "all"}:
        raise ValueError("records must cover content/evidence/geometry/all")
    content = by_component["content"]
    comparisons = {}
    for components in ("evidence", "geometry", "all"):
        candidate = by_component[components]
        mask050_non_degradation = (
            candidate["mask_hits050"] >= content["mask_hits050"]
        )
        miou_gain = candidate["mask_miou"] - content["mask_miou"]
        miou_gate = miou_gain >= args.miou_margin
        comparisons[components] = {
            "mask050_non_degradation": mask050_non_degradation,
            "miou_gain_over_content": miou_gain,
            "miou_margin_pass": miou_gate,
            "advancement_candidate": components == "all",
        }
        if components == "all":
            if not mask050_non_degradation:
                failures.append("all Mask@0.50 regressed")
            if not miou_gate:
                failures.append("all mIoU margin")

    payload = {
        "schema": SCHEMA,
        "pass": not failures,
        "gate": {
            "expected_sample_count": args.expected_sample_count,
            "expected_rec_hits025": args.expected_rec025,
            "expected_rec_hits050": args.expected_rec050,
            "miou_margin": args.miou_margin,
        },
        "records": records,
        "comparisons": comparisons,
        "failures": failures,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
