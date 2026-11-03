"""Summarize complete fixed-checkpoint interventions, with exact input pairing."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


EXPECTED_SAMPLES = 9508
IDENTITY_KEYS = (
    "sample_index", "scan_id", "target_id", "utterance",
    "point_sha256", "gt_box", "is_unique",
)


def distribution(values):
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def read_complete(path, condition, checkpoint_sha):
    forward, semantic, metadata, footers = [], [], [], []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            if row["kind"] == "metadata":
                metadata.append(row)
            elif row["kind"] == "forward_complete":
                footers.append(row)
            elif row["kind"] == "forward_sample":
                forward.append(row)
            elif row["kind"] == "semantic_sample":
                semantic.append(row)
            else:
                raise ValueError(f"unknown record kind in {path}")
    if len(metadata) != 1 or len(footers) != 1:
        raise ValueError(f"{condition}: requires one metadata and completion footer")
    if metadata[0]["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError(f"{condition}: checkpoint SHA mismatch")
    if metadata[0]["bypass"] != condition:
        raise ValueError(f"{condition}: bypass mismatch")
    if len(forward) != EXPECTED_SAMPLES or len(semantic) != EXPECTED_SAMPLES:
        raise ValueError(f"{condition}: incomplete sample counts")
    footer = footers[0]
    if not footer["verified_against_evaluator"]:
        raise ValueError(f"{condition}: evaluator agreement not verified")
    if footer["samples"] != EXPECTED_SAMPLES or footer["semantic_samples"] != EXPECTED_SAMPLES:
        raise ValueError(f"{condition}: incomplete footer counts")
    for index, (fwd, sample) in enumerate(zip(forward, semantic)):
        if fwd["sample_index"] != index or sample["sample_index"] != index:
            raise ValueError(f"{condition}: nonsequential samples")
        if fwd["bypass"] != condition or not fwd["raw_gate_finite"]:
            raise ValueError(f"{condition}: invalid forward evidence")
        if sample["selected_iou"] != sample["final_iou"]:
            raise ValueError(f"{condition}: selected/final IoU mismatch")
    summary = {
        "samples": EXPECTED_SAMPLES,
        "checkpoint_sha256": checkpoint_sha,
        "jsonl_sha256": digest.hexdigest(),
    }
    for threshold, suffix in ((0.25, "025"), (0.5, "050")):
        hits = sum(row["selected_iou"] > threshold for row in semantic)
        if hits != footer[f"semantic_hits{suffix}"]:
            raise ValueError(f"{condition}: footer hit mismatch")
        summary[f"hits{suffix}"] = hits
        summary[f"percent{suffix}"] = 100 * hits / EXPECTED_SAMPLES
        summary[f"oracle_hits{suffix}"] = sum(
            row["oracle_iou"] > threshold for row in semantic
        )
        summary[f"top10_covered{suffix}"] = sum(
            any(iou > threshold for iou in row["top10_ious"])
            for row in semantic
        )
        for kind in ("native", "reference"):
            summary[f"selected_{kind}_hits{suffix}"] = sum(
                row[f"{kind}_iou"] > threshold for row in semantic
            )
    summary["selected_reference_valid"] = sum(
        row["selected_reference_valid"] for row in semantic
    )
    for key in (
        "selected_reader_residual_l2_before_bypass",
        "selected_reader_query_l2",
        "selected_support_delta_abs_mean_before_bypass",
        "selected_support_delta_abs_max_before_bypass",
        "selected_support_foreground_flips_before_bypass",
    ):
        summary[key] = distribution([row[key] for row in semantic])
    summary["selected_gate_by_axis"] = [
        {
            "axis": axis,
            "raw": distribution([row["selected_raw_gate"][axis] for row in semantic]),
            "clamped": distribution([row["selected_clamped_gate"][axis] for row in semantic]),
            "raw_below_zero": sum(row["selected_raw_gate"][axis] < 0 for row in semantic),
            "raw_above_one": sum(row["selected_raw_gate"][axis] > 1 for row in semantic),
            "clamped_zero": sum(row["selected_clamped_gate"][axis] == 0 for row in semantic),
            "clamped_one": sum(row["selected_clamped_gate"][axis] == 1 for row in semantic),
        }
        for axis in range(3)
    ]
    for kind in ("native", "reference", "final"):
        summary[f"{kind}_mean_abs_center_error_by_axis"] = [
            statistics.fmean(abs(row[f"{kind}_center_error"][axis]) for row in semantic)
            for axis in range(3)
        ]
        summary[f"{kind}_mean_abs_size_error_by_axis"] = [
            statistics.fmean(abs(row[f"{kind}_size_error"][axis]) for row in semantic)
            for axis in range(3)
        ]
    return summary, semantic


def compare(base, intervention):
    for left, right in zip(base, intervention):
        if any(left[key] != right[key] for key in IDENTITY_KEYS):
            raise ValueError(f"input/identity mismatch at sample {left['sample_index']}")
    result = {
        "paired_samples": EXPECTED_SAMPLES,
        "exact_order_identity_points_and_gt_match": True,
        "selected_query_changed": sum(
            left["selected_query"] != right["selected_query"]
            for left, right in zip(base, intervention)
        ),
    }
    for threshold, suffix in ((0.25, "025"), (0.5, "050")):
        gained = sum(
            left["selected_iou"] <= threshold < right["selected_iou"]
            for left, right in zip(base, intervention)
        )
        lost = sum(
            right["selected_iou"] <= threshold < left["selected_iou"]
            for left, right in zip(base, intervention)
        )
        result[f"bypass_gained_hits{suffix}"] = gained
        result[f"bypass_lost_hits{suffix}"] = lost
        result[f"bypass_net_hits{suffix}"] = gained - lost
    result["selected_iou_delta"] = distribution([
        right["selected_iou"] - left["selected_iou"]
        for left, right in zip(base, intervention)
    ])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--conditions", nargs="+", choices=("none", "reader", "support", "span"),
                        default=["none", "reader", "support", "span"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.conditions[0] != "none" or len(set(args.conditions)) != len(args.conditions):
        raise ValueError("conditions must begin with none and contain no duplicates")
    preparation = json.loads((args.directory / "preparation.json").read_text())
    results = {
        "scope": "Fixed trained-checkpoint inference interventions, not trained ablations or gains over official MCLN.",
        "checkpoint_sha256": preparation["checkpoint_sha256"],
        "conditions": {},
        "paired_against_none": {},
    }
    baseline = None
    for condition in args.conditions:
        summary, samples = read_complete(
            args.directory / f"{condition}.jsonl",
            condition, preparation["checkpoint_sha256"],
        )
        results["conditions"][condition] = summary
        if condition == "none":
            baseline = samples
        else:
            results["paired_against_none"][condition] = compare(baseline, samples)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "conditions": {
            mode: {key: values[key] for key in ("samples", "hits025", "hits050")}
            for mode, values in results["conditions"].items()
        },
        "paired_against_none": results["paired_against_none"],
    }))


if __name__ == "__main__":
    main()
