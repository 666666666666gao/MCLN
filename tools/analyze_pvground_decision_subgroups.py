"""Read completed B2 audit records; no model construction or new inference."""

import argparse
import json
from pathlib import Path

from summarize_pvground_fixed_audit import compare, read_complete


def counts(rows, threshold):
    return {
        "samples": len(rows),
        "selected_hits": sum(row["selected_iou"] > threshold for row in rows),
        "oracle_hits": sum(row["oracle_iou"] > threshold for row in rows),
        "top10_covered": sum(any(value > threshold for value in row["top10_ious"])
                             for row in rows),
    }


def transitions(base, other, threshold):
    groups = {}
    for label, unique in (("unique", True), ("multiple", False)):
        for changed in (False, True):
            pairs = [(a, b) for a, b in zip(base, other)
                     if a["is_unique"] == unique
                     and (a["selected_query"] != b["selected_query"]) == changed]
            groups[f"{label}_query_{'changed' if changed else 'same'}"] = {
                "samples": len(pairs),
                "bypass_gained": sum(a["selected_iou"] <= threshold < b["selected_iou"]
                                     for a, b in pairs),
                "bypass_lost": sum(b["selected_iou"] <= threshold < a["selected_iou"]
                                   for a, b in pairs),
            }
    return groups


def geometry_groups(rows, threshold):
    groups = {}
    for native_ok in (False, True):
        for reference_ok in (False, True):
            for final_ok in (False, True):
                selected = [row for row in rows
                            if (row["native_iou"] > threshold) == native_ok
                            and (row["reference_iou"] > threshold) == reference_ok
                            and (row["final_iou"] > threshold) == final_ok]
                name = f"native{int(native_ok)}_reference{int(reference_ok)}_final{int(final_ok)}"
                groups[name] = {
                    "samples": len(selected),
                    "multiple_samples": sum(not row["is_unique"] for row in selected),
                    "reference_valid": sum(row["selected_reference_valid"] for row in selected),
                    "samples_with_clipped_axis": sum(
                        any(value < 0 or value > 1 for value in row["selected_raw_gate"])
                        for row in selected),
                    "first_example_identities": [
                        {"sample_index": row["sample_index"], "scan_id": row["scan_id"],
                         "target_id": row["target_id"]}
                        for row in selected[:5]
                    ],
                }
    assert sum(group["samples"] for group in groups.values()) == 9508
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    directory = options.directory
    assert int((directory / "pipeline_exit_code").read_text()) == 0
    preparation = json.loads((directory / "preparation.json").read_text())
    sha = preparation["checkpoint_sha256"]
    summaries, records = {}, {}
    for condition in ("none", "reader", "support", "span"):
        summaries[condition], records[condition] = read_complete(
            directory / f"{condition}.jsonl", condition, sha,
        )
        if condition != "none":
            compare(records["none"], records[condition])
    result = {
        "scope": "Postprocessing the completed fixed-B2 interventions. GT-based diagnostics, not trained ablations, attainable accuracy or causal mechanism proof.",
        "checkpoint_sha256": sha,
        "jsonl_sha256": {name: summary["jsonl_sha256"] for name, summary in summaries.items()},
        "thresholds": {},
    }
    for threshold, suffix in ((0.25, "025"), (0.5, "050")):
        details = {
            "conditions": {
                condition: {
                    label: counts([row for row in rows if row["is_unique"] == unique], threshold)
                    for label, unique in (("unique", True), ("multiple", False))
                } for condition, rows in records.items()
            },
            "paired_bypass_transitions": {
                condition: transitions(records["none"], records[condition], threshold)
                for condition in ("reader", "support", "span")
            },
            "baseline_selected_query_geometry": geometry_groups(records["none"], threshold),
        }
        for condition, groups in details["conditions"].items():
            assert sum(group["samples"] for group in groups.values()) == 9508
            assert sum(group["selected_hits"] for group in groups.values()) == summaries[condition][f"hits{suffix}"]
        result["thresholds"][suffix] = details
    options.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(options.output),
        "baseline050": result["thresholds"]["050"]["conditions"]["none"],
        "reader050": result["thresholds"]["050"]["paired_bypass_transitions"]["reader"],
        "geometry050": {
            key: {name: value for name, value in group.items()
                  if name != "first_example_identities"}
            for key, group in result["thresholds"]["050"]["baseline_selected_query_geometry"].items()
        },
    }))


if __name__ == "__main__":
    main()
