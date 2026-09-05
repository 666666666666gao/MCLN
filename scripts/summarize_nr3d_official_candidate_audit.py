"""Summarize a sealed native-path audit; no model execution or score tuning."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def failure_partition(row, top_k):
    selected = row["rec_selection"]
    if selected is not None and selected["box_iou"] > .25:
        return "correct"
    profiles = row["score_profiles"]["protected_selector"]
    legal = profiles["after_filter"]
    if legal["top_{}".format(top_k)]["hit025"]:
        return "reselectable_within_legal_topk"
    if legal["top_256"]["hit025"]:
        return "qualifying_box_only_beyond_legal_topk"
    if profiles["before_filter"]["top_256"]["hit025"]:
        return "qualifying_boxes_removed_by_filter"
    return "full256_no_qualifying_box"


def quality(values):
    valid = [value for value in values if value is not None]
    return {"available_rows": len(valid),
            "hits025": sum(value > .25 for value in valid),
            "hits050": sum(value > .5 for value in valid),
            "iou_sum": sum(valid),
            "mean_iou": sum(valid) / len(valid) if valid else None}


def describe(rows):
    rec = [row["rec_selection"] for row in rows]
    result = {"rows": len(rows), "scenes": len({row["scan_id"] for row in rows}),
              "rec": quality([item["box_iou"] if item is not None else None for item in rec]),
              "native_mask": quality([row["mask_selection"]["mask_iou"] for row in rows]),
              "rec_selected_query_mask": quality([item["mask_iou"] if item is not None else None for item in rec]),
              "unfiltered_selector_box": quality([row["mask_selection"]["box_iou"] for row in rows]),
              "rec_mask_query_difference_rows": sum(not row["rec_and_mask_same_query"] for row in rows),
              "no_legal_queries_rows": sum(item is None for item in rec)}
    result["failure_partitions"] = {
        str(top_k): dict(Counter(failure_partition(row, top_k) for row in rows))
        for top_k in (16, 32, 64)}
    result["oracle_hits"] = {
        source: {stage: {str(top_k): {
            "hits" + suffix: sum(row["score_profiles"][source][stage]["top_{}".format(top_k)]["hit" + suffix]
                                  for row in rows)
            for suffix in ("025", "050")}
            for top_k in (16, 32, 64, 256)}
            for stage in ("before_filter", "after_filter")}
        for source in ("protected_selector", "default")}
    result["box_oracle_mask"] = {}
    for stage in ("before_filter", "after_filter"):
        selected = [row["box_oracle_" + stage] for row in rows]
        result["box_oracle_mask"][stage] = quality(
            [item["mask_iou"] if item is not None else None for item in selected])
    result["full256_mask_oracle"] = quality([row["mask_oracle_all_queries"]["mask_iou"] for row in rows])
    result["conditional_mask"] = {}
    for name in ("rec_selection", "mask_selection", "box_oracle_after_filter"):
        subset = [row for row in rows if row[name] is not None and row[name]["box_iou"] > .5]
        result["conditional_mask"][name + "_box_iou_gt050"] = {
            "rows": len(subset),
            "same_query_mask": quality([row[name]["mask_iou"] for row in subset]),
            "full_mask_oracle_le050_rows": sum(row["mask_oracle_all_queries"]["mask_iou"] <= .5
                                               for row in subset)}
    result["target_sampling"] = {
        stage: {"zero_target_centers_rows": sum(row["target_sampled_center_counts"][stage] == 0 for row in rows),
                "target_center_sum": sum(row["target_sampled_center_counts"][stage] for row in rows)}
        for stage in ("sa1", "sa2", "sa3", "sa4", "fp2_seeds", "kps_queries")}
    result["target_sampling"]["input_target_points_sum"] = sum(row["root_target_input_points"] for row in rows)
    result["object_memory_proxy"] = {
        "expression_repeated_detector_slots": sum(len(row["detector_object_coverage"]["active_detector_slots"])
                                                   for row in rows),
        "covered_slots": {name: sum(len(row["detector_object_coverage"]["covered_slots"][name]) for row in rows)
                          for name in ("full_256", "valid_queries", "valid_default_top32")},
        "is_text_anchor_ground_truth": False}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.results / "receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["native_row_parity"]
    raw = (args.results / "rows.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt["rows_sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert [row["id"] for row in rows] == list(range(7899))
    overall = describe(rows)
    for suffix in ("025", "050"):
        assert overall["rec"]["hits" + suffix] == receipt["summary"]["rec_hits" + suffix]
        assert overall["native_mask"]["hits" + suffix] == receipt["summary"]["mask_hits" + suffix]
        assert overall["unfiltered_selector_box"]["hits" + suffix] == receipt["native_metrics"]["position"]["learned_selector"]["hits" + suffix]
    assert math.isclose(overall["native_mask"]["iou_sum"], receipt["summary"]["mask_iou_sum"], rel_tol=0, abs_tol=1e-8)
    groups = {
        "text_length": lambda row: ("2_to_6" if row["raw_token_count"] <= 6 else
                                    "7_to_8" if row["raw_token_count"] <= 8 else
                                    "9_to_12" if row["raw_token_count"] <= 12 else "13plus"),
        "distractors": lambda row: str(row["distractor_count"]) if row["distractor_count"] < 5 else "5plus",
        "input_points": lambda row: ("le32" if row["root_target_input_points"] <= 32 else
                                     "33_to_227" if row["root_target_input_points"] <= 227 else
                                     "228_to_1000" if row["root_target_input_points"] <= 1000 else "gt1000"),
        "class": lambda row: row["target_name"],
        "scene": lambda row: row["scan_id"],
        "full256_coverage": lambda row: str(row["score_profiles"]["protected_selector"]["before_filter"]["top_256"]["hit025"]),
    }
    grouped = {}
    for name, key_function in groups.items():
        subsets = defaultdict(list)
        for row in rows:
            subsets[key_function(row)].append(row)
        grouped[name] = {key: describe(subset) for key, subset in sorted(subsets.items())}
    result = {"schema": "mcln-nr3d-official-candidate-analysis-v1", "overall": overall,
              "groups": grouped, "historical_metric_match": receipt["historical_metric_match"],
              "rows_sha256": receipt["rows_sha256"],
              "limitations": ["Current frozen source, not recovered historical core source bytes.",
                              "Sampled target centers are not receptive-field coverage.",
                              "Box IoU and query seed membership do not label physical object identity.",
                              "Detector-slot coverage is not labelled text-anchor recall.",
                              "Descriptive full-validation diagnosis; no new head or score selection."]}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"overall": overall, "historical_metric_match": receipt["historical_metric_match"]}, indent=2))


if __name__ == "__main__":
    main()
