"""Summarize a completed P2 audit; never read a partial holdout as a result."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def summarize(rows):
    def metrics(members):
        result = {"rows": len(members), "scores": {}}
        for mode in ["protected", "default", "global", "pair"]:
            masks = [r["protected_mask_iou"] if mode == "protected"
                     else r["scores"][mode]["mask_iou"] for r in members]
            result["scores"][mode] = {
                "rec_hits025": sum(r["scores"][mode]["box_iou"] > .25 for r in members),
                "rec_hits050": sum(r["scores"][mode]["box_iou"] > .5 for r in members),
                "mask_hits025": sum(v > .25 for v in masks),
                "mask_hits050": sum(v > .5 for v in masks),
                "mask_iou_sum": sum(masks),
            }
        return result

    groups = {"overall": rows,
              "raw_tokens_2_to_6": [r for r in rows if 2 <= r["raw_token_count"] <= 6],
              "raw_tokens_7_to_8": [r for r in rows if 7 <= r["raw_token_count"] <= 8],
              "raw_tokens_9_to_12": [r for r in rows if 9 <= r["raw_token_count"] <= 12],
              "raw_tokens_13plus": [r for r in rows if r["raw_token_count"] >= 13],
              "hard_2plus_distractors": [r for r in rows if r["distractor_count"] >= 2],
              "sparse_at_most_227_sampled_points": [r for r in rows if r["target_points"] <= 227]}
    breakdown = {name: metrics(members) for name, members in groups.items()}
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scan_id"]].append(row)
    coverage = {}
    for stage in ["before_filter", "after_filter"]:
        coverage[stage] = {
            str(k): {"hits025": sum(r["oracle"][stage][str(k)] > .25 for r in rows),
                     "hits050": sum(r["oracle"][stage][str(k)] > .5 for r in rows)}
            for k in [16, 32, 64, 256]}
    failure = {"correctable_within_legal_top32": 0,
               "correct_box_only_outside_legal_top32": 0,
               "qualifying_box_removed_by_filter": 0,
               "full_256_has_no_qualifying_box": 0}
    for row in rows:
        if row["scores"]["protected"]["box_iou"] > .25:
            continue
        oracle = row["oracle"]
        if oracle["after_filter"]["32"] > .25:
            failure["correctable_within_legal_top32"] += 1
        elif oracle["after_filter"]["256"] > .25:
            failure["correct_box_only_outside_legal_top32"] += 1
        elif oracle["before_filter"]["256"] > .25:
            failure["qualifying_box_removed_by_filter"] += 1
        else:
            failure["full_256_has_no_qualifying_box"] += 1
    assert sum(failure.values()) == len(rows) - breakdown["overall"]["scores"]["protected"]["rec_hits025"]
    good_box = [r for r in rows if r["legal_box_oracle"]["box_iou"] > .5]
    selected_good_box = [r for r in rows if r["scores"]["protected"]["box_iou"] > .5]
    masks = {
        "legal_box_oracle_over050_rows": len(good_box),
        "mask_iou_sum_at_legal_box_oracle": sum(r["legal_box_oracle"]["mask_iou"] for r in good_box),
        "good_box_but_full_mask_oracle_below050": sum(r["full_mask_oracle"]["mask_iou"] <= .5 for r in good_box),
        "selected_box_over050_rows": len(selected_good_box),
        "selected_mask_iou_sum_given_good_selected_box": sum(r["scores"]["protected"]["mask_iou"] for r in selected_good_box),
        "full_mask_oracle_iou_sum_all_rows": sum(r["full_mask_oracle"]["mask_iou"] for r in rows),
        "full_mask_oracle_hits025": sum(r["full_mask_oracle"]["mask_iou"] > .25 for r in rows),
        "full_mask_oracle_hits050": sum(r["full_mask_oracle"]["mask_iou"] > .5 for r in rows),
        "is_instance_identity_classification": False,
    }
    availability = {name: sum(r["object_availability_proxy"][name] for r in rows)
                    for name in ["detector_objects", "full_256", "full_legal", "target_top32"]}
    availability["is_text_anchor_ground_truth"] = False
    return {"groups": breakdown, "scenes": {name: metrics(members) for name, members in scenes.items()},
            "coverage": coverage, "protected_rec_failures025": failure,
            "zero_legal_rows": sum(r["legal_queries"] == 0 for r in rows),
            "fewer_than_32_legal_rows": sum(r["legal_queries"] < 32 for r in rows),
            "mask_conditioned_diagnostics": masks, "object_availability_proxy": availability}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.run / "receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["protected_evaluator_row_parity"]
    raw = (args.run / "holdout_rows.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt["holdout_rows_sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == len({r["id"] for r in rows}) == receipt["census"]["holdout"]["rows"]
    result = summarize(rows)
    assert result["groups"]["overall"]["scores"]["pair"]["rec_hits025"] == receipt["summary"]["pair"]["rec_hits025"]
    result.update(schema="mcln-p2-completed-diagnostics-v1", formal_validation=False,
                  holdout_rows_sha256=receipt["holdout_rows_sha256"],
                  decision=receipt["decision"])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({name: value for name, value in result.items() if name != "scenes"},
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
