"""Join raw CSV token counts and the superpoint bound to sealed audit rows."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

from scripts.summarize_nr3d_official_candidate_audit import describe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    raw = (args.results / "rows.jsonl").read_bytes()
    receipt = json.loads((args.results / "receipt.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == receipt["rows_sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    csv_path = Path("/root/autodl-tmp/DATA_ROOT/refer_it_3d/nr3d.csv")
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == "5de4f1b47130803c88f7c57903e7b6df5473f1b903c32cf28d06fa9c25996a67"
    scenes = {row["scan_id"] for row in rows}
    with csv_path.open() as stream:
        csv_rows = [row for row in csv.DictReader(stream)
                    if row["scan_id"] in scenes and row["correct_guess"].lower() == "true"]
    assert len(csv_rows) == len(rows) == 7899
    metadata = []
    groups = {"2_to_6": [], "7_to_8": [], "9_to_12": [], "13plus": []}
    for row, csv_row in zip(rows, csv_rows):
        assert row["scan_id"] == csv_row["scan_id"] and row["target_id"] == int(csv_row["target_id"])
        count = len(ast.literal_eval(csv_row["tokens"]))
        key = ("2_to_6" if count <= 6 else "7_to_8" if count <= 8 else
               "9_to_12" if count <= 12 else "13plus")
        groups[key].append(row)
        metadata.append({"id": row["id"], "csv_token_count": count,
                         "normalized_whitespace_token_count": row["raw_token_count"]})
    superpoints = json.loads((args.results / "superpoint_oracle.json").read_text())
    assert superpoints["rows_sha256"] == receipt["rows_sha256"]
    mask_bad = [row for row in rows if row["box_oracle_after_filter"] is not None
                and row["box_oracle_after_filter"]["box_iou"] > .5
                and row["mask_oracle_all_queries"]["mask_iou"] <= .5]
    common = [row for row in rows if row["rec_selection"] is not None]
    path_comparison = {"common_rows": len(common), "thresholds": {}}
    for suffix, threshold in (("025", .25), ("050", .5)):
        before = [row["mask_selection"]["mask_iou"] > threshold for row in common]
        after = [row["rec_selection"]["mask_iou"] > threshold for row in common]
        path_comparison["thresholds"][suffix] = {
            "native_mask_hits": sum(before), "rec_query_mask_hits": sum(after),
            "fixes": sum(not a and b for a, b in zip(before, after)),
            "breaks": sum(a and not b for a, b in zip(before, after))}
    path_comparison["mask_miou_delta_percentage_points"] = 100 * sum(
        row["rec_selection"]["mask_iou"] - row["mask_selection"]["mask_iou"] for row in common) / len(common)
    result = {
        "schema": "mcln-nr3d-official-candidate-enrichment-v1", "metadata": metadata,
        "csv_token_length_groups": {key: describe(group) for key, group in groups.items()},
        "schema_correction": "Sealed rows.raw_token_count and analysis.groups.text_length count normalized annotation whitespace tokens after Scene_graph_parse, not raw CSV tokens. Use csv_token_count and csv_token_length_groups for comparison with the earlier cache table. Original rows and inference are unchanged.",
        "good_box_but_all_query_masks_le050": {
            "rows": len(mask_bad),
            "superpoint_bound_le050_rows": sum(superpoints["rows"][row["id"]]["superpoint_mask_oracle_iou"] <= .5 for row in mask_bad),
            "superpoint_bound_gt050_rows": sum(superpoints["rows"][row["id"]]["superpoint_mask_oracle_iou"] > .5 for row in mask_bad)},
        "mask_selection_path_comparison": path_comparison,
        "rows_sha256": receipt["rows_sha256"], "model_forwards": 0, "optimizer_steps": 0,
    }
    with (args.results / "enrichment.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"csv_length_counts": {key: len(group) for key, group in groups.items()},
                      "mask": result["good_box_but_all_query_masks_le050"],
                      "path_comparison": path_comparison}, indent=2))


if __name__ == "__main__":
    main()
