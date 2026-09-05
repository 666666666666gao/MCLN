"""Independent arithmetic, source and CSV-identity check of a sealed audit."""

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    manifest_path = root / "input_manifest_v2.json"
    manifest = json.loads(manifest_path.read_text())
    results = root / "results"
    receipt = json.loads((results / "receipt.json").read_text())
    assert (root / "controller.exit").read_text().strip() == "0"
    assert receipt["status"] == "complete" and receipt["native_row_parity"]
    assert receipt["manifest_sha256"] == sha(manifest_path)
    for relative, expected in manifest["files"].items():
        assert sha(root / relative) == expected, relative
    source = Path(manifest["model_source"])
    source_manifest = source / "g0_source_manifest.json"
    assert sha(source_manifest) == manifest["source_manifest_sha256"]
    for relative, expected in json.loads(source_manifest.read_text())["files"].items():
        assert sha(source / relative) == expected, relative
    assert sha(Path(manifest["checkpoint"])) == manifest["checkpoint_sha256"]
    assert sha(results / "rows.jsonl") == receipt["rows_sha256"]
    rows = [json.loads(line) for line in (results / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 7899 and [row["id"] for row in rows] == list(range(7899))
    data = Path("/root/autodl-tmp/DATA_ROOT")
    scenes = set(ast.literal_eval((source / "data/meta_data/nr3d_test_scans.txt").read_text()))
    with (data / "refer_it_3d/nr3d.csv").open() as stream:
        expected_rows = [row for row in csv.DictReader(stream)
                         if row["scan_id"] in scenes and row["correct_guess"].lower() == "true"
                         and (data / "superpoints/val" / (row["scan_id"] + "_superpoint.pth")).exists()]
    assert len(expected_rows) == len(rows)
    assert all(row["scan_id"] == expected["scan_id"] and row["target_id"] == int(expected["target_id"])
               for row, expected in zip(rows, expected_rows))
    values = [row["rec_selection"]["box_iou"] if row["rec_selection"] is not None else 0.0 for row in rows]
    masks = [row["mask_selection"]["mask_iou"] for row in rows]
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in values + masks)
    native = receipt["native_metrics"]
    for suffix, threshold in (("025", .25), ("050", .5)):
        assert sum(value > threshold for value in values) == receipt["summary"]["rec_hits" + suffix]
        assert sum(value > threshold for value in values) == sum(group["hits" + suffix] for group in native["position_subgroups"].values())
        assert sum(value > threshold for value in masks) == native["mask"]["hits" + suffix]
        assert sum(row["mask_selection"]["box_iou"] > threshold for row in rows) == native["position"]["learned_selector"]["hits" + suffix]
    assert math.isclose(sum(masks), native["mask"]["iou_sum"], rel_tol=0, abs_tol=1e-8)
    verification = {"status": "pass", "rows": len(rows), "scenes": len({row["scan_id"] for row in rows}),
                    "csv_identity_order_exact": True, "native_metric_arithmetic_exact": True,
                    "frozen_source_and_checkpoint_hashes_match": True,
                    "rows_sha256": receipt["rows_sha256"], "historical_metric_match": receipt["historical_metric_match"],
                    "verification_model_forwards": 0, "verification_optimizer_steps": 0}
    with (results / "independent_verification.json").open("x") as stream:
        json.dump(verification, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
