"""GT-only representational bound for binary masks constant on superpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def optimal_superpoint_mask_iou(labels, target_mask):
    """Best IoU of any union of occupied superpoints, using GT for diagnosis.

    For each superpoint let a/b be its target/background point counts. Adding
    that superpoint improves an existing IoU r exactly when a > r*b. Therefore
    an optimum is a prefix ordered by a/(a+b). This is not a deployed threshold.
    """
    _, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    positives = np.bincount(inverse, weights=target_mask)
    assert target_mask.sum() > 0
    order = np.argsort(-(positives / counts))
    intersections = np.cumsum(positives[order])
    unions = target_mask.sum() + np.cumsum(counts[order] - positives[order])
    return float(np.max(intersections / unions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.results / "receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["native_row_parity"]
    raw = (args.results / "rows.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt["rows_sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    os.chdir(str(args.source))
    sys.path.insert(0, str(args.source))
    from src.joint_det_dataset import unpickle_data
    import torch
    data = Path("/root/autodl-tmp/DATA_ROOT")
    scans = list(unpickle_data(str(data / "val_v3scans.pkl")))[0]
    labels = {scene: np.asarray(torch.load(str(data / "superpoints/val" / (scene + "_superpoint.pth"))))
              for scene in sorted({row["scan_id"] for row in rows})}
    objects = {}
    result_rows = []
    for row in rows:
        key = (row["scan_id"], row["target_id"])
        if key not in objects:
            scan = scans[key[0]]
            target_mask = np.zeros(len(scan.orig_pc), dtype=bool)
            target_mask[scan.three_d_objects[key[1]]["points"]] = True
            assert len(labels[key[0]]) == len(target_mask)
            objects[key] = {"input_target_points": int(target_mask.sum()),
                            "superpoint_mask_oracle_iou": optimal_superpoint_mask_iou(labels[key[0]], target_mask)}
        item = objects[key]
        assert item["input_target_points"] == row["root_target_input_points"]
        assert row["mask_oracle_all_queries"]["mask_iou"] <= item["superpoint_mask_oracle_iou"] + 1e-12
        result_rows.append(dict(item, id=row["id"], scan_id=key[0], target_id=key[1]))
    values = [item["superpoint_mask_oracle_iou"] for item in result_rows]
    result = {"schema": "mcln-nr3d-superpoint-representation-oracle-v1", "rows": result_rows,
              "summary": {"sample_count": len(rows), "unique_scene_targets": len(objects),
                          "hits025": sum(value > .25 for value in values),
                          "hits050": sum(value > .5 for value in values),
                          "miou": sum(values) / len(values)},
              "query_mask_oracle_upper_bound_verified": True,
              "row_target_point_counts_match": True, "model_forwards": 0,
              "rows_sha256": receipt["rows_sha256"],
              "interpretation": "GT-only best union of current superpoints; not a deployable segmentation result."}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
