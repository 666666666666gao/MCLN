"""Extract semantic box diagnostics without importing the training environment."""

import argparse
import json
import re
from pathlib import Path


EPOCH = re.compile(r"\bepoch (\d+), total time ")
SCORES = re.compile(
    r"last_ semantic alignment Acc0\.(25|50): "
    r"Top-1: ([0-9.]+), Top-5: ([0-9.]+), Top-10: ([0-9.]+)"
)
GROUP = re.compile(
    r"\]: (easy|hard|vd|vid|unique|multi)(50)? ([0-9.eE+-]+)\s*$"
)


def collect(log_path, metrics_path, dataset):
    epochs = {}
    epoch = 0
    subgroup_threshold = None
    with Path(log_path).open(encoding="utf-8") as stream:
        for line in stream:
            match = EPOCH.search(line)
            if match:
                epoch = int(match[1])
                subgroup_threshold = None
            match = SCORES.search(line)
            if match:
                row = epochs.setdefault(epoch, {"epoch": epoch, "subgroups": {}})
                key = "acc" + match[1]
                assert key not in row, (epoch, key, "duplicate evaluation")
                row[key] = {
                    "top1": float(match[2]),
                    "top5": float(match[3]),
                    "top10": float(match[4]),
                }
                subgroup_threshold = None
            if "]: iou@0.25" in line:
                subgroup_threshold = "25"
            elif "]: iou@0.50" in line:
                subgroup_threshold = "50"
            elif "]: mask@" in line:
                subgroup_threshold = None
            match = GROUP.search(line)
            if match and subgroup_threshold is not None:
                assert epoch in epochs, (epoch, "subgroups before scores")
                suffix = match[2] or "25"
                assert suffix == subgroup_threshold, (epoch, line)
                key = match[1] + suffix
                assert key not in epochs[epoch]["subgroups"], (epoch, key)
                epochs[epoch]["subgroups"][key] = float(match[3])

    metrics = [
        json.loads(line)
        for line in Path(metrics_path).read_text(encoding="utf-8").splitlines()
    ]
    result = []
    for native in metrics:
        assert native["rows"] == {"scanrefer": 9508, "nr3d": 7899, "sr3d": 17726}[dataset]
        assert native["primary_score"] == "last/bbf"
        assert native["same_complete_model"] is True
        row = epochs[native["epoch"]]
        for suffix, hits_key in (("25", "hits025"), ("50", "hits050")):
            score = row["acc" + suffix]
            exact = native[hits_key] / native["rows"]
            # Official text logs round fractions to five decimal places.
            assert abs(score["top1"] - exact) <= 0.00000501, (native, score)
            assert score["top1"] <= score["top5"] <= score["top10"] <= 1
            score["top1_exact"] = exact
            score["top10_minus_top1_pp"] = (score["top10"] - exact) * 100
        assert len(row["subgroups"]) == 12, row
        row["native"] = native
        result.append(row)
    return {"dataset": dataset, "rows": result, "protocol": "last/bbf semantic box"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--dataset", required=True, choices=("scanrefer", "nr3d", "sr3d"))
    args = parser.parse_args()
    print(json.dumps(collect(args.log, args.metrics, args.dataset)))


if __name__ == "__main__":
    main()
