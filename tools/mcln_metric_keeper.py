"""Preserve a semantic-metric best independently of a running trainer."""

import argparse
import json
import os
from pathlib import Path
import re
import time


SAMPLE_COUNTS = {"scanrefer": 9508, "nr3d": 7899, "sr3d": 17726}


def completed_rows(text, dataset):
    rows = {}
    epoch = None
    for line in text.splitlines():
        match = re.search(r"epoch\s+(\d+), total time", line)
        if match:
            epoch = int(match.group(1))
        if epoch is None:
            continue
        match = re.search(
            r"last_ semantic alignment Acc0\.(25|50): Top-1: ([0-9.]+)", line
        )
        if match:
            row = rows.setdefault(epoch, {"epoch": epoch, "groups": {}})
            row["acc" + match[1]] = float(match[2])
        match = re.search(r"\]: (unique|multi)(50)? ([0-9.eE+-]+)", line)
        if match and epoch in rows:
            key = match[1] + (match[2] or "25")
            rows[epoch]["groups"].setdefault(key, float(match[3]))
    complete = []
    for row in rows.values():
        keys = ["multi25", "multi50"]
        if dataset == "scanrefer":
            keys += ["unique25", "unique50"]
        if ("acc25" not in row or "acc50" not in row
                or not all(key in row["groups"] for key in keys)):
            continue
        result = {
            "dataset": dataset, "epoch": row["epoch"],
            "sample_count": SAMPLE_COUNTS[dataset],
        }
        for threshold in ("25", "50"):
            groups = row["groups"]
            if dataset == "scanrefer":
                hits = round(groups["unique" + threshold] * 1419)
                hits += round(groups["multi" + threshold] * 8089)
            else:
                hits = round(groups["multi" + threshold] * result["sample_count"])
            if abs(hits / result["sample_count"] - row["acc" + threshold]) > 0.000006:
                raise ValueError("semantic subgroup and overall metrics disagree")
            result["hits0" + threshold] = hits
        complete.append(result)
    return complete


def preserve_best(run_dir, log, dataset):
    import torch

    run_dir, log = Path(run_dir), Path(log)
    rows = completed_rows(log.read_text(errors="replace"), dataset)
    if not rows:
        return {"state": "waiting_for_complete_validation", "dataset": dataset}
    order = ("hits050", "hits025") if dataset == "scanrefer" else (
        "hits025", "hits050"
    )
    best = max(rows, key=lambda row: (row[order[0]], row[order[1]], -row["epoch"]))
    manifest_path = run_dir / "semantic_best.json"
    target = run_dir / "semantic_best.pth"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["epoch"] == best["epoch"]:
            if not target.exists():
                raise FileNotFoundError(target)
            return dict(saved, state="unchanged")
    filename = "ckpt_epoch_{}.pth".format(best["epoch"])
    candidates = [
        path for path in (
            run_dir / filename, log.parent / "preserved_checkpoints" / filename
        ) if path.exists()
    ]
    if not candidates:
        raise FileNotFoundError("best semantic epoch has no surviving weight: " + filename)
    source = candidates[0]
    checkpoint = torch.load(source, map_location="cpu")
    if checkpoint["epoch"] != best["epoch"]:
        raise ValueError("checkpoint own epoch does not match semantic best")
    best.update(
        model_tensors=len(checkpoint["model"]),
        has_optimizer="optimizer" in checkpoint,
        has_scheduler="scheduler" in checkpoint,
        source=str(source),
        checkpoint=str(target),
        primary_metric=order[0],
        updated_unix=time.time(),
    )
    temporary = run_dir / "semantic_best_next.pth"
    os.link(str(source), str(temporary))
    os.replace(str(temporary), str(target))
    manifest_next = run_dir / "semantic_best_next.json"
    manifest_next.write_text(json.dumps(best, sort_keys=True) + "\n")
    os.replace(str(manifest_next), str(manifest_path))
    return dict(best, state="updated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(SAMPLE_COUNTS))
    parser.add_argument("--interval", type=int, default=240)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        result = preserve_best(args.run_dir, args.log, args.dataset)
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
