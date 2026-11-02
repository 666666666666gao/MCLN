"""Inventory current NR/SR checkpoint storage and propose retention; never delete."""

import argparse
import datetime
import json
from pathlib import Path
import re
import shutil


def audit(root, dataset):
    run = {"nr3d": "1793553113", "sr3d": "1793552465"}[dataset]
    directory = root / "runs" / dataset
    run_dir = directory / dataset / ("mcln_pvground_" + dataset + "_20261101") / run
    best = json.loads((run_dir / "semantic_best.json").read_text())
    preserved = directory / "preserved_checkpoints"
    checkpoints = []
    for path in preserved.glob("ckpt_epoch_*.pth"):
        match = re.fullmatch(r"ckpt_epoch_(\d+)\.pth", path.name)
        assert match
        stat = path.stat()
        checkpoints.append({
            "epoch": int(match[1]), "path": str(path),
            "bytes": stat.st_size, "allocated_bytes": stat.st_blocks * 512,
            "inode": stat.st_ino, "links": stat.st_nlink,
        })
    checkpoints.sort(key=lambda row: row["epoch"])
    latest = checkpoints[-1]["epoch"]
    keep_epochs = {best["epoch"], latest, latest - 1}
    candidates = [row for row in checkpoints if row["epoch"] not in keep_epochs]
    free = shutil.disk_usage(root).free
    return {
        "audited_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": dataset, "run": run, "root": str(root),
        "free_bytes": free, "checkpoint_bytes": checkpoints[-1]["bytes"],
        "additional_same_size_checkpoints_fit": free // checkpoints[-1]["bytes"],
        "best": best, "latest_preserved_epoch": latest,
        "proposed_keep_epochs": sorted(keep_epochs),
        "checkpoints": checkpoints, "proposed_archive_or_remove": candidates,
        "candidate_count": len(candidates),
        "candidate_logical_bytes": sum(row["bytes"] for row in candidates),
        "freeable_bytes_if_single_link_files_removed": sum(
            row["allocated_bytes"] for row in candidates if row["links"] == 1
        ),
        "policy": "proposal only: retain semantic best plus latest two; retain all epoch metrics",
        "files_removed": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("nr3d", "sr3d"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.dataset)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
