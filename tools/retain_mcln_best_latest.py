"""Retain current-run semantic best and latest resumable checkpoint."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

from mcln_metric_keeper import completed_rows


ROOT = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
RUNS = {"nr3d": "1793553113", "sr3d": "1793552465"}


def retain(dataset, apply):
    import torch

    torch.set_num_threads(1)
    if dataset == "scanrefer":
        stage = ROOT / "scanrefer_restart_20261102"
        histories = list(stage.glob("stage_b/scanrefer/scanrefer_stage_b_20261102/*/stage_b_semantic_history.json"))
        if not histories:
            return {"state": "waiting_for_stage_b_validation", "dataset": dataset}
        history, = histories
        run_dir = directory = history.parent
        rows = json.loads(history.read_text())["rows"]
        preserved = directory / "preserved_checkpoints"
        tensor_count = 1168
    else:
        directory = ROOT / "runs" / dataset
        run_dir = directory / dataset / f"mcln_pvground_{dataset}_20261101" / RUNS[dataset]
        preserved = directory / "preserved_checkpoints"
        rows = completed_rows((directory / "train.log").read_text(), dataset)
        tensor_count = 1134
    latest = max(row["epoch"] for row in rows)
    order = ("hits050", "hits025") if dataset == "scanrefer" else ("hits025", "hits050")
    best = max(rows, key=lambda row: (row[order[0]], row[order[1]], -row["epoch"]))
    manifest = json.loads((run_dir / "semantic_best.json").read_text())
    if manifest["epoch"] != best["epoch"]:
        return {"state": "waiting_for_metric_keeper", "dataset": dataset}
    latest_source = run_dir / f"ckpt_epoch_{latest}.pth"
    if not latest_source.exists():
        return {"state": "waiting_for_checkpoint_save", "dataset": dataset}
    config = json.loads((run_dir / "config.json").read_text())
    if dataset == "scanrefer":
        assert config["model_only_initialization"] and config["use_pvground_innovations"]
        assert not config["pvground_train_only"]
    else:
        assert config["checkpoint_path"] is None and config["pp_checkpoint"] is None
        assert not config["use_pvground_innovations"]
    keep = {best["epoch"], latest}
    candidates = []
    for path in [*preserved.glob("ckpt_epoch_*.pth"), *run_dir.glob("ckpt_epoch_[0-9]*.pth")]:
        assert path.parent.resolve() in {preserved.resolve(), run_dir.resolve()}
        assert not path.is_symlink()
        match = re.fullmatch(r"ckpt_epoch_(\d+)\.pth", path.name)
        assert match
        epoch = int(match[1])
        if epoch < latest and epoch not in keep:
            stat = path.stat()
            candidates.append({
                "path": str(path), "epoch": epoch, "inode": stat.st_ino,
                "bytes": stat.st_size, "links": stat.st_nlink,
            })
    previous = directory / "best_latest_retention.json"
    if apply and previous.exists():
        saved = json.loads(previous.read_text())
        if saved["latest_epoch"] == latest and not candidates:
            return {"state": "unchanged", "dataset": dataset, "latest_epoch": latest}
    audits = []
    for epoch in sorted(keep):
        source = latest_source if epoch == latest else run_dir / "semantic_best.pth"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == epoch and len(checkpoint["model"]) == tensor_count
        assert "optimizer" in checkpoint and "scheduler" in checkpoint
        assert checkpoint["optimizer"]["state"]
        target = preserved / f"ckpt_epoch_{epoch}.pth"
        if apply and not target.exists():
            os.link(source, target)
        if target.exists():
            assert os.path.samefile(source, target)
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        audits.append({
            "epoch": epoch, "checkpoint": str(target), "sha256": digest.hexdigest(),
            "has_optimizer": True, "has_scheduler": True, "model_tensors": tensor_count,
        })
        del checkpoint
    free_before = shutil.disk_usage(directory).free
    assert not torch.cuda.is_initialized()
    # Remove only exact older epoch files, after auditing both retained checkpoints.
    for row in candidates:
        path = Path(row["path"])
        assert path.resolve().parent in {preserved.resolve(), run_dir.resolve()}
        assert path.stat().st_ino == row["inode"]
        if apply:
            path.unlink()
    receipt = {
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "state": "applied" if apply else "plan", "dataset": dataset,
        "run": run_dir.name, "best_epoch": best["epoch"], "latest_epoch": latest,
        "keep_epochs": sorted(keep), "audits": audits,
        "removed" if apply else "candidates": candidates,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(directory).free,
        "cuda_initialized": False,
        "policy": f"semantic {order[0]} first best plus latest; preserve all metrics and logs",
    }
    if apply:
        previous.write_text(json.dumps(receipt, indent=2) + "\n")
        with (directory / "best_latest_retention_history.jsonl").open("a") as stream:
            stream.write(json.dumps(receipt) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=(*RUNS, "scanrefer"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        print(json.dumps(retain(args.dataset, args.apply)), flush=True)
        if not args.watch:
            return
        time.sleep(240)


if __name__ == "__main__":
    main()
