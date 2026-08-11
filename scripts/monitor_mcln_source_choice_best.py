#!/usr/bin/env python
"""Preserve best MCLN source-choice checkpoints while pruning epoch files."""

import argparse
import os
import re
import shutil
import time
from pathlib import Path


def hardlink_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def find_run_dir(log_root, exp):
    base = Path(log_root) / "scanrefer" / exp
    configs = sorted(base.glob("*/config.json"))
    if not configs:
        return None
    return configs[-1].parent


def parse_records(log_path):
    records = []
    current = None
    if not log_path.exists():
        return records
    source_re = re.compile(
        r"(fixed_[A-Za-z0-9_]+|learned_selector|oracle) "
        r"Acc0\.25 Top-1: ([0-9.]+), Acc0\.50 Top-1: ([0-9.]+)"
    )
    for line in log_path.read_text(errors="replace").splitlines():
        m = re.search(r"epoch (\d+), total time ([0-9.]+)", line)
        if m:
            current = {"epoch": int(m.group(1))}
            continue
        if current is None:
            continue
        m = source_re.search(line)
        if not m:
            continue
        source = m.group(1)
        current[source + "_25"] = float(m.group(2))
        current[source + "_50"] = float(m.group(3))
        if source == "oracle":
            records.append(current)
            current = None
    return records


def preserve_best(
        run_dir, preserve_dir, keep_latest,
        baseline_acc025=-1.0, baseline_acc050=-1.0,
        name_prefix="mcln_contrastive_text"):
    records = parse_records(run_dir / "log.txt")
    if not records:
        return None
    preserve_dir.mkdir(parents=True, exist_ok=True)
    best25 = max(records, key=lambda row: row.get("learned_selector_25", -1.0))
    best50 = max(records, key=lambda row: row.get("learned_selector_50", -1.0))
    keep_epochs = set()
    preserved_paths = set()

    def checkpoint_for(row):
        return run_dir / "ckpt_epoch_{}.pth".format(int(row["epoch"]))

    def preserve_row(tag, row, metric, prefix="best"):
        epoch = int(row["epoch"])
        ckpt = checkpoint_for(row)
        if not ckpt.exists():
            return False
        baseline = baseline_acc025 if tag == "acc025" else baseline_acc050
        if row.get(metric, -1.0) <= baseline:
            return False
        dst = preserve_dir / (
            "{}_{}_{}_epoch{}_{}.pth".format(
                name_prefix, prefix, tag, epoch,
                "{:.5f}".format(row.get(metric, -1.0))
            )
        )
        hardlink_or_copy(ckpt, dst)
        preserved_paths.add(dst)
        keep_epochs.add(epoch)
        return True

    for tag, row, metric in (
        ("acc025", best25, "learned_selector_25"),
        ("acc050", best50, "learned_selector_50"),
    ):
        if not preserve_row(tag, row, metric):
            available_rows = [r for r in records if checkpoint_for(r).exists()]
            if available_rows:
                available_best = max(
                    available_rows, key=lambda r: r.get(metric, -1.0)
                )
                preserve_row(
                    tag, available_best, metric, prefix="best_available"
                )

    validated_epochs = sorted(int(row["epoch"]) for row in records)
    latest_validated_epoch = validated_epochs[-1]
    for ckpt in run_dir.glob("ckpt_epoch_*.pth"):
        if ckpt.name == "ckpt_epoch_last.pth":
            continue
        m = re.match(r"ckpt_epoch_(\d+)\.pth", ckpt.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if epoch > latest_validated_epoch:
            continue
        if epoch not in keep_epochs:
            ckpt.unlink()
    for preserved in preserve_dir.glob("{}_*.pth".format(name_prefix)):
        if preserved not in preserved_paths:
            preserved.unlink()
    return {
        "best25_epoch": best25["epoch"],
        "best25": best25.get("learned_selector_25"),
        "best25_acc50": best25.get("learned_selector_50"),
        "best50_epoch": best50["epoch"],
        "best50_acc25": best50.get("learned_selector_25"),
        "best50": best50.get("learned_selector_50"),
        "records": len(records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--preserve-dir", required=True)
    parser.add_argument("--interval", type=int, default=90)
    parser.add_argument("--keep-latest", type=int, default=2)
    parser.add_argument("--baseline-acc025", type=float, default=-1.0)
    parser.add_argument("--baseline-acc050", type=float, default=-1.0)
    parser.add_argument("--name-prefix", default="mcln_contrastive_text")
    args = parser.parse_args()

    run_dir = None
    last_report = None
    while True:
        if run_dir is None:
            run_dir = find_run_dir(args.log_root, args.exp)
        if run_dir is not None:
            report = preserve_best(
                run_dir, Path(args.preserve_dir), args.keep_latest,
                baseline_acc025=args.baseline_acc025,
                baseline_acc050=args.baseline_acc050,
                name_prefix=args.name_prefix
            )
            if report and report != last_report:
                print(report, flush=True)
                last_report = report
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
