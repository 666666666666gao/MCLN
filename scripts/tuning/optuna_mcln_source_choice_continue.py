#!/usr/bin/env python
"""Optuna continuation tuning for the MCLN ScanRefer source-choice run.

The objective is the deployable REC metric from Source choice diagnostics:
learned_selector Acc@0.25, with Acc@0.50 as a secondary term.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import optuna


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_records(log_path: Path) -> list[dict]:
    lines = log_path.read_text(errors="replace").splitlines()
    current = None
    records = []
    for line in lines:
        m = re.search(r"epoch (\d+), total time ([0-9.]+)", line)
        if m:
            current = {"epoch": int(m.group(1)), "epoch_time": float(m.group(2))}
        if current is None:
            continue
        for key, pattern in {
            "last_pos25": r"last_ position alignment Acc0\.25: Top-1: ([0-9.]+)",
            "last_pos50": r"last_ position alignment Acc0\.50: Top-1: ([0-9.]+)",
            "last_sem25": r"last_ semantic alignment Acc0\.25: Top-1: ([0-9.]+)",
            "last_sem50": r"last_ semantic alignment Acc0\.50: Top-1: ([0-9.]+)",
            "overall25": r"overall25 ([0-9.]+)",
            "overall50": r"overall50 ([0-9.]+)",
        }.items():
            m = re.search(pattern, line)
            if m:
                current[key] = float(m.group(1))
        m = re.search(
            r"(fixed_default|fixed_mask_text|learned_selector|oracle) "
            r"Acc0\.25 Top-1: ([0-9.]+), Acc0\.50 Top-1: ([0-9.]+)",
            line,
        )
        if m:
            source, acc25, acc50 = m.group(1), float(m.group(2)), float(m.group(3))
            current[source + "_25"] = acc25
            current[source + "_50"] = acc50
            if source == "oracle":
                records.append(current)
                current = None
    return records


def parse_last_train(log_path: Path):
    last_train = None
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors="replace").splitlines():
        m = re.search(r"Train: \[(\d+)\]\[(\d+)/(\d+)\]", line)
        if m:
            last_train = tuple(map(int, m.groups()))
    return last_train


def find_single_run_dir(log_root: Path, exp: str) -> Path | None:
    candidates = sorted((log_root / "scanrefer" / exp).glob("*/config.json"))
    if len(candidates) != 1:
        return None
    return candidates[0].parent


def checkpoint_epoch(path: Path) -> int:
    import torch

    ckpt = torch.load(path, map_location="cpu")
    return int(ckpt["epoch"])


def hardlink_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def trial_params(trial: optuna.trial.Trial) -> dict:
    return {
        "base_checkpoint": trial.suggest_categorical(
            "base_checkpoint", ["acc25_epoch70", "acc50_available_epoch68"]
        ),
        "train_epochs": trial.suggest_categorical("train_epochs", [4, 6, 8]),
        "batch_size": trial.suggest_categorical("batch_size", [12, 14, 16]),
        "num_workers": trial.suggest_categorical("num_workers", [2, 4]),
        "lr": trial.suggest_categorical("lr", [1e-5, 2e-5, 3e-5, 5e-5]),
        "lr_backbone": trial.suggest_categorical(
            "lr_backbone", [1e-4, 2e-4, 5e-4]
        ),
        "text_encoder_lr": trial.suggest_categorical(
            "text_encoder_lr", [1e-6, 3e-6, 1e-5]
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [2e-4, 5e-4, 1e-3]
        ),
        "clip_norm": trial.suggest_categorical("clip_norm", [0.05, 0.1, 0.2]),
        "lr_decay_after": trial.suggest_categorical("lr_decay_after", [999, 3, 5]),
        "lr_decay_rate": trial.suggest_categorical("lr_decay_rate", [0.3, 0.5, 1.0]),
        "selector_loss_weight": trial.suggest_categorical(
            "selector_loss_weight", [0.0, 0.2, 0.5, 1.0]
        ),
        "selector_lr": trial.suggest_categorical(
            "selector_lr", [2e-4, 5e-4, 1e-3]
        ),
        "selector_min_iou_gap": trial.suggest_categorical(
            "selector_min_iou_gap", [0.03, 0.05, 0.08]
        ),
        "small_lr": trial.suggest_categorical("small_lr", [False, True]),
    }


def objective_from_metrics(acc25, acc50) -> float:
    if acc25 is None:
        return -1.0
    if acc50 is None:
        acc50 = 0.0
    return 0.8 * float(acc25) + 0.2 * float(acc50)


def command_for_trial(args, params, trial_number: int, ckpt_path: Path, ckpt_epoch: int):
    exp = f"trial_{trial_number:04d}"
    max_epoch = ckpt_epoch + int(params["train_epochs"])
    decay_after = int(params["lr_decay_after"])
    if decay_after >= int(params["train_epochs"]):
        decay_epoch = 999
    else:
        decay_epoch = max(1, decay_after)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node",
        "1",
        "--master_port",
        str(args.master_port_base + (trial_number % 1000)),
        "train_dist_mod.py",
        "--num_decoder_layers",
        "6",
        "--use_color",
        "--weight_decay",
        str(params["weight_decay"]),
        "--data_root",
        args.data_root,
        "--val_freq",
        "1",
        "--batch_size",
        str(params["batch_size"]),
        "--save_freq",
        "1",
        "--print_freq",
        "400",
        "--lr_backbone",
        str(params["lr_backbone"]),
        "--lr",
        str(params["lr"]),
        "--text_encoder_lr",
        str(params["text_encoder_lr"]),
        "--dataset",
        "scanrefer",
        "--test_dataset",
        "scanrefer",
        "--detect_intermediate",
        "--joint_det",
        "--use_soft_token_loss",
        "--use_contrastive_align",
        "--log_dir",
        str(args.output_root),
        "--lr_decay_epochs",
        str(decay_epoch),
        "--lr_decay_rate",
        str(params["lr_decay_rate"]),
        "--clip_norm",
        str(params["clip_norm"]),
        "--pp_checkpoint",
        args.pp_checkpoint,
        "--butd",
        "--self_attend",
        "--augment_det",
        "--max_epoch",
        str(max_epoch),
        "--model",
        "MCLN",
        "--exp",
        exp,
        "--checkpoint_path",
        str(ckpt_path),
        "--reduce_lr",
        "--skip_missing_superpoints",
        "--use_source_choice_selector",
        "--source_choice_selector_lr",
        str(params["selector_lr"]),
        "--source_choice_selector_loss_weight",
        str(params["selector_loss_weight"]),
        "--source_choice_selector_sources",
        "default,mask_text",
        "--source_choice_selector_default_source",
        "default",
        "--source_choice_selector_choice_target",
        "precision_gain_default_sourcewise_focal_bce",
        "--source_choice_selector_min_iou_gap",
        str(params["selector_min_iou_gap"]),
        "--eval_use_selector_choice_scores",
        "--num_workers",
        str(params["num_workers"]),
        "--rng_seed",
        str(args.seed),
    ]
    if params["small_lr"]:
        cmd.append("--small_lr")
    return cmd, exp, max_epoch, decay_epoch


def cleanup_trial_checkpoints(run_dir: Path, keep_path: Path | None):
    for ckpt in run_dir.glob("ckpt_epoch_*.pth"):
        if ckpt.name == "ckpt_epoch_last.pth":
            if ckpt.exists():
                ckpt.unlink()
            continue
        if keep_path is not None and ckpt.resolve() == keep_path.resolve():
            continue
        if ckpt.exists():
            ckpt.unlink()


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_best_row(rows: list[dict]) -> dict | None:
    ok = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        try:
            objective = float(row.get("objective", "-1"))
        except ValueError:
            objective = -1.0
        if math.isfinite(objective):
            ok.append((objective, row))
    if not ok:
        return None
    return max(ok, key=lambda item: item[0])[1]


def run_trial(args, trial: optuna.trial.Trial) -> float:
    params = trial_params(trial)
    ckpt_path = Path(args.acc25_checkpoint)
    if params["base_checkpoint"] == "acc50_available_epoch68":
        ckpt_path = Path(args.acc50_checkpoint)
    ckpt_epoch = checkpoint_epoch(ckpt_path)
    cmd, exp, max_epoch, decay_epoch = command_for_trial(
        args, params, trial.number, ckpt_path, ckpt_epoch
    )
    run_root = Path(args.output_root)
    trial_dir = run_root / "trial_logs" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["OMP_NUM_THREADS"] = str(args.omp_threads)
    env["PYTHONFAULTHANDLER"] = "1"
    env["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    env["PYTHONPATH"] = (
        str(repo_root())
        + os.pathsep
        + str(repo_root() / "pointnet2")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    stdout_path = trial_dir / "trial_stdout.log"
    command_path = trial_dir / "command.txt"
    command_path.write_text(" ".join(map(str, cmd)) + "\n", encoding="utf-8")
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout:
        stdout.write(f"[trial] params={json.dumps(params, sort_keys=True)}\n")
        stdout.write(f"[trial] checkpoint={ckpt_path} epoch={ckpt_epoch}\n")
        stdout.write(f"[trial] max_epoch={max_epoch} decay_epoch={decay_epoch}\n")
        stdout.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root()),
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )

    run_dir = find_single_run_dir(Path(args.output_root), exp)
    record = {
        "trial": trial.number,
        "status": "failed",
        "return_code": proc.returncode,
        "hostname": socket.gethostname(),
        "gpu": args.gpu,
        "base_checkpoint": str(ckpt_path),
        "base_epoch": ckpt_epoch,
        "max_epoch": max_epoch,
        "lr_decay_epoch_relative": decay_epoch,
        "stdout_log": str(stdout_path),
        "run_dir": str(run_dir) if run_dir else "",
        **params,
    }
    if run_dir is None:
        record["error"] = "missing or ambiguous run_dir"
    else:
        log_path = run_dir / "log.txt"
        records = parse_records(log_path) if log_path.exists() else []
        record["log_path"] = str(log_path)
        if records:
            latest = records[-1]
            best25 = max(records, key=lambda row: row.get("learned_selector_25", -1))
            best50 = max(records, key=lambda row: row.get("learned_selector_50", -1))
            record.update(
                {
                    "latest_epoch": latest["epoch"],
                    "latest_acc25": latest.get("learned_selector_25"),
                    "latest_acc50": latest.get("learned_selector_50"),
                    "best_acc25_epoch": best25["epoch"],
                    "best_acc25": best25.get("learned_selector_25"),
                    "best_acc25_pair_acc50": best25.get("learned_selector_50"),
                    "best_acc50_epoch": best50["epoch"],
                    "best_acc50_pair_acc25": best50.get("learned_selector_25"),
                    "best_acc50": best50.get("learned_selector_50"),
                    "best_oracle25": max(
                        row.get("oracle_25", -1) for row in records
                    ),
                    "best_maskkiou25": max(
                        row.get("overall25", -1) for row in records
                    ),
                }
            )
            record["objective"] = objective_from_metrics(
                record["best_acc25"], record["best_acc25_pair_acc50"]
            )
            record["status"] = "completed" if proc.returncode == 0 else "failed"

            best_epoch = int(record["best_acc25_epoch"])
            best_ckpt = run_dir / f"ckpt_epoch_{best_epoch}.pth"
            if best_ckpt.exists():
                trial_best = run_dir / f"best_trial_acc025_epoch{best_epoch}.pth"
                hardlink_or_copy(best_ckpt, trial_best)
                record["trial_best_checkpoint"] = str(trial_best)
                cleanup_trial_checkpoints(run_dir, trial_best)
            else:
                cleanup_trial_checkpoints(run_dir, None)
                record["error"] = f"missing best ckpt {best_ckpt}"
        else:
            record["error"] = "no eval records parsed"
            cleanup_trial_checkpoints(run_dir, None)

    rows_path = Path(args.report_dir) / "trials.csv"
    rows = load_rows(rows_path)
    rows = [row for row in rows if str(row.get("trial")) != str(trial.number)]
    rows.append(record)
    write_csv(rows_path, rows)
    best_row = select_best_row(rows)
    if best_row:
        (Path(args.report_dir) / "best.json").write_text(
            json.dumps(best_row, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        best_ckpt = best_row.get("trial_best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            global_best = Path(args.output_root) / "best_optuna_rec_acc025.pth"
            if global_best.exists():
                global_best.unlink()
            hardlink_or_copy(Path(best_ckpt), global_best)

    trial.set_user_attr("record", record)
    return float(record.get("objective") or -1.0)


def enqueue_seed_trials(study: optuna.Study):
    presets = [
        {
            "base_checkpoint": "acc25_epoch70",
            "train_epochs": 4,
            "batch_size": 16,
            "num_workers": 4,
            "lr": 2e-5,
            "lr_backbone": 2e-4,
            "text_encoder_lr": 3e-6,
            "weight_decay": 5e-4,
            "clip_norm": 0.1,
            "lr_decay_after": 999,
            "lr_decay_rate": 1.0,
            "selector_loss_weight": 0.2,
            "selector_lr": 5e-4,
            "selector_min_iou_gap": 0.05,
            "small_lr": False,
        },
        {
            "base_checkpoint": "acc25_epoch70",
            "train_epochs": 6,
            "batch_size": 14,
            "num_workers": 4,
            "lr": 1e-5,
            "lr_backbone": 1e-4,
            "text_encoder_lr": 1e-6,
            "weight_decay": 1e-3,
            "clip_norm": 0.05,
            "lr_decay_after": 3,
            "lr_decay_rate": 0.5,
            "selector_loss_weight": 0.0,
            "selector_lr": 2e-4,
            "selector_min_iou_gap": 0.08,
            "small_lr": False,
        },
        {
            "base_checkpoint": "acc50_available_epoch68",
            "train_epochs": 4,
            "batch_size": 16,
            "num_workers": 4,
            "lr": 3e-5,
            "lr_backbone": 2e-4,
            "text_encoder_lr": 3e-6,
            "weight_decay": 5e-4,
            "clip_norm": 0.1,
            "lr_decay_after": 999,
            "lr_decay_rate": 1.0,
            "selector_loss_weight": 0.2,
            "selector_lr": 5e-4,
            "selector_min_iou_gap": 0.05,
            "small_lr": False,
        },
    ]
    for params in presets:
        study.enqueue_trial(params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="mcln_source_choice_continue")
    parser.add_argument("--storage", required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--data-root", default="/root/autodl-tmp/DATA_ROOT/")
    parser.add_argument(
        "--pp-checkpoint",
        default="/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth",
    )
    parser.add_argument("--acc25-checkpoint", required=True)
    parser.add_argument("--acc50-checkpoint", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--omp-threads", type=int, default=2)
    parser.add_argument("--master-port-base", type=int, default=4560)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    if args.storage.startswith("sqlite:///"):
        Path(args.storage[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )
    if len(study.trials) == 0:
        enqueue_seed_trials(study)

    if args.dry_run:
        trial = optuna.trial.FixedTrial(
            {
                "base_checkpoint": "acc25_epoch70",
                "train_epochs": 4,
                "batch_size": 16,
                "num_workers": 4,
                "lr": 2e-5,
                "lr_backbone": 2e-4,
                "text_encoder_lr": 3e-6,
                "weight_decay": 5e-4,
                "clip_norm": 0.1,
                "lr_decay_after": 999,
                "lr_decay_rate": 1.0,
                "selector_loss_weight": 0.2,
                "selector_lr": 5e-4,
                "selector_min_iou_gap": 0.05,
                "small_lr": False,
            }
        )
        params = trial_params(trial)
        ckpt = Path(args.acc25_checkpoint)
        epoch = checkpoint_epoch(ckpt)
        cmd, exp, max_epoch, decay_epoch = command_for_trial(args, params, 0, ckpt, epoch)
        print("exp", exp)
        print("checkpoint_epoch", epoch)
        print("max_epoch", max_epoch)
        print("decay_epoch_relative", decay_epoch)
        print("command", " ".join(map(str, cmd)))
        return

    study.optimize(lambda trial: run_trial(args, trial), n_trials=args.n_trials)


if __name__ == "__main__":
    main()
