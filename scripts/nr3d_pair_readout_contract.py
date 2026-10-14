"""Fixed split, positive-only objective and paired decision for P2 v1."""

import hashlib

import torch


SALT = "MCLN-NR3D-PAIR-READOUT-V1-20260905"
CONTRACT = {
    "schema": "mcln-nr3d-pair-readout-train-v1",
    "split_salt": SALT,
    "holdout_fold": 0,
    "folds": 5,
    "epochs": 1,
    "batch_size": 4,
    "num_workers": 4,
    "init_seed": 9,
    "loader_seed": 0,
    "eval_seed": 1000,
    "top_k": 32,
    "backbone_frozen": True,
    "augmentation": False,
    "optimizer": "AdamW",
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "scheduler": None,
    "loss": "IoU-weighted listwise CE over valid candidates with root IoU > .25",
    "uncovered_rows": "excluded from ranking loss; retained as evaluation misses",
    "empty_supervision_batch": "skip both optimizer updates",
    "decision_score": "single uncalibrated scalar logit",
    "checkpoint_selection": "last step only; no intermediate holdout",
    "formal_validation": False,
    "whole_system_unseen_scenes": False,
}


def split_rows(rows):
    result = {"fit": [], "holdout": []}
    for index, row in enumerate(rows):
        fold = int(hashlib.sha256(
            (SALT + "\0" + row["scan_id"]).encode("utf-8")
        ).hexdigest()[:8], 16) % 5
        result["holdout" if fold == 0 else "fit"].append(index)
    return result


def covered_ranking_loss(logits, ious, valid):
    """Return the mean supervised-row loss; uncovered rows have zero gradient."""
    positive = valid & (ious > .25)
    covered = positive.any(dim=1)
    count = int(covered.sum())
    if count == 0:
        return logits[valid].sum() * 0.0, count
    log_probs = logits[covered].masked_fill(~valid[covered], -float("inf")).log_softmax(1)
    weights = ious[covered].masked_fill(~positive[covered], 0.0)
    weights = weights / weights.sum(dim=1, keepdim=True)
    log_probs = log_probs.masked_fill(~positive[covered], 0.0)
    return -(weights * log_probs).sum(dim=1).mean(), count


def compare_rows(rows, reference, mode="pair"):
    groups = {
        "overall": rows,
        "long_13plus": [r for r in rows if r["raw_token_count"] >= 13],
        "hard_2plus_distractors": [r for r in rows if r["distractor_count"] >= 2],
    }
    report = {}
    for group, members in groups.items():
        values = {"rows": len(members)}
        for name, threshold in [("025", .25), ("050", .5)]:
            old = [r["scores"][reference]["box_iou"] > threshold for r in members]
            new = [r["scores"][mode]["box_iou"] > threshold for r in members]
            fixes = sum(not a and b for a, b in zip(old, new))
            breaks = sum(a and not b for a, b in zip(old, new))
            values[name] = {"reference_hits": sum(old), "new_hits": sum(new),
                            "fixes": fixes, "breaks": breaks, "delta_hits": fixes - breaks}
        report[group] = values
    return report


def decide(rows):
    control = compare_rows(rows, "global")
    parent = compare_rows(rows, "protected")
    mechanism = (control["overall"]["025"]["delta_hits"] > 0
                 and control["overall"]["050"]["delta_hits"] >= 0
                 and control["long_13plus"]["025"]["delta_hits"] > 0
                 and control["hard_2plus_distractors"]["025"]["delta_hits"] > 0)
    practical = (parent["overall"]["025"]["delta_hits"] > 0
                 and parent["overall"]["050"]["delta_hits"] >= 0
                 and sum(r["scores"]["pair"]["mask_iou"] for r in rows)
                 >= sum(r["protected_mask_iou"] for r in rows))
    return {"pair_vs_global": control, "pair_vs_protected": parent,
            "mechanism_screen_pass": mechanism, "practical_screen_pass": practical,
            "eligible_for_decoder_experiment": mechanism and practical,
            "formal_promotion": False}
