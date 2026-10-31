set -e
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import json
from pathlib import Path

for dataset, run_id in (
    ("scanrefer", "1793345477"),
    ("nr3d", "1793419199"),
    ("sr3d", "1793419200"),
):
    root = Path(
        "/root/autodl-tmp/pvground_scanrefer_module_ft_20261030"
        if dataset == "scanrefer"
        else f"/root/autodl-tmp/pvground_{dataset}_module_training_20261030"
    )
    log_root = (
        root / "logs_semantic/scanrefer/scanrefer_module_ft_semantic"
        if dataset == "scanrefer"
        else root / f"logs/{dataset}/{dataset}_module_ft"
    )
    run = log_root / run_id
    if not (run / "config.json").exists():
        continue
    config = json.loads((run / "config.json").read_text())
    print("RUN", dataset, run_id, "MAX_EPOCH", config["max_epoch"])
    source = (root / "code/main_utils.py").read_text().splitlines()
    matched = set()
    for index, line in enumerate(source):
        if any(token in line for token in (
            "metric_history", "early_stop", "three_round_mean_declined_twice"
        )):
            matched.update(range(max(0, index - 4), min(len(source), index + 15)))
    for index in sorted(matched):
        print(f"{index + 1}: {source[index]}")
    rows = [
        json.loads(line)
        for line in (run / "native_metrics.jsonl").read_text().splitlines()
    ]
    trained = [row for row in rows if row["epoch"] > 0]
    streak = 0
    previous = None
    for index in range(2, len(trained)):
        window = trained[index - 2:index + 1]
        totals = tuple(sum(row[key] for row in window) for key in ("hits025", "hits050"))
        decreased = previous is not None and all(
            current < old for current, old in zip(totals, previous)
        )
        streak = streak + 1 if decreased else 0
        previous = totals
        print("RECOMPUTED", json.dumps({
            "epoch": trained[index]["epoch"],
            "three_epoch_hit_sums": totals,
            "both_decreased": decreased,
            "decline_streak": streak,
        }))
PY
