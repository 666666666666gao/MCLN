set -e
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

epoch_pattern = re.compile(r"\bepoch (\d+), total time ")
score_pattern = re.compile(
    r"last_ (position|semantic) alignment Acc0\.(25|50): "
    r"Top-1: ([0-9.]+), Top-5: ([0-9.]+), Top-10: ([0-9.]+)")
result = dict(observed_at=datetime.now(timezone.utc).isoformat(),
              diagnostic="separate_score_path_log_totals",
              paired_sample_agreement_measured=False, runs=[])
for dataset, run_id, expected in (("nr3d", "1793419199", 7899),
                                   ("sr3d", "1793419200", 17726)):
    root = Path(f"/root/autodl-tmp/pvground_{dataset}_module_training_20261030")
    run = root / f"logs/{dataset}/{dataset}_module_ft/{run_id}"
    if not (run / "config.json").exists():
        continue
    epoch, scores = 0, {}
    log = root / "train.log"
    with log.open() as stream:
        for line in stream:
            match = epoch_pattern.search(line)
            if match:
                epoch = int(match[1])
            match = score_pattern.search(line)
            if match:
                key = (epoch, match[1], match[2])
                assert key not in scores
                scores[key] = [float(match[i]) for i in (3, 4, 5)]
    metrics = list(map(json.loads, (run / "native_metrics.jsonl").read_text().splitlines()))
    rows = []
    for native in metrics:
        assert native["rows"] == expected and native["primary_score"] == "last/bbf" and native["same_complete_model"]
        row = dict(epoch=native["epoch"], rows=expected, paths={})
        for path in ("position", "semantic"):
            values = {}
            for suffix, hit_key in (("25", "hits025"), ("50", "hits050")):
                top1, top5, top10 = scores[native["epoch"], path, suffix]
                assert 0 <= top1 <= top5 <= top10 <= 1
                hits = round(top1 * expected)
                assert abs(hits / expected - top1) <= .00000501
                if path == "semantic":
                    assert hits == native[hit_key]
                values["acc" + suffix] = dict(
                    top1_logged=top1, top5_logged=top5, top10_logged=top10,
                    hits_top1_reconstructed=hits, top1_exact_percent=100 * hits / expected)
            row["paths"][path] = values
        rows.append(row)
    result["runs"].append(dict(dataset=dataset, run=run_id,
        log_path=str(log), metrics_path=str(run / "native_metrics.jsonl"), epochs=rows))
output = Path("/tmp/pvground_current_score_paths.json")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
print("SHA256", hashlib.sha256(output.read_bytes()).hexdigest())
PY
