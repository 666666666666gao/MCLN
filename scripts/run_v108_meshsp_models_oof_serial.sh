#!/usr/bin/env bash
set -uo pipefail

GEOMETRY_SESSION=mcln_v108_meshsp_geometry_wait_20260814
ROOT=/tmp/mcln_repo
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
TRAIN_CACHE="$BASE/train"
GEOMETRY_CACHE="$BASE/geometry_train"
ARTIFACT_DIR="$BASE/v108_artifacts"
PARENT="$ARTIFACT_DIR/parent_h256_seed0.pth"
GEOMETRY="$ARTIFACT_DIR/geometry_h256_seed0.pth"
OOF="$BASE/v108_meshsp_pareto_oof.json"
PARENT_LOG="$BASE/v108_parent_train.log"
GEOMETRY_LOG="$BASE/v108_geometry_train.log"
OOF_LOG="$BASE/v108_meshsp_pareto_oof.log"
PIPELINE_LOG="$BASE/v108_models_oof_pipeline.log"
EXIT_FILE="$BASE/v108_models_oof_pipeline_exitcode.txt"
RECEIPT="$BASE/v108_models_oof_pipeline_receipt.json"
V99_REPORT="$ROOT/experiment_output/historical_e71_geometry/v99_pareto_contextual_hierarchical_trainonly_v1.json"
FALLBACK=/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train_missing789.txt
PY=/root/miniconda3/envs/bdetr/bin/python

finish() {
    local rc="$1"
    printf '%s\n' "$rc" >"$EXIT_FILE"
    for path in "$PIPELINE_LOG" "$EXIT_FILE"; do
        [[ ! -e "$path" ]] || chmod 0444 "$path"
    done
    exit "$rc"
}

for path in "$ARTIFACT_DIR" "$OOF" "$PARENT_LOG" "$GEOMETRY_LOG" \
            "$OOF_LOG" "$PIPELINE_LOG" "$EXIT_FILE" "$RECEIPT"; do
    if [[ -e "$path" ]]; then
        echo "V108 models/OOF output already exists: $path" >&2
        exit 64
    fi
done

exec >"$PIPELINE_LOG" 2>&1
echo "policy=single_gpu_serial_meshsp_parent_geometry_oof"
date -Is
while screen -ls 2>/dev/null | grep -Fq ".$GEOMETRY_SESSION"; do
    sleep 30
done

if [[ ! -f "$BASE/single_gpu_serial_geometry_wait_exitcode.txt" \
      || "$(tr -d '[:space:]' <"$BASE/single_gpu_serial_geometry_wait_exitcode.txt")" != "0" \
      || ! -f "$BASE/geometry_train_receipt.json" \
      || ! -f "$BASE/candidate_train_receipt.json" ]]; then
    echo "V108 candidate/geometry cache pipeline did not complete cleanly"
    finish 70
fi

"$PY" - "$BASE/candidate_train_receipt.json" \
        "$BASE/geometry_train_receipt.json" <<'PY'
import json
import sys
from pathlib import Path

candidate = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
geometry = json.loads(Path(sys.argv[2]).read_text(encoding="ascii"))
if candidate.get("schema") != "mcln-v108-meshsp-train-candidate-cache-receipt-v1":
    raise SystemExit("candidate receipt schema mismatch")
if geometry.get("schema") != "mcln-v108-meshsp-train-geometry-cache-receipt-v1":
    raise SystemExit("geometry receipt schema mismatch")
if candidate.get("validation_data_accessed") is not False:
    raise SystemExit("candidate receipt accessed validation")
if geometry.get("validation_data_accessed") is not False:
    raise SystemExit("geometry receipt accessed validation")
if candidate.get("sample_count") != 36665 or geometry.get("sample_count") != 36665:
    raise SystemExit("V108 cache sample count mismatch")
PY
rc=$?
if [[ "$rc" -ne 0 ]]; then
    echo "V108 cache receipt validation failed: rc=$rc"
    finish 71
fi

free_bytes=$(df -B1 /root/autodl-tmp | awk 'NR==2 {print $4}')
if [[ ! "$free_bytes" =~ ^[0-9]+$ || "$free_bytes" -lt 2500000000 ]]; then
    echo "insufficient free disk before V108 model fitting: $free_bytes"
    finish 72
fi

mkdir -m 0755 "$ARTIFACT_DIR"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/pointnet2"

echo "stage=parent_train"
date -Is
"$PY" scripts/train_rec_reranker.py \
    --train-cache "$TRAIN_CACHE" \
    --output "$PARENT" \
    --seed 0 \
    --hidden-dim 256 \
    --dropout 0.1 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --batch-size 256 \
    --max-epochs 100 \
    --patience 10 \
    --device cuda:0 >"$PARENT_LOG" 2>&1
rc=$?
if [[ "$rc" -ne 0 ]]; then
    echo "V108 parent training failed: rc=$rc"
    [[ ! -e "$PARENT_LOG" ]] || chmod 0444 "$PARENT_LOG"
    finish "$rc"
fi
chmod 0444 "$PARENT" "$PARENT_LOG"

echo "stage=geometry_train"
date -Is
"$PY" scripts/train_rec_geometry_reranker.py \
    --base-cache "$TRAIN_CACHE" \
    --geometry-cache "$GEOMETRY_CACHE" \
    --parent-artifact "$PARENT" \
    --output "$GEOMETRY" \
    --split-seed 0 \
    --model-seed 0 \
    --hidden-dim 256 \
    --dropout 0.1 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --batch-size 256 \
    --max-epochs 100 \
    --patience 10 \
    --device cuda:0 \
    --verbose >"$GEOMETRY_LOG" 2>&1
rc=$?
if [[ "$rc" -ne 0 ]]; then
    echo "V108 geometry training failed: rc=$rc"
    [[ ! -e "$GEOMETRY_LOG" ]] || chmod 0444 "$GEOMETRY_LOG"
    finish "$rc"
fi
chmod 0444 "$GEOMETRY" "$GEOMETRY_LOG"

echo "stage=scene_disjoint_oof"
date -Is
"$PY" scripts/run_v108_meshsp_pareto_oof.py \
    --v99-report "$V99_REPORT" \
    --base-cache "$TRAIN_CACHE" \
    --geometry-cache "$GEOMETRY_CACHE" \
    --parent-artifact "$PARENT" \
    --geometry-artifact "$GEOMETRY" \
    --fallback-scenes "$FALLBACK" \
    --output "$OOF" \
    --device cuda:0 >"$OOF_LOG" 2>&1
rc=$?
if [[ "$rc" -ne 0 ]]; then
    echo "V108 OOF execution failed: rc=$rc"
    [[ ! -e "$OOF_LOG" ]] || chmod 0444 "$OOF_LOG"
    finish "$rc"
fi

"$PY" - "$BASE" "$PARENT" "$GEOMETRY" "$OOF" "$RECEIPT" \
        "$PARENT_LOG" "$GEOMETRY_LOG" "$OOF_LOG" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

base = Path(sys.argv[1]).resolve()
parent = Path(sys.argv[2]).resolve()
geometry = Path(sys.argv[3]).resolve()
oof_path = Path(sys.argv[4]).resolve()
receipt_path = Path(sys.argv[5]).resolve()
logs = [Path(value).resolve() for value in sys.argv[6:]]

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

report = json.loads(oof_path.read_text(encoding="ascii"))
if report.get("schema") != "rec-v108-meshsp-pareto-full-train-scene-oof-v1":
    raise RuntimeError("V108 OOF schema mismatch")
if report.get("validation_data_accessed") is not False:
    raise RuntimeError("V108 OOF accessed validation")
oof = report.get("oof")
if not isinstance(oof, dict) or not isinstance(oof.get("predicates"), dict):
    raise RuntimeError("V108 OOF gate is missing")
passed = oof.get("passed") is True and all(oof["predicates"].values())
payload = {
    "schema": "mcln-v108-meshsp-models-oof-receipt-v1",
    "version": 1,
    "validation_data_accessed": False,
    "single_gpu": True,
    "gpu": "cuda:0",
    "cache_receipts": {
        "candidate": sha(base / "candidate_train_receipt.json"),
        "geometry": sha(base / "geometry_train_receipt.json"),
    },
    "artifacts": {
        "parent": {"path": str(parent), "sha256": sha(parent), "size": parent.stat().st_size},
        "geometry": {"path": str(geometry), "sha256": sha(geometry), "size": geometry.stat().st_size},
    },
    "oof": {
        "path": str(oof_path),
        "sha256": sha(oof_path),
        "passed": passed,
        "predicates": oof["predicates"],
        "diagnostics": oof["diagnostics"],
        "subgroups": oof["subgroups"],
    },
    "logs": {path.name: sha(path) for path in logs},
    "scripts": {
        "parent": sha("/tmp/mcln_repo/scripts/train_rec_reranker.py"),
        "geometry": sha("/tmp/mcln_repo/scripts/train_rec_geometry_reranker.py"),
        "oof": sha("/tmp/mcln_repo/scripts/run_v108_meshsp_pareto_oof.py"),
    },
}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
fd = os.open(str(receipt_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
try:
    written = 0
    while written < len(raw):
        count = os.write(fd, raw[written:])
        if count <= 0:
            raise OSError("V108 models/OOF receipt write made no progress")
        written += count
    os.fsync(fd)
finally:
    os.close(fd)
print(json.dumps({"oof_passed": passed, "receipt_sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))
raise SystemExit(0 if passed else 75)
PY
rc=$?
printf '%s\n' "$rc" >"$EXIT_FILE"
chmod 0444 "$OOF" "$OOF_LOG" "$RECEIPT" "$EXIT_FILE"
chmod 0555 "$ARTIFACT_DIR"
echo "V108 models/OOF pipeline rc=$rc"
date -Is
chmod 0444 "$PIPELINE_LOG"
exit "$rc"
