#!/usr/bin/env bash
set -uo pipefail

ROOT=/tmp/mcln_repo
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
TRAIN_CACHE="$BASE/train"
GEOMETRY_CACHE="$BASE/geometry_train"
ARTIFACT_DIR="$BASE/v108_artifacts"
PARENT="$ARTIFACT_DIR/parent_h256_seed0.pth"
GEOMETRY="$ARTIFACT_DIR/geometry_h256_seed0.pth"
SOURCE="$BASE/v108_meshsp_pareto_oof.json"
OOF="$BASE/v109_meshsp_nested_policy_oof.json"
PARENT_LOG="$BASE/v109_parent_rebuild.log"
GEOMETRY_LOG="$BASE/v109_geometry_rebuild.log"
OOF_LOG="$BASE/v109_meshsp_nested_policy_oof.log"
PIPELINE_LOG="$BASE/v109_nested_policy_pipeline.log"
EXIT_FILE="$BASE/v109_nested_policy_pipeline_exitcode.txt"
RECEIPT="$BASE/v109_nested_policy_pipeline_receipt.json"
FALLBACK=/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train_missing789.txt
PY=/root/miniconda3/envs/bdetr/bin/python

finish() {
    local rc="$1"
    printf '%s\n' "$rc" >"$EXIT_FILE"
    for path in "$PARENT_LOG" "$GEOMETRY_LOG" "$OOF_LOG" \
                "$PIPELINE_LOG" "$EXIT_FILE"; do
        [[ ! -e "$path" ]] || chmod 0444 "$path"
    done
    exit "$rc"
}

for path in "$ARTIFACT_DIR" "$OOF" "$PARENT_LOG" "$GEOMETRY_LOG" \
            "$OOF_LOG" "$PIPELINE_LOG" "$EXIT_FILE" "$RECEIPT"; do
    if [[ -e "$path" ]]; then
        echo "V109 output already exists: $path" >&2
        exit 64
    fi
done

exec >"$PIPELINE_LOG" 2>&1
echo "policy=single_gpu_serial_v109_nested_meta_policy"
date -Is

expected_sha256() {
    local expected="$1"
    local path="$2"
    [[ -f "$path" && ! -L "$path" ]] || return 1
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]]
}

expected_sha256 aa77fadd55d8e579ad65748f8e1a078cbb756eaf53b061ec0339b73495a17c33 "$ROOT/scripts/run_v109_meshsp_nested_policy_oof.py" || finish 80
expected_sha256 72ca54b2db0bca829011a2f480c458c0a3e450a492dd77de9d8411e84f3e9162 "$SOURCE" || finish 81
expected_sha256 983dbe5141a4bef2a3c36e23bbc0c833aa2d4a4ea026a833708b7b2dad6fdb32 "$BASE/v108_models_oof_pipeline_receipt.json" || finish 82
expected_sha256 bfe2a650e22459d09dcbca6f525cbcda136787b456bf77d734c1e2f76b67caaa "$BASE/candidate_train_receipt.json" || finish 83
expected_sha256 e45adaafb3730f45dabcea7f0c4f4492a6ea6360b7f07bdb164270bd934d9443 "$BASE/geometry_train_receipt.json" || finish 84
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
    echo "GPU0 is not idle before V109"
    finish 85
fi
free_bytes=$(df -B1 /root/autodl-tmp | awk 'NR==2 {print $4}')
if [[ ! "$free_bytes" =~ ^[0-9]+$ || "$free_bytes" -lt 2500000000 ]]; then
    echo "insufficient free disk before V109: $free_bytes"
    finish 86
fi

mkdir -m 0755 "$ARTIFACT_DIR"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/pointnet2"

echo "stage=deterministic_parent_rebuild"
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
[[ "$rc" -eq 0 ]] || finish "$rc"
expected_sha256 7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f "$PARENT" || finish 87
chmod 0444 "$PARENT" "$PARENT_LOG"

echo "stage=deterministic_geometry_rebuild"
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
[[ "$rc" -eq 0 ]] || finish "$rc"
expected_sha256 20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972 "$GEOMETRY" || finish 88
chmod 0444 "$GEOMETRY" "$GEOMETRY_LOG"

echo "stage=nested_scene_cross_fitted_policy_oof"
date -Is
"$PY" scripts/run_v109_meshsp_nested_policy_oof.py \
    --v108-report "$SOURCE" \
    --base-cache "$TRAIN_CACHE" \
    --geometry-cache "$GEOMETRY_CACHE" \
    --parent-artifact "$PARENT" \
    --geometry-artifact "$GEOMETRY" \
    --fallback-scenes "$FALLBACK" \
    --output "$OOF" \
    --device cuda:0 >"$OOF_LOG" 2>&1
rc=$?
if [[ "$rc" -ne 0 ]]; then
    echo "V109 OOF execution failed: rc=$rc"
    finish "$rc"
fi
if [[ ! -f "$OOF" ]]; then
    echo "V109 OOF did not publish its report"
    finish 89
fi

"$PY" - "$ROOT" "$BASE" "$PARENT" "$GEOMETRY" "$OOF" \
        "$PARENT_LOG" "$GEOMETRY_LOG" "$OOF_LOG" "$RECEIPT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
base = Path(sys.argv[2]).resolve()
parent = Path(sys.argv[3]).resolve()
geometry = Path(sys.argv[4]).resolve()
oof_path = Path(sys.argv[5]).resolve()
logs = [Path(value).resolve() for value in sys.argv[6:9]]
receipt_path = Path(sys.argv[9]).resolve()

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

report = json.loads(oof_path.read_text(encoding="ascii"))
oof = report.get("oof")
if (report.get("schema")
        != "rec-v109-meshsp-nested-policy-full-train-scene-oof-v1"
        or report.get("validation_data_accessed") is not False
        or not isinstance(oof, dict)
        or not isinstance(oof.get("predicates"), dict)):
    raise RuntimeError("V109 OOF report contract changed")
passed = oof.get("passed") is True and all(oof["predicates"].values())
payload = {
    "schema": "mcln-v109-meshsp-nested-policy-oof-receipt-v1",
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
        "global_policy_selection": report["global_policy_selection"],
    },
    "logs": {path.name: sha(path) for path in logs},
    "scripts": {
        "parent": sha(root / "scripts/train_rec_reranker.py"),
        "geometry": sha(root / "scripts/train_rec_geometry_reranker.py"),
        "hierarchical": sha(root / "scripts/train_scanrefer_rec_hierarchical_reranker.py"),
        "v108": sha(root / "scripts/run_v108_meshsp_pareto_oof.py"),
        "v109": sha(root / "scripts/run_v109_meshsp_nested_policy_oof.py"),
        "runner": sha(root / "scripts/run_v109_meshsp_nested_policy_serial.sh"),
    },
}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
fd = os.open(str(receipt_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
try:
    written = 0
    while written < len(raw):
        count = os.write(fd, raw[written:])
        if count <= 0:
            raise OSError("V109 receipt write made no progress")
        written += count
    os.fsync(fd)
finally:
    os.close(fd)
print(json.dumps({
    "oof_passed": passed,
    "receipt_sha256": hashlib.sha256(raw).hexdigest(),
}, sort_keys=True))
raise SystemExit(0 if passed else 76)
PY
rc=$?
printf '%s\n' "$rc" >"$EXIT_FILE"
chmod 0444 "$OOF" "$OOF_LOG" "$RECEIPT" "$EXIT_FILE"
echo "V109 nested-policy pipeline rc=$rc"
date -Is
chmod 0444 "$PIPELINE_LOG"
exit "$rc"
