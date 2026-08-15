#!/usr/bin/env bash
set -uo pipefail

ROOT=/tmp/mcln_repo
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
TRAIN_CACHE="$BASE/train"
GEOMETRY_CACHE="$BASE/geometry_train"
ARTIFACT_DIR="$BASE/v108_artifacts"
PARENT="$ARTIFACT_DIR/parent_h256_seed0.pth"
GEOMETRY="$ARTIFACT_DIR/geometry_h256_seed0.pth"
OOF="$BASE/v108_meshsp_pareto_oof.json"
OOF_LOG="$BASE/v108_meshsp_pareto_oof.log"
PIPELINE_LOG="$BASE/v108_models_oof_pipeline.log"
EXIT_FILE="$BASE/v108_models_oof_pipeline_exitcode.txt"
RECEIPT="$BASE/v108_models_oof_pipeline_receipt.json"
V99_REPORT="$ROOT/experiment_output/historical_e71_geometry/v99_pareto_contextual_hierarchical_trainonly_v1.json"
FALLBACK=/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train_missing789.txt
FAILED="$BASE/failed_oof_legacy_loader_gate_20260814"
PY=/root/miniconda3/envs/bdetr/bin/python

finish() {
    local rc="$1"
    printf '%s\n' "$rc" >"$EXIT_FILE"
    for path in "$OOF_LOG" "$PIPELINE_LOG" "$EXIT_FILE"; do
        [[ ! -e "$path" ]] || chmod 0444 "$path"
    done
    exit "$rc"
}

for path in "$OOF" "$OOF_LOG" "$PIPELINE_LOG" "$EXIT_FILE" "$RECEIPT"; do
    if [[ -e "$path" ]]; then
        echo "V108 OOF resume output already exists: $path" >&2
        exit 64
    fi
done

exec >"$PIPELINE_LOG" 2>&1
echo "policy=single_gpu_v108_oof_resume_after_legacy_loader_gate"
date -Is

expected_sha256() {
    local expected="$1"
    local path="$2"
    [[ -f "$path" && ! -L "$path" ]] || return 1
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]]
}

expected_sha256 7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f "$PARENT" || finish 80
expected_sha256 20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972 "$GEOMETRY" || finish 81
expected_sha256 bfe2a650e22459d09dcbca6f525cbcda136787b456bf77d734c1e2f76b67caaa "$BASE/candidate_train_receipt.json" || finish 82
expected_sha256 e45adaafb3730f45dabcea7f0c4f4492a6ea6360b7f07bdb164270bd934d9443 "$BASE/geometry_train_receipt.json" || finish 83
expected_sha256 0dfd24bd12e83487b0a85d1574d5087313ba4af76c7b0255f5110d5b28540551 "$BASE/v108_parent_train.log" || finish 84
expected_sha256 c4c8e20fb196ea218bcb090e5b1032b73fa8f8c75da1b9d03d4cbb3701220903 "$BASE/v108_geometry_train.log" || finish 85
expected_sha256 244c18e9489ce2e0f9935f2256c0d0a7938b9d596fd5af60263acf0bc1db6a59 "$FAILED/v108_meshsp_pareto_oof.log" || finish 86
expected_sha256 2c34fcc23a0783ec46efe3582f18e856785e416fe56d6f2676581fc3a9093192 "$FAILED/v108_models_oof_pipeline.log" || finish 87
expected_sha256 4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865 "$FAILED/v108_models_oof_pipeline_exitcode.txt" || finish 88
expected_sha256 94dbce107e8412e00ac777cdea732cf6f93d6a7985550258ef7ae46763053c8d "$ROOT/scripts/run_v108_meshsp_pareto_oof.py" || finish 89

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
    echo "GPU0 is not idle before V108 OOF resume"
    finish 90
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/pointnet2"
echo "stage=scene_disjoint_oof_resume"
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
oof_rc=$?
if [[ "$oof_rc" -ne 0 ]]; then
    echo "V108 OOF resume execution failed: rc=$oof_rc"
    finish "$oof_rc"
fi
if [[ ! -f "$OOF" ]]; then
    echo "V108 OOF resume did not publish its report"
    finish 91
fi

"$PY" - "$BASE" "$PARENT" "$GEOMETRY" "$OOF" "$RECEIPT" \
        "$BASE/v108_parent_train.log" "$BASE/v108_geometry_train.log" \
        "$OOF_LOG" "$FAILED" "$oof_rc" <<'PY'
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
logs = [Path(value).resolve() for value in sys.argv[6:9]]
failed = Path(sys.argv[9]).resolve()
oof_execution_rc = int(sys.argv[10])

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
if oof_execution_rc != 0:
    raise RuntimeError("V108 OOF execution did not complete cleanly")
payload = {
    "schema": "mcln-v108-meshsp-models-oof-receipt-v2",
    "version": 2,
    "validation_data_accessed": False,
    "single_gpu": True,
    "gpu": "cuda:0",
    "resume_reason": "legacy_authoritative_loader_rejected_new_bound_artifacts",
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
    "failed_attempt": {
        path.name: sha(path) for path in sorted(failed.iterdir()) if path.is_file()
    },
    "scripts": {
        "parent": sha("/tmp/mcln_repo/scripts/train_rec_reranker.py"),
        "geometry": sha("/tmp/mcln_repo/scripts/train_rec_geometry_reranker.py"),
        "oof": sha("/tmp/mcln_repo/scripts/run_v108_meshsp_pareto_oof.py"),
        "resume": sha("/tmp/mcln_repo/scripts/run_v108_meshsp_oof_resume.sh"),
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
echo "V108 resumed models/OOF pipeline rc=$rc"
date -Is
chmod 0444 "$PIPELINE_LOG"
exit "$rc"
