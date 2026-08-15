#!/usr/bin/env bash
set -uo pipefail

ROOT=/tmp/mcln_repo
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
PARENT="$BASE/v108_artifacts/parent_h256_seed0.pth"
GEOMETRY="$BASE/v108_artifacts/geometry_h256_seed0.pth"
OOF="$BASE/v108_meshsp_pareto_oof.json"
OOF_LOG="$BASE/v108_meshsp_pareto_oof.log"
PIPELINE_LOG="$BASE/v108_models_oof_pipeline.log"
EXIT_FILE="$BASE/v108_models_oof_pipeline_exitcode.txt"
RECEIPT="$BASE/v108_models_oof_pipeline_receipt.json"
PY=/root/miniconda3/envs/bdetr/bin/python

for path in "$PIPELINE_LOG" "$EXIT_FILE" "$RECEIPT"; do
    if [[ -e "$path" ]]; then
        echo "V108 finalization output already exists: $path" >&2
        exit 64
    fi
done

exec >"$PIPELINE_LOG" 2>&1
echo "policy=finalize_completed_v108_oof_after_exit_contract_repair"
date -Is

"$PY" - "$ROOT" "$BASE" "$PARENT" "$GEOMETRY" "$OOF" \
        "$OOF_LOG" "$RECEIPT" <<'PY'
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
oof_log = Path(sys.argv[6]).resolve()
receipt_path = Path(sys.argv[7]).resolve()

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

expected = {
    parent: "7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f",
    geometry: "20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972",
    oof_path: "72ca54b2db0bca829011a2f480c458c0a3e450a492dd77de9d8411e84f3e9162",
    oof_log: "fad68d24ff28fe2249b8e45f56bf10898ba12d64b5254420932909acfb1d795a",
    base / "candidate_train_receipt.json": "bfe2a650e22459d09dcbca6f525cbcda136787b456bf77d734c1e2f76b67caaa",
    base / "geometry_train_receipt.json": "e45adaafb3730f45dabcea7f0c4f4492a6ea6360b7f07bdb164270bd934d9443",
    base / "v108_parent_train.log": "0dfd24bd12e83487b0a85d1574d5087313ba4af76c7b0255f5110d5b28540551",
    base / "v108_geometry_train.log": "c4c8e20fb196ea218bcb090e5b1032b73fa8f8c75da1b9d03d4cbb3701220903",
}
for path, expected_sha in expected.items():
    if not path.is_file() or path.is_symlink() or sha(path) != expected_sha:
        raise RuntimeError("V108 finalization input changed: {}".format(path))

report = json.loads(oof_path.read_text(encoding="ascii"))
oof = report.get("oof")
predicates = oof.get("predicates") if isinstance(oof, dict) else None
diagnostics = oof.get("diagnostics") if isinstance(oof, dict) else None
subgroups = oof.get("subgroups") if isinstance(oof, dict) else None
if (report.get("schema") != "rec-v108-meshsp-pareto-full-train-scene-oof-v1"
        or report.get("validation_data_accessed") is not False
        or report.get("prior_calibration_used_for_selection") is not False
        or not isinstance(predicates, dict)
        or oof.get("passed") is not False
        or diagnostics.get("delta_hits025") != 70
        or diagnostics.get("delta_hits050") != 245
        or diagnostics.get("bootstrap025", {}).get("lower_bound_95") != 32
        or diagnostics.get("bootstrap050", {}).get("lower_bound_95") != 183
        or predicates.get("delta025_at_least_oracle_scaled_gap") is not False
        or any(value is not True for key, value in predicates.items()
               if key != "delta025_at_least_oracle_scaled_gap")
        or not isinstance(subgroups, dict)):
    raise RuntimeError("V108 OOF result differs from the completed failed gate")

failure_dirs = sorted(
    path for path in base.glob("failed_oof_*_20260814") if path.is_dir()
)
if len(failure_dirs) != 4:
    raise RuntimeError("V108 failure evidence directory count changed")
payload = {
    "schema": "mcln-v108-meshsp-models-oof-receipt-v2",
    "version": 2,
    "validation_data_accessed": False,
    "single_gpu": True,
    "gpu": "cuda:0",
    "resume_reason": "legacy_v99_loader_contracts_repaired_for_bound_v108_inputs",
    "gate_outcome": "failed_delta025_by_two_hits",
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
        "passed": False,
        "predicates": predicates,
        "diagnostics": diagnostics,
        "subgroups": subgroups,
    },
    "logs": {
        "v108_parent_train.log": sha(base / "v108_parent_train.log"),
        "v108_geometry_train.log": sha(base / "v108_geometry_train.log"),
        "v108_meshsp_pareto_oof.log": sha(oof_log),
    },
    "failed_attempts": {
        directory.name: {
            path.name: sha(path)
            for path in sorted(directory.iterdir()) if path.is_file()
        }
        for directory in failure_dirs
    },
    "scripts": {
        "parent": sha(root / "scripts/train_rec_reranker.py"),
        "geometry": sha(root / "scripts/train_rec_geometry_reranker.py"),
        "hierarchical": sha(root / "scripts/train_scanrefer_rec_hierarchical_reranker.py"),
        "oof": sha(root / "scripts/run_v108_meshsp_pareto_oof.py"),
        "resume": sha(root / "scripts/run_v108_meshsp_oof_resume.sh"),
        "finalizer": sha(root / "scripts/finalize_v108_meshsp_oof.sh"),
    },
}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
fd = os.open(str(receipt_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
try:
    written = 0
    while written < len(raw):
        count = os.write(fd, raw[written:])
        if count <= 0:
            raise OSError("V108 receipt write made no progress")
        written += count
    os.fsync(fd)
finally:
    os.close(fd)
print(json.dumps({
    "oof_passed": False,
    "receipt_sha256": hashlib.sha256(raw).hexdigest(),
}, sort_keys=True))
PY
rc=$?
if [[ "$rc" -eq 0 ]]; then
    rc=75
fi
printf '%s\n' "$rc" >"$EXIT_FILE"
chmod 0444 "$OOF" "$OOF_LOG" "$RECEIPT" "$EXIT_FILE"
echo "V108 finalized models/OOF pipeline rc=$rc"
date -Is
chmod 0444 "$PIPELINE_LOG"
exit "$rc"
