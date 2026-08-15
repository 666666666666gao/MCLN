#!/usr/bin/env bash
set -uo pipefail

ROOT=/tmp/mcln_repo
DATA_VIEW=/root/autodl-tmp/DATA_ROOT_mcln_meshsp
CHECKPOINT=/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth
OLD_BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_currentcode_oldsp/train
BASE_ROOT=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
NEW_BASE="$BASE_ROOT/train"
CONTROL_RECEIPT=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_currentcode_oldsp/candidate_train_receipt.json
FALLBACK=/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train_missing789.txt
AB_REPORT="$BASE_ROOT/candidate_ab_audit.json"
AB_LOG="$BASE_ROOT/candidate_ab_audit.log"
AUDIT_DIR="$BASE_ROOT/mask_geometry_audit_train256"
AUDIT_LOG="$BASE_ROOT/mask_geometry_audit_train256.log"
GEOMETRY="$BASE_ROOT/geometry_train"
GEOMETRY_LOG="$BASE_ROOT/geometry_train.log"
EXIT_FILE="$BASE_ROOT/geometry_pipeline_exitcode.txt"
RECEIPT="$BASE_ROOT/geometry_train_receipt.json"
PY=/root/miniconda3/envs/bdetr/bin/python

if [[ ! -f "$BASE_ROOT/candidate_train_receipt.json" \
      || "$(cat "$BASE_ROOT/candidate_train_exitcode.txt")" != "0" ]]; then
    echo "complete V108 candidate cache receipt is absent" >&2
    exit 64
fi
if [[ ! -f "$CONTROL_RECEIPT" ]]; then
    echo "complete current-code old-SP control receipt is absent" >&2
    exit 69
fi
"$PY" - "$CONTROL_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
if receipt.get("schema") != "mcln-v108-currentcode-oldsp-train-candidate-cache-receipt-v1":
    raise SystemExit("V108 control receipt schema mismatch")
if receipt.get("validation_data_accessed") is not False:
    raise SystemExit("V108 control receipt accessed validation")
if receipt.get("sample_count") != 36665 or receipt.get("scene_count") != 562:
    raise SystemExit("V108 control receipt coverage mismatch")
if receipt.get("extraction_batch_size") != 12:
    raise SystemExit("V108 control receipt batch mismatch")
if receipt.get("source_sha256", {}).get("train_dist_mod.py") != (
        "34a6ed34ffc09979479deb4b5b4c72cf0c6a98ef6c768e9a5630d652bb754078"):
    raise SystemExit("V108 control extraction source changed")
PY
rc=$?
if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
fi
for path in "$AB_REPORT" "$AB_LOG" "$AUDIT_DIR" "$AUDIT_LOG" \
            "$GEOMETRY" "$GEOMETRY.building" "$GEOMETRY_LOG" \
            "$EXIT_FILE" "$RECEIPT"; do
    if [[ -e "$path" ]]; then
        echo "V108 geometry pipeline output already exists: $path" >&2
        exit 65
    fi
done
if [[ "$(readlink "$DATA_VIEW/superpoints/train")" != "/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train" ]]; then
    echo "mesh-derived train superpoint view is not active" >&2
    exit 66
fi
if [[ "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" != "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208" ]]; then
    echo "epoch71 checkpoint identity changed" >&2
    exit 67
fi
free_bytes=$(df -B1 /root/autodl-tmp | awk 'NR==2 {print $4}')
if [[ ! "$free_bytes" =~ ^[0-9]+$ || "$free_bytes" -lt 3500000000 ]]; then
    echo "insufficient free disk before V108 geometry cache: $free_bytes" >&2
    exit 68
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/pointnet2"

"$PY" scripts/audit_v108_meshsp_candidate_cache.py \
    --old-cache "$OLD_BASE" \
    --new-cache "$NEW_BASE" \
    --fallback-scenes "$FALLBACK" \
    --output "$AB_REPORT" >"$AB_LOG" 2>&1
rc=$?
if [[ "$rc" -ne 0 ]]; then
    printf '%s\n' "$rc" >"$EXIT_FILE"
    chmod 0444 "$AB_LOG" "$EXIT_FILE"
    exit "$rc"
fi

"$PY" scripts/audit_scanrefer_mask_geometry.py \
    --data-root "$DATA_VIEW/" \
    --checkpoint "$CHECKPOINT" \
    --train-cache "$NEW_BASE" \
    --output-dir "$AUDIT_DIR" \
    --scene-count 64 \
    --expressions-per-scene 4 \
    --selection-seed 0 \
    --batch-size 12 \
    --cache-extraction-batch-size 12 \
    --cache-replay-boundaries 0 \
    --num-workers 2 \
    --device cuda:0 >"$AUDIT_LOG" 2>&1
rc=$?
if [[ "$rc" -ne 0 ]]; then
    printf '%s\n' "$rc" >"$EXIT_FILE"
    chmod 0444 "$AB_LOG" "$AB_REPORT" "$AUDIT_LOG" "$EXIT_FILE"
    exit "$rc"
fi

"$PY" scripts/cache_scanrefer_rec_mask_geometry.py \
    --split train \
    --data-root "$DATA_VIEW/" \
    --checkpoint "$CHECKPOINT" \
    --base-cache "$NEW_BASE" \
    --output-dir "$GEOMETRY" \
    --audit-provenance "$AUDIT_DIR/selection.json" \
    --portable-provenance \
    --audit-train-cache "$NEW_BASE" \
    --batch-size 36 \
    --num-workers 4 \
    --shard-size 252 \
    --device cuda:0 >"$GEOMETRY_LOG" 2>&1
rc=$?
printf '%s\n' "$rc" >"$EXIT_FILE"
if [[ "$rc" -ne 0 ]]; then
    chmod 0444 "$AB_LOG" "$AB_REPORT" "$AUDIT_LOG" "$GEOMETRY_LOG" "$EXIT_FILE"
    exit "$rc"
fi

"$PY" - "$GEOMETRY" "$RECEIPT" "$AB_REPORT" "$AUDIT_DIR" \
        "$AUDIT_LOG" "$GEOMETRY_LOG" "$CONTROL_RECEIPT" \
        "$BASE_ROOT/candidate_train_receipt.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

geometry = Path(sys.argv[1]).resolve()
receipt = Path(sys.argv[2]).resolve()
ab_report = Path(sys.argv[3]).resolve()
audit_dir = Path(sys.argv[4]).resolve()
audit_log = Path(sys.argv[5]).resolve()
geometry_log = Path(sys.argv[6]).resolve()
control_receipt = Path(sys.argv[7]).resolve()
meshsp_receipt = Path(sys.argv[8]).resolve()

def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest_path = geometry / "manifest.json"
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
required = {
    "split": "train",
    "dataset_size": 36665,
    "source_dataset_size": 36665,
    "sample_count": 36665,
    "complete": True,
    "checkpoint_epoch": 71,
    "checkpoint_sha256": "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208",
    "extraction_batch_size": 36,
    "num_workers": 4,
}
for key, value in required.items():
    if manifest.get(key) != value:
        raise RuntimeError(f"geometry manifest {key} mismatch")
shards = manifest.get("shards")
if not isinstance(shards, list) or len(shards) != 146:
    raise RuntimeError("geometry shard count mismatch")
expected_names = [f"shard_{index:06d}.pt" for index in range(146)]
if [row.get("name") for row in shards] != expected_names:
    raise RuntimeError("geometry shard sequence mismatch")
expected_files = {"manifest.json", *expected_names}
actual_files = {path.name for path in geometry.iterdir() if path.is_file()}
if actual_files != expected_files:
    raise RuntimeError("geometry output contains unexpected files")
if not json.loads(ab_report.read_text()).get("passed"):
    raise RuntimeError("candidate A/B audit did not pass")
selection = audit_dir / "selection.json"
summary = audit_dir / "summary.json"
rows = audit_dir / "rows.pt"
for path in (selection, summary, rows):
    if not path.is_file():
        raise RuntimeError(f"mask geometry audit output missing: {path}")
total_bytes = sum((geometry / name).stat().st_size for name in expected_names)
payload = {
    "schema": "mcln-v108-meshsp-train-geometry-cache-receipt-v1",
    "version": 1,
    "validation_data_accessed": False,
    "sample_count": 36665,
    "scene_count": 562,
    "extraction_batch_size": 36,
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "cache_content_digest": manifest.get("cache_content_digest"),
    "base_cache_content_sha256": manifest.get(
        "base_cache_binding", {}).get("content_sha256"),
    "shard_count": len(shards),
    "shard_bytes": total_bytes,
    "candidate_ab_audit_sha256": file_sha(ab_report),
    "control_candidate_receipt_sha256": file_sha(control_receipt),
    "meshsp_candidate_receipt_sha256": file_sha(meshsp_receipt),
    "mask_geometry_selection_sha256": file_sha(selection),
    "mask_geometry_summary_sha256": file_sha(summary),
    "mask_geometry_rows_sha256": file_sha(rows),
    "mask_geometry_audit_log_sha256": file_sha(audit_log),
    "geometry_log_sha256": file_sha(geometry_log),
}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
fd = os.open(str(receipt), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
try:
    os.write(fd, raw)
    os.fsync(fd)
finally:
    os.close(fd)
print(json.dumps(payload, sort_keys=True))
PY
rc=$?
printf '%s\n' "$rc" >"$EXIT_FILE"
if [[ "$rc" -ne 0 ]]; then
    chmod 0444 "$AB_LOG" "$AB_REPORT" "$AUDIT_LOG" "$GEOMETRY_LOG" "$EXIT_FILE"
    exit "$rc"
fi

find "$AUDIT_DIR" -maxdepth 1 -type f -exec chmod 0444 {} +
chmod 0555 "$AUDIT_DIR"
find "$GEOMETRY" -maxdepth 1 -type f -exec chmod 0444 {} +
chmod 0555 "$GEOMETRY"
chmod 0444 "$AB_LOG" "$AB_REPORT" "$AUDIT_LOG" "$GEOMETRY_LOG" \
    "$EXIT_FILE" "$RECEIPT"
exit 0
