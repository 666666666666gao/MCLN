#!/usr/bin/env bash
set -euo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly OUTPUT_ROOT="/root/autodl-tmp/DATA_ROOT/output"
readonly V135_RUN_DIR="${OUTPUT_ROOT}/network_v135_relation_counterfactual/v135_relation_counterfactual_formal_e1_e4_b8x1/scanrefer/v135_relation_counterfactual_formal_e1_e4_b8x1/1786818966"
readonly V135_ADMISSION="${OUTPUT_ROOT}/network_v135_relation_counterfactual/gates/v135_formal_admission_a3baad6_20260816.json"
readonly V135_CONTRACT="${OUTPUT_ROOT}/network_v135_relation_counterfactual/gates/v135_relation_counterfactual_contract_a3baad6_20260816.json"
readonly V135_LAUNCH_LOG="${OUTPUT_ROOT}/network_v135_relation_counterfactual/launch/v135_relation_counterfactual_formal_e1_e4_b8x1_20260816_023602.log"
readonly V135_FINAL_AUDIT="${OUTPUT_ROOT}/network_v135_relation_counterfactual/gates/v135_formal_result_audit_v2_1786818966.json"
readonly V135_PROCESS_PATTERN="train_dist_mod.py .*--exp v135_relation_counterfactual_formal_e1_e4_b8x1"
readonly SINGLE_ROOT="${OUTPUT_ROOT}/network_scanrefer_single_stage_phase2"
readonly SINGLE_EXP="scanrefer_single_stage_phase2_pure_scanrefer_smoke_e1_b18x1"
readonly SINGLE_RUN_ROOT="${SINGLE_ROOT}/${SINGLE_EXP}/scanrefer/${SINGLE_EXP}"
readonly SINGLE_SOURCE="${OUTPUT_ROOT}/single_stage_best_postprocess/scanrefer/mcln_epoch71_parent_geometry_single_stage_e1_e100_b18x4/1785907694/ckpt_best_rec_acc025.pth"
readonly SINGLE_SOURCE_SHA="8804109f0db25113cc6683314dcc7ab1ca2f7a93c1307a1fcf76420c6dc43eec"
readonly SINGLE_SOURCE_AUDIT="${SINGLE_ROOT}/gates/source_epoch7_audit_r3_pure_scanrefer.json"
readonly SINGLE_SMOKE_GATE="${SINGLE_ROOT}/gates/smoke_gate_r2_pure_scanrefer.json"
readonly CONTROLLER_LOCK="${OUTPUT_ROOT}/network_v135_relation_counterfactual/v135_to_scanrefer.lock"

verify_single_smoke_gate() {
  "${PYTHON_BIN}" scripts/audit_scanrefer_single_stage_phase2.py \
    --mode smoke-verify \
    --repo-root "${ROOT_DIR}" \
    --checkpoint "${SINGLE_SOURCE}" \
    --expected-sha256 "${SINGLE_SOURCE_SHA}" \
    --expected-epoch 7 \
    --data-root /root/autodl-tmp/DATA_ROOT/ \
    --source-audit "${SINGLE_SOURCE_AUDIT}" \
    --gate "${SINGLE_SMOKE_GATE}"
}

cd "${ROOT_DIR}"
exec 9>"${CONTROLLER_LOCK}"
if ! flock -n 9; then
  echo "another V135 continuation controller owns ${CONTROLLER_LOCK}" >&2
  exit 5
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] waiting for exact V135 formal process"
while pgrep -f -- "${V135_PROCESS_PATTERN}" >/dev/null; do
  sleep 30
done
sleep 5

if [[ ! -e "${V135_FINAL_AUDIT}" ]]; then
  "${PYTHON_BIN}" scripts/audit_v135_formal_result.py \
    --run-dir "${V135_RUN_DIR}" \
    --formal-admission "${V135_ADMISSION}" \
    --contract-receipt "${V135_CONTRACT}" \
    --launch-log "${V135_LAUNCH_LOG}" \
    --output "${V135_FINAL_AUDIT}"
else
  echo "resuming from existing V135 final audit: ${V135_FINAL_AUDIT}"
fi

decision="$(${PYTHON_BIN} - "${V135_FINAL_AUDIT}" <<'PY'
import json
import os
import stat
import sys

path = sys.argv[1]
entry = os.lstat(path)
if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
    raise SystemExit("V135 final audit is not a regular file")
if stat.S_IMODE(entry.st_mode) != 0o444:
    raise SystemExit("V135 final audit is not mode 0444")
with open(path, "r", encoding="utf-8") as handle:
    receipt = json.load(handle)
if receipt.get("schema") != "mcln-v135-formal-result-audit-v2":
    raise SystemExit("V135 final audit schema changed")
if receipt.get("status") != "complete":
    raise SystemExit("V135 final audit is incomplete")
expected_run = (
    "/root/autodl-tmp/DATA_ROOT/output/network_v135_relation_counterfactual/"
    "v135_relation_counterfactual_formal_e1_e4_b8x1/scanrefer/"
    "v135_relation_counterfactual_formal_e1_e4_b8x1/1786818966"
)
if receipt.get("run_dir") != expected_run:
    raise SystemExit("V135 final audit run binding changed")
if [row.get("epoch") for row in receipt.get("epochs", ())] != [1, 2, 3, 4]:
    raise SystemExit("V135 final audit epoch coverage changed")
expected_provenance = {
    "contract": "eb3d9d6fc80ac447a676fffd71d86f9bddf3391dae3e620442fe7308a8c65542",
    "formal_admission": "c612ac28611b6345ed70a01c45e3cf338d8f64cbd0f033e84723885c71266dd8",
    "v99_archive": "6c5a98cd5734bb6916a1af250b71c0e4c19725378fddbdc7611796252967afdb",
    "v109_permanent_retention": "28b721bad9c7474c891877c5f8d4afb9cc684f491428f9da076948e4c7421b7e",
}
for name, expected_sha in expected_provenance.items():
    if receipt.get("provenance", {}).get(name, {}).get("sha256") != expected_sha:
        raise SystemExit("V135 final audit provenance changed: " + name)
hits025 = [row.get("rec", {}).get("hits025") for row in receipt["epochs"]]
if any(isinstance(value, bool) or not isinstance(value, int)
       for value in hits025):
    raise SystemExit("V135 final audit REC@0.25 hits are invalid")
best_hits025 = max(hits025)
improves_v99 = best_hits025 > 5572
expected_decision = (
    "retain_v135_and_review" if improves_v99
    else "freeze_v99_and_start_pure_scanrefer_single_stage"
)
if receipt.get("best_rec_hits025") != best_hits025:
    raise SystemExit("V135 final audit best REC@0.25 is inconsistent")
if receipt.get("improves_v99_rec025") is not improves_v99:
    raise SystemExit("V135 final audit V99 comparison is inconsistent")
if receipt.get("decision") != expected_decision:
    raise SystemExit("V135 final audit decision is inconsistent")
print(receipt["decision"])
PY
)"
echo "V135 final decision=${decision}"

if [[ "${decision}" == "retain_v135_and_review" ]]; then
  echo "V135 exceeds V99 REC@0.25; stopping before single-stage"
  exit 0
fi
if [[ "${decision}" != "freeze_v99_and_start_pure_scanrefer_single_stage" ]]; then
  echo "unexpected V135 final decision: ${decision}" >&2
  exit 7
fi

if [[ -e "${SINGLE_SMOKE_GATE}" ]]; then
  echo "resuming from existing pure-ScanRefer smoke gate"
  verify_single_smoke_gate
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] pure ScanRefer smoke gate verified"
  exit 0
fi
if ! gpu_pids="$(nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader,nounits)"; then
  echo "GPU state query failed; refusing to start single-stage smoke" >&2
  exit 8
fi
if grep -Eq '^[[:space:]]*[0-9]+' <<<"${gpu_pids}"; then
  echo "GPU is not idle after V135; refusing to start single-stage smoke" >&2
  exit 8
fi
declare -A runs_before=()
while IFS= read -r path; do
  runs_before["${path}"]=1
done < <(find "${SINGLE_RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -print \
  2>/dev/null || true)

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] starting pure ScanRefer smoke"
MODE=smoke MASTER_PORT=5142 bash scripts/run_scanrefer_single_stage_phase2.sh

smoke_runs=()
while IFS= read -r path; do
  if [[ -z "${runs_before[$path]+x}" ]]; then
    smoke_runs+=("${path}")
  fi
done < <(find "${SINGLE_RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -print)
if ((${#smoke_runs[@]} != 1)); then
  echo "pure-ScanRefer smoke must produce exactly one new run directory" >&2
  exit 10
fi
smoke_run="${smoke_runs[0]}"
smoke_metrics="${smoke_run}/eval_metrics_epoch_1.json"
smoke_config="${smoke_run}/config.json"
if [[ ! -f "${smoke_metrics}" || ! -f "${smoke_config}" ]]; then
  echo "pure-ScanRefer smoke evidence is incomplete" >&2
  exit 10
fi

"${PYTHON_BIN}" scripts/audit_scanrefer_single_stage_phase2.py \
  --mode smoke-build \
  --repo-root "${ROOT_DIR}" \
  --checkpoint "${SINGLE_SOURCE}" \
  --expected-sha256 "${SINGLE_SOURCE_SHA}" \
  --expected-epoch 7 \
  --data-root /root/autodl-tmp/DATA_ROOT/ \
  --source-audit "${SINGLE_SOURCE_AUDIT}" \
  --smoke-metrics "${smoke_metrics}" \
  --smoke-config "${smoke_config}" \
  --output "${SINGLE_SMOKE_GATE}"
verify_single_smoke_gate

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] pure ScanRefer smoke audited"
echo "smoke_run=${smoke_run}"
echo "smoke_gate=${SINGLE_SMOKE_GATE}"
