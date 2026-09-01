#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly SOURCE_CHECKPOINT="${DATA_ROOT%/}/gf_detector_l6o256.pth"
readonly SOURCE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly DATASET="nr3d"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v99_baseline_gt/nr3d"
readonly REQUIRED_RESUME_CHECKPOINT="/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/backbone/nr3d_mcln_joint_butdcls_v99_plateau_lr10_extension_e60_e62_b16x1_w4p2_20260823_053950/nr3d/nr3d_mcln_joint_butdcls_v99_plateau_lr10_extension_e60_e62_b16x1_w4p2/1787434801/ckpt_epoch_last.pth"
readonly REQUIRED_RESUME_SHA256="0d2622018289b1d9c4d9290b1b53f68dbf69f7f2ddbdd7ab3abaaf29fe0b2a3c"
readonly REQUIRED_RESUME_EPOCH=60
readonly RESUME_LR_SCALE="1.0"
readonly EXP="nr3d_mcln_joint_butdcls_v99_plateau_lr10_recovery_e61_e62_b16x1_w4p2"
readonly TRAIN_SCREEN_NAME="mcln_nr3d_plateau_lr10_recovery_train"
readonly GUARD_SCREEN_NAME="mcln_nr3d_plateau_lr10_recovery_guard"
readonly GUARD_SCRIPT="/root/mcln_nr3d_plateau_lr10_recovery_guard.py"
readonly GUARD_SCRIPT_SHA256="7ead92808d95c317b47ab0bbcfcd39e7149c09e16be19f2891be7ebc884dce00"
readonly WATCHDOG_SCRIPT="/root/mcln_nr3d_plateau_lr10_watchdog.py"
readonly WATCHDOG_SCRIPT_SHA256="e0d3a4d663eb1166b0b5c4cad59d0c45ec0ccba049392a8cb0ceb98c0af36529"
readonly REQUIRED_MAIN_UTILS_SHA256="f0ff9c2bcde8d39e516092b63580fbdd494c9bc48a487d0c561f5ede8bdfe4b9"
readonly REQUIRED_PIPELINE_SHA256="264eabacb8c034ad51f4fc30ce33ef990408a19e68400069a74c575f58da31a9"
readonly LOWLR_RUN_LEAF="${OUTPUT_ROOT}/backbone/nr3d_mcln_joint_butdcls_v99_plateau_lr10_e58_e59_b16x1_w4p2_20260823_011833/nr3d/nr3d_mcln_joint_butdcls_v99_plateau_lr10_e58_e59_b16x1_w4p2/1787419124"
readonly LOWLR_E58_RECEIPT="${LOWLR_RUN_LEAF}/eval_metrics_epoch_58.json"
readonly LOWLR_E58_SHA256="444a44a3d826980d324b3d028447ae3f450e88fd941960a5d58dd9cac1ecc496"
readonly LOWLR_E59_RECEIPT="${LOWLR_RUN_LEAF}/eval_metrics_epoch_59.json"
readonly LOWLR_E59_SHA256="e0edb0ab65f755f81785c3636e0435a84c490b26c54e3fae8db65478fcf11ace"
readonly LOWLR_GUARD_STATE="${OUTPUT_ROOT}/control/plateau_lr10/guard_state.json"
readonly LOWLR_GUARD_STATE_SHA256="d63d90eb6e8bb38b4b67afb8a3b1a18eeb6d9063e6beb05405571b3cccbfe97b"
readonly LOWLR_E59_CHECKPOINT="${LOWLR_RUN_LEAF}/ckpt_epoch_59.pth"
readonly LOWLR_E59_CHECKPOINT_SHA256="7850d5bfcd936b920d7991edeb9be664e1fd1a8861b66598c52e3d89a1c9cff5"
readonly EXTENSION_E60_RECEIPT="/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/backbone/nr3d_mcln_joint_butdcls_v99_plateau_lr10_extension_e60_e62_b16x1_w4p2_20260823_053950/nr3d/nr3d_mcln_joint_butdcls_v99_plateau_lr10_extension_e60_e62_b16x1_w4p2/1787434801/eval_metrics_epoch_60.json"
readonly EXTENSION_E60_RECEIPT_SHA256="f94086aa3c5c97b28a6db8458925c5c23b68c7b915cf4452193c1eec9dc60e51"
readonly EXTENSION_E60_GUARD_STATE="/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/control/plateau_lr10_extension/guard_state.json"
readonly EXTENSION_E60_GUARD_STATE_SHA256="2f7efdc3ed3aac6521106760ce3b9ce12e90698d387e91d408fba911be2bfd07"
readonly BATCH_SIZE=16
readonly MAX_EPOCH=62
readonly EXPECTED_EVAL_SAMPLE_COUNT=7899
readonly MASTER_PORT=5311
readonly MIN_FREE_GB=7
readonly BACKBONE_JOINT_TRAINING=1
readonly INFERENCE_USES_GROUND_TRUTH=1
readonly USE_BACKBONE_INITIALIZATION=1
readonly TASK_CHECKPOINT_TRANSFER=0
readonly BACKBONE_AUGMENT_DET=0
CHECKPOINT_RETENTION_METRICS=(rec_acc025)
DATASET_LR_ARGS=(
  --lr_backbone 1e-3 --lr 1e-4
  --lr_decay_epochs 150
)
BACKBONE_EXTRA_ARGS=(
  --print_freq 20
  --joint_det
  --butd_cls
  --resume_lr_scale "${RESUME_LR_SCALE}"
)

for variable_name in \
    BACKBONE_RESUME_CHECKPOINT BACKBONE_RESUME_SHA256 BACKBONE_RESUME_EPOCH; do
  if [[ -n "${!variable_name:-}" ]]; then
    case "${variable_name}" in
      BACKBONE_RESUME_CHECKPOINT) required_value="${REQUIRED_RESUME_CHECKPOINT}" ;;
      BACKBONE_RESUME_SHA256) required_value="${REQUIRED_RESUME_SHA256}" ;;
      BACKBONE_RESUME_EPOCH) required_value="${REQUIRED_RESUME_EPOCH}" ;;
    esac
    if [[ "${!variable_name}" != "${required_value}" ]]; then
      echo "${variable_name} conflicts with the pinned E60 recovery resume" >&2
      exit 2
    fi
  fi
done
export BACKBONE_RESUME_CHECKPOINT="${REQUIRED_RESUME_CHECKPOINT}"
export BACKBONE_RESUME_SHA256="${REQUIRED_RESUME_SHA256}"
export BACKBONE_RESUME_EPOCH="${REQUIRED_RESUME_EPOCH}"
export VALIDATE_BACKBONE_RESUME=0
export MODE="${MODE:-backbone}"
case "${MODE}" in
  preflight|backbone) ;;
  *) echo "this wrapper supports MODE=preflight or MODE=backbone only" >&2; exit 2 ;;
esac
command -v setsid >/dev/null 2>&1 || { echo "setsid is required" >&2; exit 3; }

require_exact_sha256() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; exit 3; }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA-256 changed: ${actual}" >&2
    exit 3
  }
}

require_exact_sha256 "${GUARD_SCRIPT}" "${GUARD_SCRIPT_SHA256}" "formal guard"
require_exact_sha256 "${WATCHDOG_SCRIPT}" "${WATCHDOG_SCRIPT_SHA256}" "training-side watchdog"
require_exact_sha256 "${ROOT_DIR}/main_utils.py" "${REQUIRED_MAIN_UTILS_SHA256}" "resume-LR implementation"
require_exact_sha256 "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh" "${REQUIRED_PIPELINE_SHA256}" "shared launcher"
require_exact_sha256 "${LOWLR_E58_RECEIPT}" "${LOWLR_E58_SHA256}" "low-LR E58 receipt"
require_exact_sha256 "${LOWLR_E59_RECEIPT}" "${LOWLR_E59_SHA256}" "low-LR E59 receipt"
require_exact_sha256 "${LOWLR_GUARD_STATE}" "${LOWLR_GUARD_STATE_SHA256}" "low-LR guard state"
require_exact_sha256 "${LOWLR_E59_CHECKPOINT}" "${LOWLR_E59_CHECKPOINT_SHA256}" "low-LR E59 checkpoint"
require_exact_sha256 "$EXTENSION_E60_RECEIPT" "$EXTENSION_E60_RECEIPT_SHA256" "extension E60 receipt"
require_exact_sha256 "$EXTENSION_E60_GUARD_STATE" "$EXTENSION_E60_GUARD_STATE_SHA256" "extension E60 guard state"
require_exact_sha256 "$REQUIRED_RESUME_CHECKPOINT" "$REQUIRED_RESUME_SHA256" "recovery E60 checkpoint"

PROOF_E58_RECEIPT="${LOWLR_E58_RECEIPT}" \
PROOF_E59_RECEIPT="${LOWLR_E59_RECEIPT}" \
PROOF_GUARD_STATE="${LOWLR_GUARD_STATE}" \
PROOF_CHECKPOINT="${LOWLR_E59_CHECKPOINT}" \
"${PYTHON_BIN}" - <<'PY'
import json
import math
import os
from pathlib import Path

import torch

expected = ((58, 4452), (59, 4432))
baseline = 4463
sample_count = 7899
paths = (
    Path(os.environ["PROOF_E58_RECEIPT"]),
    Path(os.environ["PROOF_E59_RECEIPT"]),
)
for (epoch, expected_hits), path in zip(expected, paths):
    payload = json.loads(path.read_text(encoding="utf-8"))
    metric = payload["position_subgroups"]["multiple"]
    observed_count = int(metric["sample_count"])
    hits025 = int(metric["hits025"])
    acc025 = float(metric["acc025"])
    if observed_count != sample_count:
        raise SystemExit("low-LR receipt sample_count mismatch")
    if hits025 != expected_hits or hits025 > baseline:
        raise SystemExit("low-LR receipt does not prove non-improvement")
    if not math.isclose(
            acc025, hits025 / sample_count,
            rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("low-LR receipt accuracy/hits mismatch")

state = json.loads(
    Path(os.environ["PROOF_GUARD_STATE"]).read_text(encoding="utf-8")
)
latest = state["latest"]
if (
        int(state["baseline_best_hits"]) != baseline
        or int(state["best_hits"]) != baseline
        or int(state["expected_sample_count"]) != sample_count
        or int(state["consecutive_non_improvements"]) != 2
        or int(latest["epoch"]) != 59
        or int(latest["hits025"]) != 4432
        or Path(latest["receipt"]).resolve() != paths[1].resolve()):
    raise SystemExit("formal guard state does not prove low-LR patience2")

checkpoint = torch.load(
    os.environ["PROOF_CHECKPOINT"], map_location="cpu"
)
config = checkpoint.get("config")
config = vars(config) if hasattr(config, "__dict__") else config
optimizer = checkpoint.get("optimizer", {})
groups = optimizer.get("param_groups", [])
observed_lrs = [float(group["lr"]) for group in groups]
expected_lrs = [1e-6, 1e-5, 1e-6, 1.25e-6]
if int(checkpoint.get("epoch", -1)) != 59:
    raise SystemExit("resume checkpoint is not epoch 59")
if len(groups) != 4 or len(optimizer.get("state", {})) != 716:
    raise SystemExit("resume checkpoint optimizer provenance mismatch")
if any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15)
        for a, b in zip(observed_lrs, expected_lrs)):
    raise SystemExit("resume checkpoint LR mismatch")
if (
        float(config.get("resume_lr_scale_lineage", -1.0)) != 0.1
        or int(config.get("gradient_accumulation_steps", -1)) != 1
        or int(config.get("batch_size", -1)) != 16
        or config.get("lr_scheduler") != "step"
        or int(config.get("warmup_epoch", 0)) != -1):
    raise SystemExit("resume checkpoint training contract mismatch")
print(
    "extension_proof=E58:4452,E59:4432,baseline:4463,"
    "resume:E59,same_lr:E60-E62"
)
PY


PROOF_E60_RECEIPT="$EXTENSION_E60_RECEIPT" \
PROOF_E60_GUARD_STATE="$EXTENSION_E60_GUARD_STATE" \
PROOF_E60_CHECKPOINT="$REQUIRED_RESUME_CHECKPOINT" \
"$PYTHON_BIN" - <<'PYRECOVERY'
import json
import math
import os
from pathlib import Path

import torch

sample_count = 7899
baseline = 4463
receipt_path = Path(os.environ["PROOF_E60_RECEIPT"])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
metric = receipt["position_subgroups"]["multiple"]
if (
        int(metric["sample_count"]) != sample_count
        or int(metric["hits025"]) != 4437
        or int(metric["hits050"]) != 3722
        or not math.isclose(
            float(metric["acc025"]), 4437 / sample_count,
            rel_tol=0.0, abs_tol=1e-12)):
    raise SystemExit("E60 recovery receipt mismatch")

state = json.loads(
    Path(os.environ["PROOF_E60_GUARD_STATE"]).read_text(encoding="utf-8")
)
latest = state["latest"]
if (
        state.get("schema") != "mcln-nr3d-plateau-lr10-extension-guard-v1"
        or int(state["baseline_best_hits"]) != baseline
        or int(state["best_hits"]) != baseline
        or int(state["consecutive_non_improvements"]) != 1
        or int(latest["epoch"]) != 60
        or int(latest["hits025"]) != 4437
        or Path(latest["receipt"]).resolve() != receipt_path.resolve()):
    raise SystemExit("E60 formal guard state mismatch")

checkpoint = torch.load(
    os.environ["PROOF_E60_CHECKPOINT"], map_location="cpu"
)
config = checkpoint.get("config")
config = vars(config) if hasattr(config, "__dict__") else config
optimizer = checkpoint.get("optimizer", {})
groups = optimizer.get("param_groups", [])
expected_lrs = [1e-6, 1e-5, 1e-6, 1.25e-6]
observed_lrs = [float(group["lr"]) for group in groups]
if (
        int(checkpoint.get("epoch", -1)) != 60
        or len(groups) != 4
        or len(optimizer.get("state", {})) != 716
        or any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15)
            for a, b in zip(observed_lrs, expected_lrs))
        or float(config.get("resume_lr_scale_lineage", -1.0)) != 0.1
        or int(config.get("gradient_accumulation_steps", -1)) != 1
        or int(config.get("batch_size", -1)) != 16
        or config.get("lr_scheduler") != "step"
        or int(config.get("warmup_epoch", 0)) != -1):
    raise SystemExit("E60 checkpoint continuation contract mismatch")
print(
    "recovery_proof=E60:4437/3722,baseline:4463,"
    "resume:E60,same_lr:E61-E62,prior_misses:1"
)
PYRECOVERY

start_backbone_guard() {
  local run_root="$1"
  local train_screen_id="${STY:-}"
  local train_screen_name=""
  local train_screen_pid=""
  local train_screen_start_ticks=""
  local ready_file="${run_root}/formal_guard_ready.json"
  local watchdog_heartbeat="${run_root}/formal_guard_watchdog.json"
  local guard_ready=0
  local watchdog_ready=0
  local guard_pid=""
  local guard_start_ticks=""
  local watchdog_pid=""
  local watchdog_launch_start_ticks=""

  train_screen_name="${train_screen_id##*.}"
  train_screen_pid="${train_screen_id%%.*}"
  if [[ -z "${train_screen_id}" || "${train_screen_id}" != *.* ||
        "${train_screen_name}" != "${TRAIN_SCREEN_NAME}" ||
        ! "${train_screen_pid}" =~ ^[0-9]+$ ]]; then
    echo "current screen is '${STY:-unset}', expected <pid>.${TRAIN_SCREEN_NAME}" >&2
    exit 4
  fi
  train_screen_start_ticks="$("${PYTHON_BIN}" -       "${train_screen_pid}" "${TRAIN_SCREEN_NAME}" <<'PY'
import pathlib
import sys
pid = int(sys.argv[1])
expected_name = sys.argv[2]
root = pathlib.Path("/proc") / str(pid)
stat = (root / "stat").read_text(encoding="utf-8")
ticks = int(stat[stat.rfind(")") + 2:].split()[19])
cmdline = (root / "cmdline").read_bytes().replace(b"\0", b" ").decode(
    "utf-8", errors="replace"
)
if "SCREEN" not in cmdline or expected_name not in cmdline:
    raise SystemExit("screen process command identity mismatch")
print(ticks)
PY
  )"
  screen -S "${train_screen_id}" -Q select . >/dev/null 2>&1 || {
    echo "formal run screen is not alive: ${train_screen_id}" >&2
    exit 4
  }
  if screen -S "${GUARD_SCREEN_NAME}" -Q select . >/dev/null 2>&1; then
    echo "formal guard screen already exists: ${GUARD_SCREEN_NAME}" >&2
    exit 4
  fi
  [[ ! -e "${ready_file}" ]] || {
    echo "formal guard ready receipt already exists: ${ready_file}" >&2
    exit 4
  }
  [[ ! -e "${watchdog_heartbeat}" ]] || {
    echo "training watchdog heartbeat already exists: ${watchdog_heartbeat}" >&2
    exit 4
  }

  screen -dmS "${GUARD_SCREEN_NAME}" bash -lc     "exec '${PYTHON_BIN}' '${GUARD_SCRIPT}' --run-root '${run_root}' --screen '${train_screen_id}' --screen-start-ticks '${train_screen_start_ticks}' --experiment '${EXP}' --ready-file '${ready_file}' --watchdog-heartbeat '${watchdog_heartbeat}' 2>&1 | tee -a /root/mcln_nr3d_plateau_lr10_guard.log"
  for _ in $(seq 1 30); do
    sleep 1
    if ! screen -S "${GUARD_SCREEN_NAME}" -Q select . >/dev/null 2>&1; then
      echo "formal guard exited before publishing ready receipt" >&2
      exit 4
    fi
    if [[ -f "${ready_file}" ]]; then
      if ! "${PYTHON_BIN}" - "${ready_file}" "${run_root}"           "${train_screen_id}" "${EXP}"           "${train_screen_start_ticks}" <<'PY'
import json
import os
import pathlib
import sys

ready = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_root = str(pathlib.Path(sys.argv[2]).resolve())
expected_watchdog = str(
    (pathlib.Path(sys.argv[2]) / "formal_guard_watchdog.json").resolve()
)
if (
        ready.get("schema") != "mcln-formal-guard-ready-v1"
        or ready.get("run_root") != expected_root
        or ready.get("screen") != sys.argv[3]
        or ready.get("experiment") != sys.argv[4]
        or int(ready.get("screen_start_ticks", -1)) != int(sys.argv[5])
        or ready.get("watchdog_heartbeat") != expected_watchdog):
    raise SystemExit("formal guard ready receipt identity mismatch")
pid = int(ready["pid"])
start_ticks = int(ready["process_start_ticks"])
os.kill(pid, 0)
stat = pathlib.Path("/proc") / str(pid) / "stat"
content = stat.read_text(encoding="utf-8")
observed_ticks = int(content[content.rfind(")") + 2:].split()[19])
if observed_ticks != start_ticks:
    raise SystemExit("formal guard PID identity mismatch")
print("formal_guard_ready_pid={} start_ticks={}".format(pid, start_ticks))
PY
      then
        screen -S "${GUARD_SCREEN_NAME}" -X quit >/dev/null 2>&1 || true
        exit 4
      fi
      guard_ready=1
      break
    fi
  done
  if [[ "${guard_ready}" != "1" ]]; then
    echo "formal guard did not publish ready receipt within 30 seconds" >&2
    screen -S "${GUARD_SCREEN_NAME}" -X quit >/dev/null 2>&1 || true
    exit 4
  fi

  read -r guard_pid guard_start_ticks < <(
    "${PYTHON_BIN}" - "${ready_file}" <<'PY'
import json
import pathlib
import sys
ready = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(ready["pid"]), int(ready["process_start_ticks"]))
PY
  )
  setsid "${PYTHON_BIN}" "${WATCHDOG_SCRIPT}"     --guard-pid "${guard_pid}"     --guard-start-ticks "${guard_start_ticks}"     --guard-command-fragment "${GUARD_SCRIPT}"     --screen "${train_screen_id}"     --screen-start-ticks "${train_screen_start_ticks}"     --experiment "${EXP}"     --run-root "${run_root}"     --heartbeat "${watchdog_heartbeat}"     >>/root/mcln_nr3d_plateau_lr10_watchdog.log 2>&1 </dev/null &
  watchdog_pid=$!
  watchdog_launch_start_ticks="$("$PYTHON_BIN" - "$watchdog_pid" <<'PYWATCHDOGPID' || true
import pathlib
import sys

pid = int(sys.argv[1])
content = (
    pathlib.Path("/proc") / str(pid) / "stat"
).read_text(encoding="utf-8")
print(int(content[content.rfind(")") + 2:].split()[19]))
PYWATCHDOGPID
  )"

  for _ in $(seq 1 10); do
    sleep 1
    kill -0 "${watchdog_pid}" >/dev/null 2>&1 || break
    if [[ -f "${watchdog_heartbeat}" ]] &&        "${PYTHON_BIN}" - "${watchdog_heartbeat}" "${run_root}"            "${train_screen_id}" "${EXP}" "${guard_pid}"            "${guard_start_ticks}" "${watchdog_pid}"            "${train_screen_start_ticks}" "${WATCHDOG_SCRIPT}" <<'PY'
import json
import os
import pathlib
import sys

heartbeat = json.loads(
    pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
)
expected = {
    "schema": "mcln-formal-training-watchdog-v1",
    "run_root": str(pathlib.Path(sys.argv[2]).resolve()),
    "screen": sys.argv[3],
    "experiment": sys.argv[4],
    "guard_pid": int(sys.argv[5]),
    "guard_start_ticks": int(sys.argv[6]),
    "watchdog_pid": int(sys.argv[7]),
    "screen_start_ticks": int(sys.argv[8]),
}
for name, value in expected.items():
    if heartbeat.get(name) != value:
        raise SystemExit("watchdog heartbeat identity mismatch: " + name)
pid = int(heartbeat["watchdog_pid"])
start_ticks = int(heartbeat["watchdog_start_ticks"])
os.kill(pid, 0)
root = pathlib.Path("/proc") / str(pid)
stat = (root / "stat").read_text(encoding="utf-8")
observed_ticks = int(stat[stat.rfind(")") + 2:].split()[19])
cmdline = (root / "cmdline").read_bytes().replace(b"\0", b" ").decode(
    "utf-8", errors="replace"
)
if observed_ticks != start_ticks:
    raise SystemExit("watchdog PID identity mismatch")
if sys.argv[9] not in cmdline or sys.argv[4] not in cmdline:
    raise SystemExit("watchdog command identity mismatch")
print("training_watchdog_ready_pid={} start_ticks={}".format(pid, start_ticks))
PY
    then
      watchdog_ready=1
      break
    fi
  done
  if [[ "${watchdog_ready}" != "1" ]]; then
    echo "training-side watchdog did not become ready" >&2
    if [[ -n "$watchdog_launch_start_ticks" ]]; then
      "$PYTHON_BIN" - "$watchdog_pid" "$watchdog_launch_start_ticks" \
          "$WATCHDOG_SCRIPT" "$EXP" <<'PYWATCHDOGSTOP' || true
import os
import pathlib
import signal
import sys

pid = int(sys.argv[1])
expected_ticks = int(sys.argv[2])
script = sys.argv[3]
experiment = sys.argv[4]
root = pathlib.Path("/proc") / str(pid)
try:
    content = (root / "stat").read_text(encoding="utf-8")
    observed_ticks = int(content[content.rfind(")") + 2:].split()[19])
    cmdline = (root / "cmdline").read_bytes().replace(
        b"\0", b" "
    ).decode("utf-8", errors="replace")
except OSError:
    raise SystemExit(0)
if (
        observed_ticks == expected_ticks
        and script in cmdline
        and experiment in cmdline):
    os.kill(pid, signal.SIGTERM)
PYWATCHDOGSTOP
    fi
    screen -S "$GUARD_SCREEN_NAME" -X quit >/dev/null 2>&1 || true
    exit 4
  fi
  echo "formal_guard_screen=${GUARD_SCREEN_NAME}"
  echo "formal_train_screen=${train_screen_id} start_ticks=${train_screen_start_ticks}"
}

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
