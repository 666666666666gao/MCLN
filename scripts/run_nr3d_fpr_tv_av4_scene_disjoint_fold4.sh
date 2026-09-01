#!/usr/bin/env bash
set -euo pipefail

readonly TRUST_ROOT="/root/mcln_fpr_av4_scene_fold4_trust/v1"
readonly TRUSTED_STATIC_EXEC_PATH="${TRUST_ROOT}/mcln_fpr_audit_static_exec.x86_64"
readonly TRUSTED_STATIC_SOURCE_PATH="${TRUST_ROOT}/mcln_fpr_audit_static_exec.c"
readonly TRUSTED_LAUNCHER_PATH="${TRUST_ROOT}/run_nr3d_fpr_tv_density_audit.sh"
readonly TRUSTED_STATIC_EXEC_SHA256="15ab2d486f1b231ff28eb50fedbeaed1744913a172d18ce139fef50f184c0972"
readonly TRUSTED_STATIC_SOURCE_SHA256="0bf6cfcfb015a91474579ba0c0f186c49c6a38695601d904d3216724cc67dcdc"
[[ "${MCLN_FPR_TRUSTED_CLEAN_ENV:-}" == "1" ]] || {
  echo "launcher must be entered through the reviewed static executor" >&2
  exit 2
}
[[ "${MCLN_FPR_STATIC_EXEC_PATH:-}" == "${TRUSTED_STATIC_EXEC_PATH}"
   && "${MCLN_FPR_STATIC_SOURCE_PATH:-}" == "${TRUSTED_STATIC_SOURCE_PATH}"
   && "${MCLN_FPR_STATIC_EXEC_SHA256:-}" == "${TRUSTED_STATIC_EXEC_SHA256}"
   && "${MCLN_FPR_STATIC_SOURCE_SHA256:-}" == "${TRUSTED_STATIC_SOURCE_SHA256}"
   && "${MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256:-}" =~ ^[0-9a-f]{64}$
   && "${MCLN_FPR_LAUNCHER_FD:-}" == "3"
   && "${MCLN_FPR_LAUNCHER_DEVICE:-}" =~ ^[0-9]+$
   && "${MCLN_FPR_LAUNCHER_INODE:-}" =~ ^[1-9][0-9]*$
   && "${MCLN_FPR_FORMAL_PGID:-}" =~ ^[1-9][0-9]*$
   && "${MCLN_FPR_STATIC_PARENT_PID:-}" =~ ^[1-9][0-9]*$
   && "${MCLN_FPR_STATIC_PARENT_START_TICKS:-}" =~ ^[1-9][0-9]*$ ]] || {
  echo "trusted static-executor provenance is incomplete" >&2
  exit 2
}
[[ "${MCLN_FPR_STATIC_PARENT_PID}" == "${PPID}" ]] || {
  echo "formal launcher parent is not the reviewed static executor" >&2
  exit 2
}
readonly parent_exe="$(/usr/bin/readlink -f "/proc/${PPID}/exe")"
readonly parent_start_ticks="$(/usr/bin/awk '{print $22}' "/proc/${PPID}/stat")"
readonly current_process_group="$(
  /usr/bin/ps -o pgid= -p "$$" | /usr/bin/tr -d ' '
)"
[[ "${parent_exe}" == "${TRUSTED_STATIC_EXEC_PATH}"
   && "${parent_start_ticks}" == "${MCLN_FPR_STATIC_PARENT_START_TICKS}"
   && "${current_process_group}" == "${MCLN_FPR_FORMAL_PGID}" ]] || {
  echo "static-executor process identity changed" >&2
  exit 2
}
mapfile -d '' -t parent_argv < "/proc/${PPID}/cmdline"
[[ ${#parent_argv[@]} -eq 4
   && "${parent_argv[0]}" == "${TRUSTED_STATIC_EXEC_PATH}"
   && "${parent_argv[1]}" == "${MODE:-preflight}"
   && "${parent_argv[2]}" == "${TRUSTED_STATIC_EXEC_SHA256}"
   && "${parent_argv[3]}" == "${MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256}" ]] || {
  echo "static-executor command identity changed" >&2
  exit 2
}
readonly actual_static_exec_sha256="$(
  /usr/bin/sha256sum "${TRUSTED_STATIC_EXEC_PATH}" | /usr/bin/awk '{print $1}'
)"
readonly actual_static_source_sha256="$(
  /usr/bin/sha256sum "${TRUSTED_STATIC_SOURCE_PATH}" | /usr/bin/awk '{print $1}'
)"
readonly consumed_launcher_fd="/proc/$$/fd/3"
readonly consumed_launcher_sha256="$(
  /usr/bin/sha256sum "${consumed_launcher_fd}" | /usr/bin/awk '{print $1}'
)"
[[ "${actual_static_exec_sha256}" == "${TRUSTED_STATIC_EXEC_SHA256}"
   && "${actual_static_source_sha256}" == "${TRUSTED_STATIC_SOURCE_SHA256}"
   && "${consumed_launcher_sha256}" == \
      "${MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256}"
   && "$(/usr/bin/stat -Lc '%d' "${consumed_launcher_fd}")" == \
      "${MCLN_FPR_LAUNCHER_DEVICE}"
   && "$(/usr/bin/stat -Lc '%i' "${consumed_launcher_fd}")" == \
      "${MCLN_FPR_LAUNCHER_INODE}" ]] || {
  echo "trusted static-executor artifact changed" >&2
  exit 2
}
exec 3<&-
[[ "$(/usr/bin/stat -Lc '%u:%g:%a:%F' "${TRUSTED_STATIC_EXEC_PATH}")" \
      == "0:0:755:regular file"
   && "$(/usr/bin/stat -Lc '%u:%g:%a:%F' "${TRUSTED_STATIC_SOURCE_PATH}")" \
      == "0:0:644:regular file"
   && "$(/usr/bin/stat -Lc '%u:%g:%a:%F' "${TRUSTED_LAUNCHER_PATH}")" \
      == "0:0:755:regular file" ]] || {
  echo "trusted static-executor owner or mode changed" >&2
  exit 2
}
readonly static_program_headers="$(
  /usr/bin/readelf -lW "${TRUSTED_STATIC_EXEC_PATH}"
)"
if /usr/bin/grep -Eq '(^|[[:space:]])INTERP([[:space:]]|$)' \
     <<<"${static_program_headers}"; then
  echo "trusted clean-env executor must be statically linked" >&2
  exit 2
fi
if [[ -n "${BASH_ENV:-}" || -n "${ENV:-}" || -n "${LD_PRELOAD:-}"
      || -n "${LD_AUDIT:-}" || -n "${LD_LIBRARY_PATH:-}"
      || -n "${PYTHONOPTIMIZE:-}" || -n "${PYTHONWARNINGS:-}"
      || -n "${PYTHONSTARTUP:-}" || -n "${PYTHONHOME:-}"
      || -n "${PYTHONUSERBASE:-}" || -n "${CDPATH:-}"
      || -n "${GLOBIGNORE:-}" ]]; then
  echo "ambient shell, loader, and Python injection variables are forbidden" >&2
  exit 2
fi
if /usr/bin/env | /usr/bin/grep -Eq '^(SHELLOPTS|BASHOPTS|PS4|BASH_FUNC_)='; then
  echo "exported Bash option, debug, or function variables are forbidden" >&2
  exit 2
fi
mapfile -t inherited_functions < <(compgen -A function || true)
if ((${#inherited_functions[@]} != 0)); then
  echo "inherited shell functions are forbidden: ${inherited_functions[*]}" >&2
  exit 2
fi
unset BASH_ENV ENV CDPATH GLOBIGNORE LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH
unset PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONOPTIMIZE PYTHONWARNINGS
unset PS4
export PATH="/root/miniconda3/envs/bdetr/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
hash -r

terminate_formal_group() {
  trap - HUP INT TERM
  kill -TERM -- "-${MCLN_FPR_FORMAL_PGID}" 2>/dev/null || true
  exit 143
}
trap terminate_formal_group HUP INT TERM

readonly ROOT_DIR="/root/autodl-tmp/mcln_fpr_av4_scene_fold4_review_20260901"
readonly SOURCE_ROOT="/root/autodl-tmp/mcln_fpr_av4_scene_fold4_review_20260901"
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly DATASET="nr3d"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v99_baseline_gt/nr3d"
readonly SOURCE_CHECKPOINT="${OUTPUT_ROOT}/control/tier_hard_query_e57_e58_e62_patience2/formal_input_snapshot_v1/protected_e57.pth"
readonly SOURCE_CHECKPOINT_SHA256="fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
readonly GROUPFREE_CHECKPOINT="${DATA_ROOT%/}/gf_detector_l6o256.pth"
readonly GROUPFREE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly DATA_MANIFEST="${SOURCE_ROOT}/scripts/nr3d_fpr_tv_av4_train_only_data_manifest_v2.json"
readonly DATA_MANIFEST_SHA256="155e2233efbe5c312c19c6dc709ce8c564c601d50e73d6907a3702f000d9d173"
readonly RUNTIME_MANIFEST="${SOURCE_ROOT}/scripts/fpr_tv_av4_scene_fold4_runtime_manifest_v1.json"
readonly RUNTIME_MANIFEST_SHA256="575fe4e15e9a6380ebe3d05c71d30d5d8292e36a135b3d42e881cc158c0cdb1b"
readonly REQUIRED_TRAIN_ENTRY_SHA256="add7ad10e5a91248ccf1c593a280df73b6a19c2cdc3b53b898ce2f64be628c64"
readonly REQUIRED_MAIN_UTILS_SHA256="48bb2034e7c14b2ed17d08fb67f580523173d84815ed48d9303e855d829a1532"
readonly REQUIRED_LOSSES_SHA256="54a74ba522d57d74e934f60a6f8a8becdbe0f188ab99c948ea55c436fbd9fe36"
readonly REQUIRED_VERIFIER_SHA256="de11e5710bd74bfb1110ce035d35feb1a6f595dcc8d8573aed7d9142f92d1628"
readonly REQUIRED_VERIFIER_TEST_SHA256="89552561ddcb35fbc27295dcf2052122bd906f08a8ca2211f7e05e49fa3b5d5a"
readonly REQUIRED_SCENE_TEST_SHA256="f59223ea224c130dc0eb07145fec2f257251393919fbd73f310b5f06bf4283f6"
readonly REQUIRED_FINITE_TEST_SHA256="3556652aa38e4e55a584dcd2d7c6e50b4dfb4da5a92a67abb04dd324ff871214"
readonly REQUIRED_FOLD4_CONTRACT_TEST_SHA256="cf58a03609e7229debabf9c68ef05a6192bb8bf6edbf51a100b36a1401875fda"
readonly REQUIRED_SPEC_SHA256="13f410b5f27da7ec27a866e3d41abb161b407a5221cc457a5221f5b34e7fce95"
readonly REQUIRED_VALIDATOR_SHA256="eb413f1201aa5154f040cc22ba2125a6fff7a61b4be44f35963025fd18cc1055"
readonly REQUIRED_MODEL_SHA256="89ff010c18652cee98f0b4628662138d02844051ad359dc6be2affc596c33ca8"
readonly REQUIRED_SELECTOR_SHA256="61211cea91de4c3f3c11e44bc5ff035711307f3381d5a4069d4ab7842bee17dc"
readonly REQUIRED_DATASET_SHA256="800bac2caf9b7a319bdc200f60386000e4e374a559d9581113a8eb57d525f9f0"
readonly REQUIRED_SNAPSHOT_EXECUTOR_SHA256="839b6d8479b94e610288723219b0149203dde89ab0a85a6dd3bd9d4776d04c88"
readonly CONFIG_SHA256="aaf4d8edc59e99e056f294b4c031467d2570fb43a879099261b4048054ce4177"
readonly FOLD=4
readonly EXPECTED_FIT_SCENES=417
readonly EXPECTED_HOLDOUT_SCENES=94
readonly EXPECTED_FIT_SAMPLES=27004
readonly EXPECTED_HOLDOUT_SAMPLES=5915
readonly EXPECTED_OPTIMIZER_STEPS=1688
readonly V1_OUTPUT_BASE="${OUTPUT_ROOT}/fpr_tv_scene_disjoint_v1"
readonly V1_FOLD0_DECISION="${V1_OUTPUT_BASE}/fold_0/decision.json"
readonly V1_FOLD0_DECISION_SHA256="02f1951b298f0c059a5adc4bbc2b9abe542c8a03f4202a91b9effd1882a22e67"
readonly V1_FOLD0_RECEIPT="${V1_OUTPUT_BASE}/fold_0/runtime_output/nr3d/nr3d_v99_fpr_tv_scene_fold0_e58/1788159699/fpr_scene_disjoint_audit_fold_0_epoch_58.json"
readonly V1_FOLD0_RECEIPT_SHA256="843bf42ceec16b55677ab7279877c115854fe4270f58f364f885631dc33aa8ff"
readonly V1_FOLD4_ROOT="${V1_OUTPUT_BASE}/fold_4"
readonly V2_FOLD1_DECISION="${OUTPUT_ROOT}/fpr_tv_scene_disjoint_v2/fold_1/decision.json"
readonly V2_FOLD1_DECISION_SHA256="beefd711d21ca0a2d314b696c3b99aa8cf0910b30edc1713b898978d59d69cb7"
readonly V2_FOLD1_RECEIPT="${OUTPUT_ROOT}/fpr_tv_scene_disjoint_v2/fold_1/runtime_output/nr3d/nr3d_v99_fpr_tv_v2_scene_fold1_e58/1788170744/fpr_scene_disjoint_audit_fold_1_epoch_58.json"
readonly V2_FOLD1_RECEIPT_SHA256="331c79a390d15494d9475628a24f16461dd1c5fb4a6149a86902a1af3e2df12e"
readonly V2_FOLD4_ROOT="${OUTPUT_ROOT}/fpr_tv_scene_disjoint_v2/fold_4"
readonly DENSITY_FOLD2_DECISION="${OUTPUT_ROOT}/audit/nr3d_v99_density_target_box_scene_fold2_e57_e58_b100_pair_one_shot/paired_decision.json"
readonly DENSITY_FOLD2_DECISION_SHA256="c494fcdf1db53de3babaa9536ea4c9ca29903413de43224793d9698b450191d5"
readonly V3_FOLD3_DECISION="${OUTPUT_ROOT}/fpr_tv_scene_disjoint_v3/fold_3/decision.json"
readonly V3_FOLD3_DECISION_SHA256="d66fd84c901972b11fc0f78f7b70f679f4d3d2f9aae1b1233e6a3105261e072b"
readonly V3_FOLD3_RECEIPT="${OUTPUT_ROOT}/fpr_tv_scene_disjoint_v3/fold_3/runtime_output/nr3d/nr3d_v99_fpr_tv_v3_scene_fold3_e58/1788221033/fpr_scene_disjoint_audit_fold_3_epoch_58.json"
readonly V3_FOLD3_RECEIPT_SHA256="35a53516bc7f5f8bd6f1da53f609734798699d3afddc678d1e025eaec1e2b5aa"
readonly V3_FOLD4_ROOT="${OUTPUT_ROOT}/fpr_tv_scene_disjoint_v3/fold_4"
readonly AV4_MECHANISM_ROOT="${OUTPUT_ROOT}/audit/nr3d_v99_fpr_tv_av4_counterfactual_parent_audit_recovery_v1_e58_b100_b16x1_one_shot"
readonly AV4_MECHANISM_DECISION="${AV4_MECHANISM_ROOT}/counterfactual_parent_decision.json"
readonly AV4_MECHANISM_DECISION_SHA256="aa492439073c60eec8b4cea34538715cc10934f65d53f5f7787e7824b3caec51"
readonly AV4_MECHANISM_RECEIPT="${AV4_MECHANISM_ROOT}/runtime_output/nr3d/nr3d_v99_fpr_tv_av4_counterfactual_parent_audit_recovery_v1_e58_b100_b16x1/1788246651/train_audit_receipt_epoch_58.json"
readonly AV4_MECHANISM_RECEIPT_SHA256="717cea9e3a34f66610526586ce13022846f06830cfeda5b36c2802b6408c0b4e"
readonly REQUIRED_RESUME_EPOCH=57
readonly AUDIT_EPOCH=58
readonly BATCH_SIZE=16
readonly EXPECTED_EVAL_SAMPLE_COUNT="${EXPECTED_HOLDOUT_SAMPLES}"
readonly MASTER_PORT=5434
readonly MIN_FREE_GB=5
readonly SNAPSHOT_OWNER_UID=65532
readonly SNAPSHOT_OWNER_GID=65532
readonly EXP="nr3d_v99_fpr_tv_av4_scene_fold4_e58"
readonly OUTPUT_BASE="${OUTPUT_ROOT}/fpr_tv_av4_scene_disjoint_v1"
readonly AUDIT_ROOT="${OUTPUT_BASE}/fold_${FOLD}"
readonly LOCK_FILE="/root/autodl-tmp/mcln_v99_backbone_gpu0.lock"

MODE="${MODE:-preflight}"
readonly MODE
case "${MODE}" in
  preflight|backbone) ;;
  *)
    echo "audit launcher supports only MODE=preflight or MODE=backbone" >&2
    exit 2
    ;;
esac
if (($# != 0)); then
  echo "usage: MODE=preflight|backbone $0" >&2
  exit 2
fi
readonly LAUNCHER_PATH="${TRUSTED_LAUNCHER_PATH}"
readonly REVIEWED_LAUNCHER_SHA256="${MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256}"
readonly OBSERVED_LAUNCHER_SHA256="${consumed_launcher_sha256}"
unset PYTHONPATH
cd "${ROOT_DIR}"

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

readonly LAUNCHER_START_SHA256="${OBSERVED_LAUNCHER_SHA256}"

require_sha256() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || {
    echo "missing ${label}: ${path}" >&2
    exit 3
  }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA changed: ${actual}" >&2
    exit 3
  }
}

require_fixed_inputs() {
  require_sha256 "${ROOT_DIR}/train_dist_mod.py" \
    "${REQUIRED_TRAIN_ENTRY_SHA256}" "training entrypoint"
  require_sha256 "${ROOT_DIR}/main_utils.py" \
    "${REQUIRED_MAIN_UTILS_SHA256}" "main_utils"
  require_sha256 "${ROOT_DIR}/models/losses.py" \
    "${REQUIRED_LOSSES_SHA256}" "loss implementation"
  require_sha256 "${ROOT_DIR}/models/parent_relative_text_verifier.py" \
    "${REQUIRED_VERIFIER_SHA256}" "Parent-relative verifier"
  require_sha256 "${ROOT_DIR}/tests/test_parent_relative_text_verifier.py" \
    "${REQUIRED_VERIFIER_TEST_SHA256}" "Parent-relative verifier tests"
  require_sha256 "${ROOT_DIR}/tests/test_fpr_scene_disjoint_audit.py" \
    "${REQUIRED_SCENE_TEST_SHA256}" "scene-disjoint audit tests"
  require_sha256 "${ROOT_DIR}/tests/test_main_utils_finite_training.py" \
    "${REQUIRED_FINITE_TEST_SHA256}" "finite-training tests"
  require_sha256 "${ROOT_DIR}/tests/test_fpr_av4_scene_fold4_contract.py" \
    "${REQUIRED_FOLD4_CONTRACT_TEST_SHA256}" "fold4 contract tests"
  require_sha256 "${ROOT_DIR}/FPR_TV_AV4_SCENE_DISJOINT_FOLD4_SPEC_2026-09-01.md" \
    "${REQUIRED_SPEC_SHA256}" "A-V4 fold4 specification"
  require_sha256 "${ROOT_DIR}/scripts/validate_nr3d_fpr_tv_av4_scene_fold4.py" \
    "${REQUIRED_VALIDATOR_SHA256}" "fold4 decision validator"
  require_sha256 "${ROOT_DIR}/models/mcln.py" \
    "${REQUIRED_MODEL_SHA256}" "MCLN model"
  require_sha256 "${ROOT_DIR}/models/source_choice_selector.py" \
    "${REQUIRED_SELECTOR_SHA256}" "source-choice selector"
  require_sha256 "${ROOT_DIR}/src/joint_det_dataset.py" \
    "${REQUIRED_DATASET_SHA256}" "dataset implementation"
  require_sha256 "${ROOT_DIR}/scripts/mcln_density_audit_snapshot_exec.py" \
    "${REQUIRED_SNAPSHOT_EXECUTOR_SHA256}" "capability-drop executor"
  require_sha256 "${V1_FOLD0_DECISION}" \
    "${V1_FOLD0_DECISION_SHA256}" "FPR v1 fold0 decision"
  require_sha256 "${V1_FOLD0_RECEIPT}" \
    "${V1_FOLD0_RECEIPT_SHA256}" "FPR v1 fold0 receipt"
  require_sha256 "${V2_FOLD1_DECISION}" \
    "${V2_FOLD1_DECISION_SHA256}" "FPR v2 fold1 decision"
  require_sha256 "${V2_FOLD1_RECEIPT}" \
    "${V2_FOLD1_RECEIPT_SHA256}" "FPR v2 fold1 receipt"
  require_sha256 "${DENSITY_FOLD2_DECISION}" \
    "${DENSITY_FOLD2_DECISION_SHA256}" "density fold2 decision"
  require_sha256 "${V3_FOLD3_DECISION}" \
    "${V3_FOLD3_DECISION_SHA256}" "FPR v3 fold3 decision"
  require_sha256 "${V3_FOLD3_RECEIPT}" \
    "${V3_FOLD3_RECEIPT_SHA256}" "FPR v3 fold3 receipt"
  require_sha256 "${AV4_MECHANISM_DECISION}" \
    "${AV4_MECHANISM_DECISION_SHA256}" "A-V4 mechanism decision"
  require_sha256 "${AV4_MECHANISM_RECEIPT}" \
    "${AV4_MECHANISM_RECEIPT_SHA256}" "A-V4 mechanism receipt"
  require_sha256 "${SOURCE_CHECKPOINT}" \
    "${SOURCE_CHECKPOINT_SHA256}" "protected E57 checkpoint"
  require_sha256 "${GROUPFREE_CHECKPOINT}" \
    "${GROUPFREE_SHA256}" "GroupFree checkpoint"
  require_sha256 "${DATA_MANIFEST}" \
    "${DATA_MANIFEST_SHA256}" "Nr3D data manifest"
  require_sha256 "${RUNTIME_MANIFEST}" \
    "${RUNTIME_MANIFEST_SHA256}" "reviewed runtime manifest"
}

verify_preregistered_history() {
  local current_root_state="${1:-absent}"
  /usr/bin/env -i \
    PATH="${PATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "${PYTHON_BIN}" - \
    "${V1_FOLD0_DECISION}" "${V1_FOLD0_DECISION_SHA256}" \
    "${V1_FOLD0_RECEIPT}" "${V1_FOLD0_RECEIPT_SHA256}" \
    "${V2_FOLD1_DECISION}" "${V2_FOLD1_DECISION_SHA256}" \
    "${V2_FOLD1_RECEIPT}" "${V2_FOLD1_RECEIPT_SHA256}" \
    "${DENSITY_FOLD2_DECISION}" "${DENSITY_FOLD2_DECISION_SHA256}" \
    "${V3_FOLD3_DECISION}" "${V3_FOLD3_DECISION_SHA256}" \
    "${V3_FOLD3_RECEIPT}" "${V3_FOLD3_RECEIPT_SHA256}" \
    "${AV4_MECHANISM_DECISION}" "${AV4_MECHANISM_DECISION_SHA256}" \
    "${AV4_MECHANISM_RECEIPT}" "${AV4_MECHANISM_RECEIPT_SHA256}" \
    "${current_root_state}" \
    "${V1_FOLD4_ROOT}" "${V2_FOLD4_ROOT}" "${V3_FOLD4_ROOT}" \
    "${AUDIT_ROOT}" <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import stat
import sys

pairs = [tuple(sys.argv[index:index + 2]) for index in range(1, 19, 2)]
current_root_state = sys.argv[19]
historical_absent_roots = sys.argv[20:23]
current_root = sys.argv[23]


def load_frozen_json(path, expected_sha):
    if os.path.islink(path):
        raise SystemExit("historical evidence symlink rejected")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("historical evidence is not regular")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev, after.st_ino, after.st_size):
        raise SystemExit("historical evidence changed while reading")
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise SystemExit("historical evidence SHA mismatch")
    return json.loads(raw.decode("utf-8"))


fold0_decision = load_frozen_json(*pairs[0])
fold0_receipt = load_frozen_json(*pairs[1])
if (fold0_decision.get("schema") != "mcln-fpr-tv-scene-fold-decision-v1"
        or fold0_decision.get("fold") != 0
        or fold0_decision.get("fold_gate_pass") is not False
        or fold0_decision.get("long_training_authorized") is not False
        or fold0_decision.get("receipt_path") != pairs[1][0]
        or fold0_decision.get("receipt_sha256") != pairs[1][1]
        or fold0_decision.get("switch_count") != 876):
    raise SystemExit("FPR v1 fold0 evidence mismatch")
for suffix, fix_count, break_count in (("025", 40, 82), ("050", 119, 398)):
    transition = fold0_decision.get("thresholds", {}).get(suffix, {})
    if (transition.get("fix_count") != fix_count
            or transition.get("break_count") != break_count):
        raise SystemExit("FPR v1 fold0 transition mismatch")
if (fold0_receipt.get("split", {}).get("fold") != 0
        or fold0_receipt.get("fold_gate_pass") is not False
        or fold0_receipt.get("long_training_authorized") is not False):
    raise SystemExit("FPR v1 fold0 receipt mismatch")

fold1_decision = load_frozen_json(*pairs[2])
fold1_receipt = load_frozen_json(*pairs[3])
if (fold1_decision.get("schema") != "mcln-fpr-tv-scene-fold-decision-v1"
        or fold1_decision.get("fold") != 1
        or fold1_decision.get("fold_gate_pass") is not False
        or fold1_decision.get("long_training_authorized") is not False
        or fold1_decision.get("receipt_path") != pairs[3][0]
        or fold1_decision.get("receipt_sha256") != pairs[3][1]
        or fold1_decision.get("switch_count") != 0):
    raise SystemExit("FPR v2 fold1 evidence mismatch")
for suffix in ("025", "050"):
    transition = fold1_decision.get("thresholds", {}).get(suffix, {})
    if any(transition.get(key) != 0 for key in (
            "fix_count", "break_count", "net_hits")):
        raise SystemExit("FPR v2 fold1 transition mismatch")
if (fold1_receipt.get("split", {}).get("fold") != 1
        or fold1_receipt.get("fold_gate_pass") is not False
        or fold1_receipt.get("long_training_authorized") is not False):
    raise SystemExit("FPR v2 fold1 receipt mismatch")

density = load_frozen_json(*pairs[4])
if (density.get("schema") !=
        "mcln-density-target-box-scene-disjoint-decision-v1"
        or density.get("split", {}).get("fold") != 2
        or density.get("density_gate_passed") is not False
        or density.get("long_training_authorized") is not False
        or density.get("next_allowed_step") != "seal_method"):
    raise SystemExit("density fold2 evidence mismatch")

fold3_decision = load_frozen_json(*pairs[5])
fold3_receipt = load_frozen_json(*pairs[6])
if (fold3_decision.get("schema") != "mcln-fpr-tv-scene-fold-decision-v3"
        or fold3_decision.get("fold") != 3
        or fold3_decision.get("fold_gate_pass") is not False
        or fold3_decision.get("long_training_authorized") is not False
        or fold3_decision.get("receipt_path") != pairs[6][0]
        or fold3_decision.get("receipt_sha256") != pairs[6][1]
        or fold3_decision.get("switch_count") != 844):
    raise SystemExit("FPR v3 fold3 evidence mismatch")
for suffix, fix_count, break_count in (("025", 23, 38), ("050", 104, 283)):
    transition = fold3_decision.get("thresholds", {}).get(suffix, {})
    if (transition.get("fix_count") != fix_count
            or transition.get("break_count") != break_count):
        raise SystemExit("FPR v3 fold3 transition mismatch")
if (fold3_receipt.get("split", {}).get("fold") != 3
        or fold3_receipt.get("fold_gate_pass") is not False
        or fold3_receipt.get("long_training_authorized") is not False):
    raise SystemExit("FPR v3 fold3 receipt mismatch")

av4_decision = load_frozen_json(*pairs[7])
av4_receipt = load_frozen_json(*pairs[8])
checks = av4_decision.get("checks")
if (av4_decision.get("schema") !=
        "mcln-fpr-tv-counterfactual-parent-audit-recovery-v2"
        or av4_decision.get("counterfactual_density_gate_passed") is not True
        or av4_decision.get("long_training_authorized") is not False
        or av4_decision.get("formal_validation_accessed") is not False
        or av4_decision.get("audit_batches") != 100
        or av4_decision.get("sample_count") != 1600
        or av4_decision.get("checkpoint_epoch") != 57
        or av4_decision.get("checkpoint_sha256") !=
        "fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
        or av4_decision.get("receipt") != pairs[8][0]
        or av4_decision.get("receipt_sha256") != pairs[8][1]
        or not isinstance(checks, dict) or not checks
        or not all(value is True for value in checks.values())):
    raise SystemExit("A-V4 mechanism decision mismatch")
if (av4_receipt.get("schema") != "mcln-train-loss-epoch-v1"
        or av4_receipt.get("epoch") != 58
        or av4_receipt.get("batch_count") != 100
        or av4_receipt.get("optimizer_step_count") != 100
        or av4_receipt.get("sample_count") != 1600
        or av4_receipt.get("audit_only") is not True
        or av4_receipt.get("formal_validation_accessed") is not False
        or av4_receipt.get("long_training_authorized") is not False):
    raise SystemExit("A-V4 mechanism receipt mismatch")

for root in historical_absent_roots:
    if os.path.lexists(root):
        raise SystemExit("fold4 one-shot root already exists: {}".format(root))
if current_root_state == "absent":
    if os.path.lexists(current_root):
        raise SystemExit("fold4 one-shot root already exists: {}".format(
            current_root))
elif current_root_state == "consumed":
    if (not os.path.isdir(current_root)
            or os.path.islink(current_root)
            or os.path.realpath(current_root) != current_root):
        raise SystemExit("current fold4 root is not the consumed canonical directory")
else:
    raise SystemExit("unsupported current fold4 root state")
print("preregistered_history_verified=fold0_fold1_fold2_fold3_av4_mechanism")
PY
}

verify_or_copy_runtime_closure() {
  local mode="$1" root="$2" destination="${3:--}"
  local manifest_path="${4:-${RUNTIME_MANIFEST}}"
  /usr/bin/env -i \
    PATH="${PATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "${PYTHON_BIN}" - "${manifest_path}" \
    "${RUNTIME_MANIFEST_SHA256}" "${mode}" "${root}" \
    "${destination}" <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import stat
import sys

manifest_path, expected_sha, mode, root, destination = sys.argv[1:]
snapshot_owner_uid = 65532
snapshot_owner_gid = 65532
with open(manifest_path, "rb") as handle:
    raw = handle.read()
if hashlib.sha256(raw).hexdigest() != expected_sha:
    raise SystemExit("runtime manifest SHA-256 mismatch")
manifest = json.loads(raw.decode("utf-8"))
if manifest.get("schema") != (
        "mcln-fpr-tv-av4-scene-fold4-reviewed-runtime-v1"):
    raise SystemExit("unexpected runtime manifest schema")
records = manifest.get("files")
if not isinstance(records, dict) or len(records) != manifest.get("file_count"):
    raise SystemExit("runtime manifest record count mismatch")
if manifest.get("total_size") != sum(
        record.get("size", -1) for record in records.values()):
    raise SystemExit("runtime manifest total size mismatch")

def checked_relative(value):
    if not isinstance(value, str) or not value:
        raise SystemExit("invalid empty runtime path")
    normalized = os.path.normpath(value)
    if (normalized != value or os.path.isabs(value)
            or value == ".." or value.startswith("../")):
        raise SystemExit("unsafe runtime path: {}".format(value))
    return value

def read_verified(path, record, expected_mode, expected_owner=None):
    if os.path.islink(path):
        raise SystemExit("runtime symlink rejected: {}".format(path))
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("runtime path is not regular: {}".format(path))
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev, after.st_ino, after.st_size):
        raise SystemExit("runtime file changed while reading: {}".format(path))
    payload = b"".join(chunks)
    if format(stat.S_IMODE(after.st_mode), "04o") != expected_mode:
        raise SystemExit("runtime mode mismatch: {}".format(path))
    if expected_owner is not None and (
            after.st_uid != expected_owner[0]
            or after.st_gid != expected_owner[1]):
        raise SystemExit("runtime owner mismatch: {}".format(path))
    if len(payload) != record.get("size"):
        raise SystemExit("runtime size mismatch: {}".format(path))
    if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
        raise SystemExit("runtime SHA mismatch: {}".format(path))
    return payload

root = os.path.realpath(root)
if mode in ("verify-source", "copy") and manifest.get("source_root") != root:
    raise SystemExit("runtime manifest source_root mismatch")
if mode == "copy":
    if not os.path.isdir(destination) or os.listdir(destination):
        raise SystemExit("runtime copy destination must be empty")
for relative, record in sorted(records.items()):
    relative = checked_relative(relative)
    source = os.path.join(root, relative)
    if mode in ("verify-source", "copy"):
        payload = read_verified(source, record, record.get("mode"))
    elif mode == "verify-snapshot":
        payload = read_verified(
            source,
            record,
            "0444",
            (snapshot_owner_uid, snapshot_owner_gid),
        )
    else:
        raise SystemExit("unknown runtime closure mode")
    if mode == "copy":
        target = os.path.join(destination, relative)
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, mode=0o755, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(fd, payload[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chown(target, snapshot_owner_uid, snapshot_owner_gid)
        os.chmod(target, 0o444)
if mode == "copy":
    for current, directories, _files in os.walk(destination, topdown=False):
        for name in directories:
            directory = os.path.join(current, name)
            os.chown(directory, snapshot_owner_uid, snapshot_owner_gid)
            os.chmod(directory, 0o555)
    os.chown(destination, snapshot_owner_uid, snapshot_owner_gid)
    os.chmod(destination, 0o555)
if mode == "verify-snapshot":
    observed = set()
    for current, directories, files in os.walk(root):
        current_info = os.lstat(current)
        if (
                current_info.st_uid != snapshot_owner_uid
                or current_info.st_gid != snapshot_owner_gid
                or stat.S_IMODE(current_info.st_mode) != 0o555):
            raise SystemExit("snapshot directory contract changed")
        for name in directories:
            if os.path.islink(os.path.join(current, name)):
                raise SystemExit("snapshot directory symlink rejected")
        for name in files:
            path = os.path.join(current, name)
            if os.path.islink(path):
                raise SystemExit("snapshot file symlink rejected")
            observed.add(os.path.relpath(path, root).replace(os.sep, "/"))
    if observed != set(records):
        raise SystemExit("runtime snapshot file set mismatch")
print("runtime_closure_{}={} files".format(mode, len(records)))
PY
}

verify_dataset_manifest() {
  local manifest_path="$1" expected_sha="$2"
  /usr/bin/env -i \
    PATH="${PATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    DATA_ROOT_ENV="$(readlink -f -- "${DATA_ROOT}")" \
    DATA_MANIFEST_ENV="${manifest_path}" DATA_MANIFEST_SHA_ENV="${expected_sha}" \
    "${PYTHON_BIN}" - <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import stat

root = os.environ["DATA_ROOT_ENV"]
manifest_path = os.environ["DATA_MANIFEST_ENV"]
expected_sha = os.environ["DATA_MANIFEST_SHA_ENV"]
expected_sources = [
    "train_v3scans.pkl",
    "refer_it_3d/nr3d.csv",
    "roberta-base",
    "superpoints/train",
    "group_free_pred_bboxes/group_free_pred_bboxes_train",
]

def require_real_descendant(relative):
    current = root
    for component in relative.split("/"):
        current = os.path.join(current, component)
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise SystemExit("dataset source component is a symlink: " + current)
    real = os.path.realpath(current)
    if os.path.commonpath([root, real]) != root:
        raise SystemExit("dataset source escaped DATA_ROOT: " + current)
    return current

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

with open(manifest_path, "rb") as handle:
    raw = handle.read()
if hashlib.sha256(raw).hexdigest() != expected_sha:
    raise SystemExit("dataset manifest SHA changed")
manifest = json.loads(raw.decode("utf-8"))
if manifest.get("schema") != "mcln-nr3d-fpr-tv-av4-train-only-data-manifest-v2":
    raise SystemExit("dataset manifest schema changed")
if manifest.get("data_root") != root:
    raise SystemExit("dataset manifest root changed")
if manifest.get("sources") != expected_sources:
    raise SystemExit("dataset manifest source closure changed")

current_paths = []
for relative_source in expected_sources:
    source = require_real_descendant(relative_source)
    if not os.path.exists(source):
        raise SystemExit("dataset input is missing: " + source)
    if os.path.isdir(source):
        for current, directories, files in os.walk(source):
            directories.sort()
            files.sort()
            for name in directories:
                if os.path.islink(os.path.join(current, name)):
                    raise SystemExit("dataset directory symlink")
            current_paths.extend(
                os.path.join(current, name) for name in files
            )
    else:
        current_paths.append(source)

rows = manifest.get("files")
if not isinstance(rows, list):
    raise SystemExit("dataset manifest lacks files")
if [row.get("path") for row in rows] != [
        os.path.relpath(path, root) for path in current_paths]:
    raise SystemExit("dataset file inventory changed")
if manifest.get("file_count") != len(rows):
    raise SystemExit("dataset manifest file count changed")
if manifest.get("total_size") != sum(row.get("size", -1) for row in rows):
    raise SystemExit("dataset manifest total size changed")
for path, row in zip(current_paths, rows):
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit("dataset input is not regular: " + path)
    if int(info.st_size) != row.get("size"):
        raise SystemExit("dataset input size changed: " + path)
    if int(stat.S_IMODE(info.st_mode)) != row.get("mode"):
        raise SystemExit("dataset input mode changed: " + path)
    if sha256_file(path) != row.get("sha256"):
        raise SystemExit("dataset input SHA changed: " + path)
print("dataset_manifest_verified={} files={} bytes={}".format(
    expected_sha, len(rows), manifest["total_size"]
))
PY
}

copy_independent_input() {
  local source="$1" destination="$2" expected_sha="$3" label="$4"
  cp --reflink=auto -- "${source}" "${destination}"
  if [[ "$(stat -Lc '%d:%i' -- "${source}")" == \
        "$(stat -Lc '%d:%i' -- "${destination}")" ]]; then
    echo "${label} snapshot did not receive an independent inode" >&2
    exit 7
  fi
  chown "${SNAPSHOT_OWNER_UID}:${SNAPSHOT_OWNER_GID}" "${destination}"
  chmod 0444 "${destination}"
  require_sha256 "${destination}" "${expected_sha}" "${label} snapshot"
}

verify_independent_input() {
  local source="$1" snapshot="$2" expected_sha="$3" label="$4"
  [[ -f "${source}" && ! -L "${source}" ]] || {
    echo "${label} source is not a regular non-symlink file" >&2
    exit 7
  }
  [[ -f "${snapshot}" && ! -L "${snapshot}" ]] || {
    echo "${label} snapshot is not a regular non-symlink file" >&2
    exit 7
  }
  if [[ "$(stat -Lc '%d:%i' -- "${source}")" == \
        "$(stat -Lc '%d:%i' -- "${snapshot}")" ]]; then
    echo "${label} snapshot no longer has an independent inode" >&2
    exit 7
  fi
  if [[ "$(stat -Lc '%a' -- "${snapshot}")" != "444" ]]; then
    echo "${label} snapshot mode changed" >&2
    exit 7
  fi
  if [[ "$(stat -Lc '%u:%g' -- "${snapshot}")" != \
        "${SNAPSHOT_OWNER_UID}:${SNAPSHOT_OWNER_GID}" ]]; then
    echo "${label} snapshot owner changed" >&2
    exit 7
  fi
  require_sha256 "${source}" "${expected_sha}" "${label} source"
  require_sha256 "${snapshot}" "${expected_sha}" "${label} snapshot"
}

verify_checkpoint_contract() {
  /usr/bin/env -i \
    PATH="${PATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "${PYTHON_BIN}" - "${SOURCE_CHECKPOINT}" \
    "${SOURCE_CHECKPOINT_SHA256}" <<'PY'
import hashlib
import math
import sys

import torch

path, expected_sha256 = sys.argv[1:]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise SystemExit("protected checkpoint SHA changed before load")
    handle.seek(0)
    checkpoint = torch.load(handle, map_location="cpu")
    handle.seek(0)
    second = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        second.update(chunk)
if second.hexdigest() != expected_sha256:
    raise SystemExit("protected checkpoint changed during load")

config = checkpoint.get("config")
config = vars(config) if hasattr(config, "__dict__") else dict(config or {})
optimizer = checkpoint.get("optimizer", {})
groups = optimizer.get("param_groups", [])
expected_lrs = [1e-5, 1e-4, 1e-5, 1.25e-5]
actual_lrs = [group.get("lr") for group in groups]
if checkpoint.get("epoch") != 57:
    raise SystemExit("protected checkpoint is not completed E57")
if config.get("batch_size") != 16:
    raise SystemExit("protected checkpoint is not B16")
if config.get("gradient_accumulation_steps", 1) != 1:
    raise SystemExit("protected checkpoint is not accumulation 1")
if config.get("joint_det") is not True or config.get("butd_cls") is not True:
    raise SystemExit("protected checkpoint is not joint_det + butd_cls")
if config.get("use_source_choice_selector") is not True:
    raise SystemExit("protected checkpoint is not the V99 selector model")
if config.get("source_choice_selector_sources") != (
        "default,default_rank_blend_contrastive010"):
    raise SystemExit("protected checkpoint V99 sources changed")
if len(groups) != 4 or len(optimizer.get("state", {})) != 716:
    raise SystemExit("protected optimizer topology changed")
if len(actual_lrs) != len(expected_lrs) or any(
        value is None or not math.isclose(
            float(value), expected, rel_tol=0.0, abs_tol=1e-12
        ) for value, expected in zip(actual_lrs, expected_lrs)):
    raise SystemExit("protected optimizer current LRs changed")
print("checkpoint_contract=E57_B16x1_joint_det_butd_cls_V99_full_state")
PY
}

run_default_off_regression() {
  /usr/bin/env -i \
    PATH="${PATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/pointnet2" \
    "${PYTHON_BIN}" -m pytest -q -p no:cacheprovider \
      tests/test_parent_relative_text_verifier.py \
      tests/test_mcln_training_groups.py \
      tests/test_fpr_scene_disjoint_audit.py \
      tests/test_fpr_av4_audit_contract.py \
      tests/test_fpr_av4_scene_fold4_contract.py
}

build_train_args() {
  local log_dir="$1" groupfree_checkpoint="$2" resume_checkpoint="$3"
  train_args=(
    --num_target 256 --sampling kps
    --num_encoder_layers 3 --num_decoder_layers 6
    --self_position_embedding loc_learned --query_points_obj_topk 4
    --use_color --weight_decay 0.0005
    --data_root "${DATA_ROOT}"
    --val_freq 1 --batch_size "${BATCH_SIZE}"
    --num_workers 4 --dataloader_prefetch_factor 2 --persistent_train_workers
    --save_freq 1 --print_freq 20 --clip_norm 0.1 --rng_seed 0
    --lr_backbone 1e-3 --lr 1e-4 --lr_decay_epochs 150 --warmup-epoch -1
    --dataset "${DATASET}" --test_dataset "${DATASET}"
    --joint_det --butd_cls
    --max_train_batches 0 --gradient_accumulation_steps 1
    --local_rank 0
    --detect_intermediate --use_soft_token_loss --use_contrastive_align
    --log_dir "${log_dir}"
    --pp_checkpoint "${groupfree_checkpoint}"
    --pp_checkpoint_sha256 "${GROUPFREE_SHA256}"
    --self_attend --skip_missing_superpoints
    --checkpoint_path "${resume_checkpoint}"
    --resume_lr_scale 1.0
    --start_epoch "${AUDIT_EPOCH}" --max_epoch "${AUDIT_EPOCH}"
    --model MCLN --exp "${EXP}"
    --use_source_choice_selector --eval_use_selector_choice_scores
    --source_choice_selector_sources default,default_rank_blend_contrastive010
    --source_choice_selector_default_source default
    --source_choice_selector_hidden_dim 288
    --source_choice_selector_lr 1.25e-4
    --source_choice_selector_loss_weight 0.5
    --source_choice_selector_choice_target precision_gain_default_sourcewise_focal_bce
    --source_choice_selector_min_iou_gap 0.03
    --use_parent_relative_text_verifier
    --parent_relative_text_verifier_train_only
    --parent_relative_text_verifier_counterfactual_training
    --parent_relative_text_verifier_top_k 5
    --parent_relative_text_verifier_max_candidates 10
    --parent_relative_text_verifier_hidden_dim 256
    --parent_relative_text_verifier_heads 4
    --parent_relative_text_verifier_dropout 0.1
    --parent_relative_text_verifier_max_parent_score_gap 0.25
    --parent_relative_text_verifier_promotion_margin 0.0001
    --parent_relative_text_verifier_min_parse_confidence 0.5
    --parent_relative_text_verifier_min_anchor_mass 0.5
    --parent_relative_text_verifier_promotion_epsilon 0.0001
    --parent_relative_text_verifier_lr 0.0003
    --parent_relative_text_verifier_loss_weight 1.0
    --parent_relative_text_verifier_positive_margin 0.25
    --parent_relative_text_verifier_neutral_margin 0.25
    --fpr_scene_disjoint_audit
    --fpr_scene_disjoint_av4_audit
    --fpr_scene_disjoint_fold "${FOLD}"
    --fpr_scene_disjoint_expected_fit_scenes "${EXPECTED_FIT_SCENES}"
    --fpr_scene_disjoint_expected_holdout_scenes "${EXPECTED_HOLDOUT_SCENES}"
    --fpr_scene_disjoint_expected_fit_samples "${EXPECTED_FIT_SAMPLES}"
    --fpr_scene_disjoint_expected_holdout_samples "${EXPECTED_HOLDOUT_SAMPLES}"
    --fpr_scene_disjoint_checkpoint_sha256 "${SOURCE_CHECKPOINT_SHA256}"
    --expected_eval_sample_count "${EXPECTED_EVAL_SAMPLE_COUNT}"
  )
}

verify_preregistered_config() {
  /usr/bin/env -i \
    PATH="${PATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/pointnet2" \
    "${PYTHON_BIN}" - "${CONFIG_SHA256}" "${train_args[@]}" <<'PY'
from __future__ import print_function

import sys

from main_utils import (
    build_fpr_scene_disjoint_config_receipt,
    parse_option,
    prepare_source_moe_gate_checkpoint_config,
)

expected = sys.argv[1]
sys.argv = ["train_dist_mod.py"] + sys.argv[2:]
args = prepare_source_moe_gate_checkpoint_config(parse_option())
receipt = build_fpr_scene_disjoint_config_receipt(args)
if receipt.get("sha256") != expected:
    raise SystemExit("A-V4 fold4 preregistered configuration mismatch")
print("preregistered_config_verified={}".format(expected))
PY
}

check_resources() {
  local free_kb free_gb gpu_used
  free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
  free_gb="$((free_kb / 1024 / 1024))"
  gpu_used="$(nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits -i 0 | tr -d ' ')"
  if ((gpu_used >= 500)); then
    echo "GPU0 is busy (${gpu_used} MiB)" >&2
    exit 4
  fi
  if ((free_gb < MIN_FREE_GB)); then
    echo "need at least ${MIN_FREE_GB} GiB free under DATA_ROOT" >&2
    exit 5
  fi
}

build_train_args \
  "${AUDIT_ROOT}/runtime_output" \
  "${GROUPFREE_CHECKPOINT}" \
  "${SOURCE_CHECKPOINT}"
require_fixed_inputs
verify_preregistered_history absent
verify_or_copy_runtime_closure verify-source "${SOURCE_ROOT}"
verify_dataset_manifest "${DATA_MANIFEST}" "${DATA_MANIFEST_SHA256}"
verify_checkpoint_contract
run_default_off_regression
verify_preregistered_config
check_resources

static_parent_holds_lock=false
for descriptor in "/proc/${PPID}/fd/"*; do
  if [[ "$(/usr/bin/readlink -f -- "${descriptor}" 2>/dev/null || true)" == \
        "${LOCK_FILE}" ]]; then
    static_parent_holds_lock=true
    break
  fi
done
if [[ "${static_parent_holds_lock}" != "true" ]]; then
  echo "reviewed static executor does not own the global GPU lock" >&2
  exit 6
fi
if [[ "${MODE}" == "preflight" ]]; then
  if [[ -e "${AUDIT_ROOT}" ]]; then
    echo "one-shot audit root is already consumed: ${AUDIT_ROOT}" >&2
    exit 7
  fi
  echo "preflight=pass audit_only=true fold=${FOLD} fit=${EXPECTED_FIT_SAMPLES}/${EXPECTED_FIT_SCENES} holdout=${EXPECTED_HOLDOUT_SAMPLES}/${EXPECTED_HOLDOUT_SCENES} config_sha256=${CONFIG_SHA256}"
  exit 0
fi
check_resources

mkdir -p "$(dirname "${AUDIT_ROOT}")"
if ! mkdir "${AUDIT_ROOT}"; then
  echo "one-shot audit root is already consumed: ${AUDIT_ROOT}" >&2
  exit 7
fi
readonly INPUT_ROOT="${AUDIT_ROOT}/input_snapshot"
readonly RUNTIME_OUTPUT="${AUDIT_ROOT}/runtime_output"
readonly RUNTIME_HOME="${AUDIT_ROOT}/runtime_home"
readonly CODE_SNAPSHOT="${AUDIT_ROOT}/code_snapshot"
mkdir "${INPUT_ROOT}" "${RUNTIME_OUTPUT}" "${RUNTIME_HOME}" \
  "${CODE_SNAPSHOT}"
readonly RESUME_SNAPSHOT="${INPUT_ROOT}/protected_e57.pth"
readonly GROUPFREE_SNAPSHOT="${INPUT_ROOT}/gf_detector_l6o256.pth"
readonly DATA_MANIFEST_SNAPSHOT="${INPUT_ROOT}/nr3d_fpr_tv_av4_train_only_data_manifest_v2.json"
readonly RUNTIME_MANIFEST_SNAPSHOT="${INPUT_ROOT}/fpr_tv_av4_scene_fold4_runtime_manifest_v1.json"
readonly V1_FOLD0_DECISION_SNAPSHOT="${INPUT_ROOT}/fpr_v1_fold0_decision.json"
readonly V1_FOLD0_RECEIPT_SNAPSHOT="${INPUT_ROOT}/fpr_v1_fold0_receipt.json"
readonly V2_FOLD1_DECISION_SNAPSHOT="${INPUT_ROOT}/fpr_v2_fold1_decision.json"
readonly V2_FOLD1_RECEIPT_SNAPSHOT="${INPUT_ROOT}/fpr_v2_fold1_receipt.json"
readonly DENSITY_FOLD2_DECISION_SNAPSHOT="${INPUT_ROOT}/density_fold2_decision.json"
readonly V3_FOLD3_DECISION_SNAPSHOT="${INPUT_ROOT}/fpr_v3_fold3_decision.json"
readonly V3_FOLD3_RECEIPT_SNAPSHOT="${INPUT_ROOT}/fpr_v3_fold3_receipt.json"
readonly AV4_MECHANISM_DECISION_SNAPSHOT="${INPUT_ROOT}/av4_mechanism_decision.json"
readonly AV4_MECHANISM_RECEIPT_SNAPSHOT="${INPUT_ROOT}/av4_mechanism_receipt.json"
verify_or_copy_runtime_closure copy "${SOURCE_ROOT}" "${CODE_SNAPSHOT}"
verify_or_copy_runtime_closure verify-snapshot "${CODE_SNAPSHOT}"
copy_independent_input "${SOURCE_CHECKPOINT}" "${RESUME_SNAPSHOT}" \
  "${SOURCE_CHECKPOINT_SHA256}" "E57"
copy_independent_input "${GROUPFREE_CHECKPOINT}" "${GROUPFREE_SNAPSHOT}" \
  "${GROUPFREE_SHA256}" "GroupFree"
copy_independent_input "${DATA_MANIFEST}" "${DATA_MANIFEST_SNAPSHOT}" \
  "${DATA_MANIFEST_SHA256}" "data manifest"
copy_independent_input "${RUNTIME_MANIFEST}" "${RUNTIME_MANIFEST_SNAPSHOT}" \
  "${RUNTIME_MANIFEST_SHA256}" "runtime manifest"
copy_independent_input "${V1_FOLD0_DECISION}" "${V1_FOLD0_DECISION_SNAPSHOT}" \
  "${V1_FOLD0_DECISION_SHA256}" "FPR v1 fold0 decision"
copy_independent_input "${V1_FOLD0_RECEIPT}" "${V1_FOLD0_RECEIPT_SNAPSHOT}" \
  "${V1_FOLD0_RECEIPT_SHA256}" "FPR v1 fold0 receipt"
copy_independent_input "${V2_FOLD1_DECISION}" "${V2_FOLD1_DECISION_SNAPSHOT}" \
  "${V2_FOLD1_DECISION_SHA256}" "FPR v2 fold1 decision"
copy_independent_input "${V2_FOLD1_RECEIPT}" "${V2_FOLD1_RECEIPT_SNAPSHOT}" \
  "${V2_FOLD1_RECEIPT_SHA256}" "FPR v2 fold1 receipt"
copy_independent_input "${DENSITY_FOLD2_DECISION}" \
  "${DENSITY_FOLD2_DECISION_SNAPSHOT}" \
  "${DENSITY_FOLD2_DECISION_SHA256}" "density fold2 decision"
copy_independent_input "${V3_FOLD3_DECISION}" "${V3_FOLD3_DECISION_SNAPSHOT}" \
  "${V3_FOLD3_DECISION_SHA256}" "FPR v3 fold3 decision"
copy_independent_input "${V3_FOLD3_RECEIPT}" "${V3_FOLD3_RECEIPT_SNAPSHOT}" \
  "${V3_FOLD3_RECEIPT_SHA256}" "FPR v3 fold3 receipt"
copy_independent_input "${AV4_MECHANISM_DECISION}" \
  "${AV4_MECHANISM_DECISION_SNAPSHOT}" \
  "${AV4_MECHANISM_DECISION_SHA256}" "A-V4 mechanism decision"
copy_independent_input "${AV4_MECHANISM_RECEIPT}" \
  "${AV4_MECHANISM_RECEIPT_SNAPSHOT}" \
  "${AV4_MECHANISM_RECEIPT_SHA256}" "A-V4 mechanism receipt"
chown "${SNAPSHOT_OWNER_UID}:${SNAPSHOT_OWNER_GID}" "${INPUT_ROOT}"
chmod 0555 "${INPUT_ROOT}"
mkdir "${RUNTIME_HOME}/hf" "${RUNTIME_HOME}/xdg" "${RUNTIME_HOME}/torch"
readonly SNAPSHOT_EXECUTOR="${CODE_SNAPSHOT}/scripts/mcln_density_audit_snapshot_exec.py"

readonly LAUNCH_LOG="${RUNTIME_OUTPUT}/launch.log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1
echo "audit_only=true long_training_authorized=false"
echo "audit_root=${AUDIT_ROOT}"
echo "resume_snapshot=${RESUME_SNAPSHOT}"
echo "audit_epoch=${AUDIT_EPOCH} fold=${FOLD} fit_samples=${EXPECTED_FIT_SAMPLES} holdout_samples=${EXPECTED_HOLDOUT_SAMPLES}"

build_train_args "${RUNTIME_OUTPUT}" "${GROUPFREE_SNAPSHOT}" \
  "${RESUME_SNAPSHOT}"
full_command=(
  /usr/bin/env -i
  HOME="${RUNTIME_HOME}" USER=root LOGNAME=root LANG=C.UTF-8 LC_ALL=C.UTF-8
  PATH="${PATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
  PYTHONPATH="${CODE_SNAPSHOT}:${CODE_SNAPSHOT}/pointnet2"
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
  HF_HOME="${RUNTIME_HOME}/hf" TRANSFORMERS_CACHE="${RUNTIME_HOME}/hf"
  XDG_CACHE_HOME="${RUNTIME_HOME}/xdg" TORCH_HOME="${RUNTIME_HOME}/torch"
  RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 LOCAL_SIZE=1
  MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT}"
  "${PYTHON_BIN}" -I -S "${SNAPSHOT_EXECUTOR}"
  --code-root "${CODE_SNAPSHOT}" --write-root "${RUNTIME_OUTPUT}"
  --allow-write "${RUNTIME_HOME}" --
  "${PYTHON_BIN}" "${CODE_SNAPSHOT}/train_dist_mod.py"
  "${train_args[@]}"
)
readonly COMMAND_FILE="${AUDIT_ROOT}/train_command.txt"
printf '%q ' "${full_command[@]}" >"${COMMAND_FILE}"
printf '\n' >>"${COMMAND_FILE}"
chmod 0444 "${COMMAND_FILE}"

readonly PRE_AUDIT="${AUDIT_ROOT}/pre_audit_provenance.json"
/usr/bin/env -i \
  PATH="${PATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "${PYTHON_BIN}" - "${PRE_AUDIT}" "${CODE_SNAPSHOT}" \
  "${SOURCE_ROOT}" "${LAUNCHER_PATH}" "${COMMAND_FILE}" \
  "${RESUME_SNAPSHOT}" "${GROUPFREE_SNAPSHOT}" \
  "${DATA_MANIFEST_SNAPSHOT}" "${RUNTIME_MANIFEST_SNAPSHOT}" \
  "${V1_FOLD0_DECISION_SNAPSHOT}" "${V1_FOLD0_RECEIPT_SNAPSHOT}" \
  "${V2_FOLD1_DECISION_SNAPSHOT}" "${V2_FOLD1_RECEIPT_SNAPSHOT}" \
  "${DENSITY_FOLD2_DECISION_SNAPSHOT}" \
  "${V3_FOLD3_DECISION_SNAPSHOT}" "${V3_FOLD3_RECEIPT_SNAPSHOT}" \
  "${AV4_MECHANISM_DECISION_SNAPSHOT}" \
  "${AV4_MECHANISM_RECEIPT_SNAPSHOT}" \
  "${TRUSTED_STATIC_EXEC_PATH}" "${TRUSTED_STATIC_SOURCE_PATH}" <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import sys

(
    output_path,
    code_root,
    review_root,
    launcher,
    command,
    resume,
    groupfree,
    data_manifest,
    runtime_manifest,
    fold0_decision,
    fold0_receipt,
    fold1_decision,
    fold1_receipt,
    density_fold2,
    fold3_decision,
    fold3_receipt,
    av4_decision,
    av4_receipt,
    static_executor,
    static_source,
) = sys.argv[1:]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


paths = {
    "launcher": os.path.realpath(launcher),
    "train_entry": os.path.join(code_root, "train_dist_mod.py"),
    "main_utils": os.path.join(code_root, "main_utils.py"),
    "losses": os.path.join(code_root, "models", "losses.py"),
    "parent_relative_text_verifier": os.path.join(
        code_root, "models", "parent_relative_text_verifier.py"
    ),
    "parent_relative_text_verifier_tests": os.path.join(
        review_root, "tests", "test_parent_relative_text_verifier.py"
    ),
    "scene_audit_tests": os.path.join(
        review_root, "tests", "test_fpr_scene_disjoint_audit.py"
    ),
    "finite_training_tests": os.path.join(
        review_root, "tests", "test_main_utils_finite_training.py"
    ),
    "fold4_contract_tests": os.path.join(
        review_root, "tests", "test_fpr_av4_scene_fold4_contract.py"
    ),
    "fold4_spec": os.path.join(
        review_root, "FPR_TV_AV4_SCENE_DISJOINT_FOLD4_SPEC_2026-09-01.md"
    ),
    "fold4_validator": os.path.join(
        code_root,
        "scripts",
        "validate_nr3d_fpr_tv_av4_scene_fold4.py",
    ),
    "model": os.path.join(code_root, "models", "mcln.py"),
    "source_choice_selector": os.path.join(
        code_root, "models", "source_choice_selector.py"
    ),
    "dataset": os.path.join(code_root, "src", "joint_det_dataset.py"),
    "snapshot_executor": os.path.join(
        code_root, "scripts", "mcln_density_audit_snapshot_exec.py"
    ),
    "resume_checkpoint": os.path.realpath(resume),
    "groupfree_checkpoint": os.path.realpath(groupfree),
    "data_manifest": os.path.realpath(data_manifest),
    "runtime_manifest": os.path.realpath(runtime_manifest),
    "fpr_v1_fold0_decision": os.path.realpath(fold0_decision),
    "fpr_v1_fold0_receipt": os.path.realpath(fold0_receipt),
    "fpr_v2_fold1_decision": os.path.realpath(fold1_decision),
    "fpr_v2_fold1_receipt": os.path.realpath(fold1_receipt),
    "density_fold2_decision": os.path.realpath(density_fold2),
    "fpr_v3_fold3_decision": os.path.realpath(fold3_decision),
    "fpr_v3_fold3_receipt": os.path.realpath(fold3_receipt),
    "av4_mechanism_decision": os.path.realpath(av4_decision),
    "av4_mechanism_receipt": os.path.realpath(av4_receipt),
    "static_executor": os.path.realpath(static_executor),
    "static_source": os.path.realpath(static_source),
    "train_command": os.path.realpath(command),
}
payload = {
    "schema": "mcln-fpr-tv-av4-scene-fold4-pre-audit-v1",
    "code_snapshot_root": os.path.realpath(code_root),
    "paths": paths,
    "observed_sha256": {
        name: sha256_file(path) for name, path in paths.items()
    },
}
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
descriptor = os.open(
    output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
)
try:
    offset = 0
    while offset < len(encoded):
        offset += os.write(descriptor, encoded[offset:])
    os.fsync(descriptor)
finally:
    os.close(descriptor)
directory = os.open(
    os.path.dirname(os.path.realpath(output_path)),
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY

cd "${CODE_SNAPSHOT}"
/usr/bin/env -i \
  HOME="${RUNTIME_HOME}" USER=root LOGNAME=root LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PATH="${PATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${CODE_SNAPSHOT}:${CODE_SNAPSHOT}/pointnet2" \
  "${PYTHON_BIN}" -I -S "${SNAPSHOT_EXECUTOR}" \
  --code-root "${CODE_SNAPSHOT}" --write-root "${RUNTIME_OUTPUT}" \
  --allow-write "${RUNTIME_HOME}" \
  --verify-only
"${full_command[@]}"

verify_or_copy_runtime_closure verify-snapshot "${CODE_SNAPSHOT}" - \
  "${RUNTIME_MANIFEST_SNAPSHOT}"
verify_independent_input "${SOURCE_CHECKPOINT}" "${RESUME_SNAPSHOT}" \
  "${SOURCE_CHECKPOINT_SHA256}" "E57"
verify_independent_input "${GROUPFREE_CHECKPOINT}" "${GROUPFREE_SNAPSHOT}" \
  "${GROUPFREE_SHA256}" "GroupFree"
verify_independent_input "${DATA_MANIFEST}" "${DATA_MANIFEST_SNAPSHOT}" \
  "${DATA_MANIFEST_SHA256}" "data manifest"
verify_independent_input "${RUNTIME_MANIFEST}" \
  "${RUNTIME_MANIFEST_SNAPSHOT}" "${RUNTIME_MANIFEST_SHA256}" \
  "runtime manifest"
verify_independent_input "${V1_FOLD0_DECISION}" \
  "${V1_FOLD0_DECISION_SNAPSHOT}" "${V1_FOLD0_DECISION_SHA256}" \
  "FPR v1 fold0 decision"
verify_independent_input "${V1_FOLD0_RECEIPT}" \
  "${V1_FOLD0_RECEIPT_SNAPSHOT}" "${V1_FOLD0_RECEIPT_SHA256}" \
  "FPR v1 fold0 receipt"
verify_independent_input "${V2_FOLD1_DECISION}" \
  "${V2_FOLD1_DECISION_SNAPSHOT}" "${V2_FOLD1_DECISION_SHA256}" \
  "FPR v2 fold1 decision"
verify_independent_input "${V2_FOLD1_RECEIPT}" \
  "${V2_FOLD1_RECEIPT_SNAPSHOT}" "${V2_FOLD1_RECEIPT_SHA256}" \
  "FPR v2 fold1 receipt"
verify_independent_input "${DENSITY_FOLD2_DECISION}" \
  "${DENSITY_FOLD2_DECISION_SNAPSHOT}" "${DENSITY_FOLD2_DECISION_SHA256}" \
  "density fold2 decision"
verify_independent_input "${V3_FOLD3_DECISION}" \
  "${V3_FOLD3_DECISION_SNAPSHOT}" "${V3_FOLD3_DECISION_SHA256}" \
  "FPR v3 fold3 decision"
verify_independent_input "${V3_FOLD3_RECEIPT}" \
  "${V3_FOLD3_RECEIPT_SNAPSHOT}" "${V3_FOLD3_RECEIPT_SHA256}" \
  "FPR v3 fold3 receipt"
verify_independent_input "${AV4_MECHANISM_DECISION}" \
  "${AV4_MECHANISM_DECISION_SNAPSHOT}" "${AV4_MECHANISM_DECISION_SHA256}" \
  "A-V4 mechanism decision"
verify_independent_input "${AV4_MECHANISM_RECEIPT}" \
  "${AV4_MECHANISM_RECEIPT_SNAPSHOT}" "${AV4_MECHANISM_RECEIPT_SHA256}" \
  "A-V4 mechanism receipt"
verify_dataset_manifest "${DATA_MANIFEST_SNAPSHOT}" \
  "${DATA_MANIFEST_SHA256}"
verify_preregistered_history consumed
verify_checkpoint_contract
verify_or_copy_runtime_closure verify-source "${SOURCE_ROOT}"
require_fixed_inputs
if [[ "${LAUNCHER_START_SHA256}" != \
      "${MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256}" ]] \
   || [[ "${LAUNCHER_START_SHA256}" != \
      "$(sha256_file "${LAUNCHER_PATH}")" ]]; then
  echo "launcher changed during fold4 audit" >&2
  exit 8
fi
require_sha256 "${TRUSTED_STATIC_EXEC_PATH}" \
  "${TRUSTED_STATIC_EXEC_SHA256}" "trusted static executor"
require_sha256 "${TRUSTED_STATIC_SOURCE_PATH}" \
  "${TRUSTED_STATIC_SOURCE_SHA256}" "trusted static source"

mapfile -t receipts < <(
  find "${RUNTIME_OUTPUT}" -type f \
    -name "fpr_scene_disjoint_audit_fold_${FOLD}_epoch_${AUDIT_EPOCH}.json" \
    -print
)
mapfile -t metrics_receipts < <(
  find "${RUNTIME_OUTPUT}" -type f \
    -name "eval_metrics_epoch_${AUDIT_EPOCH}.json" -print
)
if (("${#receipts[@]}" != 1 || "${#metrics_receipts[@]}" != 1)); then
  echo "expected one fold receipt and one metrics receipt" >&2
  exit 8
fi
readonly RECEIPT="${receipts[0]}"
readonly METRICS="${metrics_receipts[0]}"
if [[ "$(dirname "${RECEIPT}")" != "$(dirname "${METRICS}")" ]]; then
  echo "fold receipt and metrics receipt are in different run leaves" >&2
  exit 8
fi
readonly RUN_LEAF="$(dirname "${RECEIPT}")"
readonly EXPECTED_RUN_PARENT="${RUNTIME_OUTPUT}/nr3d/${EXP}"
[[ "$(dirname "${RUN_LEAF}")" == "${EXPECTED_RUN_PARENT}"
   && "$(basename "${RUN_LEAF}")" =~ ^[0-9]+$
   && "$(readlink -f "${RUN_LEAF}")" == "${RUN_LEAF}" ]] || {
  echo "unexpected or non-canonical scene-audit run leaf" >&2
  exit 8
}
if find "${AUDIT_ROOT}" \
    \( -path "${INPUT_ROOT}" -o -path "${CODE_SNAPSHOT}" \) -prune \
    -o -type f -name '*.pth' -print -quit | grep -q .; then
  echo "fold4 audit unexpectedly generated a weight" >&2
  exit 8
fi

readonly DECISION="${AUDIT_ROOT}/decision.json"
set +e
/usr/bin/env -i \
  PATH="${PATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${CODE_SNAPSHOT}" \
  "${PYTHON_BIN}" -I -S \
  "${CODE_SNAPSHOT}/scripts/validate_nr3d_fpr_tv_av4_scene_fold4.py" \
  --receipt "${RECEIPT}" --metrics "${METRICS}" \
  --pre-audit "${PRE_AUDIT}" --decision "${DECISION}" \
  --checkpoint-sha256 "${SOURCE_CHECKPOINT_SHA256}" \
  --config-sha256 "${CONFIG_SHA256}" \
  --launcher-sha256 "${LAUNCHER_START_SHA256}" \
  --static-executor-sha256 "${TRUSTED_STATIC_EXEC_SHA256}" \
  --static-source-sha256 "${TRUSTED_STATIC_SOURCE_SHA256}" \
  --runtime-manifest-sha256 "${RUNTIME_MANIFEST_SHA256}" \
  --data-manifest-sha256 "${DATA_MANIFEST_SHA256}" \
  --groupfree-sha256 "${GROUPFREE_SHA256}"
readonly decision_status=$?
set -e
if ((decision_status != 0 && decision_status != 20)); then
  echo "fold4 decision validation failed with status ${decision_status}" >&2
  exit "${decision_status}"
fi
[[ -f "${DECISION}" ]] || {
  echo "fold4 decision was not durably written" >&2
  exit 8
}
chmod 0444 "${RECEIPT}" "${METRICS}" "${PRE_AUDIT}" \
  "${DECISION}" "${LAUNCH_LOG}"
echo "fold4_receipt=${RECEIPT}"
echo "fold4_receipt_sha256=$(sha256_file "${RECEIPT}")"
echo "fold4_metrics=${METRICS}"
echo "fold4_metrics_sha256=$(sha256_file "${METRICS}")"
echo "fold4_decision=${DECISION}"
echo "fold4_decision_sha256=$(sha256_file "${DECISION}")"
echo "long_training_authorized=false"
if ((decision_status == 20)); then
  echo "A-V4 scene-disjoint fold4 gate failed; method is sealed" >&2
  exit 20
fi
echo "fold4_gate_pass=true next_step=independent_review_only"
