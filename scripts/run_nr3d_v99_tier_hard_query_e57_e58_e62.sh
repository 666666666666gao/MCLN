#!/bin/bash
set -euo pipefail

[[ "${MCLN_TIER_TRUSTED_CLEAN_ENV:-}" == "1" ]] || {
  echo "launcher must be entered through the reviewed clean-env bootstrap" >&2
  exit 2
}
[[ "${TRUSTED_BOOTSTRAP_SHA256:-}" =~ ^[0-9a-f]{64}$
    && -f "${TRUSTED_BOOTSTRAP_PATH:-}" ]] || {
  echo "trusted bootstrap provenance is incomplete" >&2
  exit 2
}
actual_bootstrap_sha256="$(/usr/bin/sha256sum "${TRUSTED_BOOTSTRAP_PATH}" | /usr/bin/awk '{print $1}')"
[[ "${actual_bootstrap_sha256}" == "${TRUSTED_BOOTSTRAP_SHA256}" ]] || {
  echo "trusted bootstrap SHA changed" >&2
  exit 2
}
if [[ -n "${BASH_ENV:-}" || -n "${ENV:-}" || -n "${LD_PRELOAD:-}"
      || -n "${LD_AUDIT:-}" || -n "${PYTHONOPTIMIZE:-}"
      || -n "${PYTHONWARNINGS:-}" || -n "${PYTHONSTARTUP:-}"
      || -n "${PYTHONHOME:-}" || -n "${PYTHONUSERBASE:-}" ]]; then
  echo "ambient shell, loader, and Python injection variables must be empty" >&2
  exit 2
fi
if /usr/bin/env | /usr/bin/grep -Eq '^(SHELLOPTS|BASHOPTS|PS4)='; then
  echo "exported Bash option/debug variables are forbidden" >&2
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

readonly SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${SOURCE_ROOT}"
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly SOURCE_CHECKPOINT_ORIGINAL="${DATA_ROOT%/}/gf_detector_l6o256.pth"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT_ORIGINAL}"
readonly SOURCE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly DATASET="nr3d"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v99_baseline_gt/nr3d"
readonly REQUIRED_RESUME_CHECKPOINT_ORIGINAL="${OUTPUT_ROOT}/control/official_rec_monitor/official_best_rec025_epoch_57_0p56500823.pth"
REQUIRED_RESUME_CHECKPOINT="${REQUIRED_RESUME_CHECKPOINT_ORIGINAL}"
readonly REQUIRED_RESUME_SHA256="fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
readonly REQUIRED_RESUME_EPOCH=57
readonly AUDIT_ROOT="${OUTPUT_ROOT}/audit/nr3d_mcln_joint_butdcls_v99_tier_hard_query_audit_e58_b100_b16x1_once"
readonly AUDIT_DECISION_ORIGINAL="${AUDIT_ROOT}/audit_decision.json"
AUDIT_DECISION="${AUDIT_DECISION_ORIGINAL}"
readonly AUDIT_DECISION_SHA256="9231000032bb58e31a4930d77db31d336cab3b73d2d36cf79010300817aa7b75"
readonly AUDIT_RECEIPT_ORIGINAL="${AUDIT_ROOT}/nr3d/nr3d_mcln_joint_butdcls_v99_tier_hard_query_audit_e58_b100_b16x1/1788066299/train_audit_receipt_epoch_58.json"
AUDIT_RECEIPT="${AUDIT_RECEIPT_ORIGINAL}"
readonly AUDIT_RECEIPT_SHA256="f93fbb19798d3035da5eedc299419e913569f8ef4f123efdb8ff3ca6d74168f6"
readonly AUDIT_PROVENANCE_ORIGINAL="${AUDIT_ROOT}/audit_provenance.json"
AUDIT_PROVENANCE="${AUDIT_PROVENANCE_ORIGINAL}"
readonly AUDIT_PROVENANCE_SHA256="4f579e97015b002b1b445d6daf3a38a8e69e8eefc5c956358e1091ceb92cc46e"
readonly CONTROL_ROOT="${OUTPUT_ROOT}/control/tier_hard_query_e57_e58_e62_patience2"
readonly FIRST_RECOVERY_ROOT="${CONTROL_ROOT}/startup_recovery_v1"
readonly RECOVERY_ROOT="${CONTROL_ROOT}/startup_recovery_v2"
readonly APPROVAL_ORIGINAL="${CONTROL_ROOT}/independent_density_approval.json"
APPROVAL="${APPROVAL_ORIGINAL}"
readonly APPROVAL_SHA256="0a79b4e3a6d7f4a75836c1e3d1f78939bc045c929ad37f08b19c82f710a05f69"
readonly ORIGINAL_FORMAL_CLAIM="${CONTROL_ROOT}/formal_claim.json"
readonly ORIGINAL_FORMAL_CLAIM_SHA256="6d394b6d23d1a161e7b8465f515e8cc37e67db6a0595dfa66b60c103917cfd6a"
readonly ORIGINAL_GUARD_STATE="${CONTROL_ROOT}/guard_state.json"
readonly ORIGINAL_GUARD_STATE_SHA256="fbadb2ea52c4239c81e0f9a1217ee4c0f66327b66c597c74b334cdeeb68eb35e"
readonly ORIGINAL_GUARD_LOG="${CONTROL_ROOT}/guard.log"
readonly ORIGINAL_GUARD_LOG_SHA256="f196efe18f71fff7bdbaa8140b2cc4b33118365c4a2ac619852cef49fab8d831"
readonly ORIGINAL_WATCHDOG_HEARTBEAT="${CONTROL_ROOT}/watchdog_heartbeat.json"
readonly ORIGINAL_WATCHDOG_HEARTBEAT_SHA256="e47ec018a58420830f2c4662014f0af044bb83d08b1136e395dc7edb4ce1654c"
readonly ORIGINAL_LAUNCH_LOG="${OUTPUT_ROOT}/launch/nr3d_mcln_joint_butdcls_v99_tier_hard_query_e57_e58_e62_b16a1_backbone_20260830_150936.log"
readonly ORIGINAL_LAUNCH_LOG_SHA256="5e5d80c4192b12db1089e160f878a733f7f416758cc97c776169e933bdc4f93b"
readonly ORIGINAL_CODE_MANIFEST="${CONTROL_ROOT}/formal_code_snapshot_v1/CODE_MANIFEST.json"
readonly ORIGINAL_CODE_MANIFEST_SHA256="2df64a2dba60c0630afa67fc64d26f9fc548aa9564101c8b8de6569de5868357"
readonly ORIGINAL_INPUT_MANIFEST="${CONTROL_ROOT}/formal_input_snapshot_v1/INPUT_MANIFEST.json"
readonly ORIGINAL_INPUT_MANIFEST_SHA256="202c302b2e1b34d2652bdd92b82f465ff6c15dd6b0e7d4280bd9109c18994007"
readonly ORIGINAL_FAILED_RUN_ROOT="${OUTPUT_ROOT}/backbone/nr3d_mcln_joint_butdcls_v99_tier_hard_query_e57_e58_e62_b16a1_20260830_150936"
readonly FIRST_RECOVERY_CLAIM="${FIRST_RECOVERY_ROOT}/formal_recovery_claim.json"
readonly FIRST_RECOVERY_CLAIM_SHA256="7e1655386ed0cc616d393e32682f41cb3d4bc46a0c0995d128a8a3956a4e65aa"
readonly FIRST_RECOVERY_GUARD_STATE="${FIRST_RECOVERY_ROOT}/guard_state.json"
readonly FIRST_RECOVERY_GUARD_STATE_SHA256="5c1717a7543b226f37d9547093ca54d228b83fe693452ea846ec6edccc9ef640"
readonly FIRST_RECOVERY_GUARD_READY="${FIRST_RECOVERY_ROOT}/guard_ready.json"
readonly FIRST_RECOVERY_GUARD_READY_SHA256="d27d8cdd26130cd7a3db6ed1456072d38f22b588f8bebcc0939f9405eef76bbf"
readonly FIRST_RECOVERY_HEARTBEAT="${FIRST_RECOVERY_ROOT}/watchdog_heartbeat.json"
readonly FIRST_RECOVERY_HEARTBEAT_SHA256="4f016697e93f8267ccf92c24e4786d490f828837bbc57b2045d5a4901759564e"
readonly FIRST_RECOVERY_GUARD_LOG="${FIRST_RECOVERY_ROOT}/guard.log"
readonly FIRST_RECOVERY_GUARD_LOG_SHA256="24f736f0ffc08afa5dee701d88241eba0838bc3bc24519f6669f76a6ced360d6"
readonly FIRST_RECOVERY_WATCHDOG_LOG="${FIRST_RECOVERY_ROOT}/watchdog.log"
readonly FIRST_RECOVERY_WATCHDOG_LOG_SHA256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
readonly FIRST_RECOVERY_SCREEN_LOG="${CONTROL_ROOT}/formal_recovery_screen.log"
readonly FIRST_RECOVERY_SCREEN_LOG_SHA256="10dcfe57dab968e03c033e791ac9483372bd32c818f4a8a023a9e2ef51df7b03"
readonly FIRST_RECOVERY_LAUNCH_LOG="${OUTPUT_ROOT}/launch/nr3d_mcln_joint_butdcls_v99_tier_hard_query_e57_e58_e62_b16a1_recovery1_backbone_20260830_154357.log"
readonly FIRST_RECOVERY_LAUNCH_LOG_SHA256="cd3961ef01c7dbc95eb0c40d760bb10f51cff806ae2035cc553f12683d02c1d8"
readonly FIRST_RECOVERY_CODE_ROOT="${FIRST_RECOVERY_ROOT}/formal_code_snapshot_v1"
readonly FIRST_RECOVERY_CODE_MANIFEST="${FIRST_RECOVERY_CODE_ROOT}/CODE_MANIFEST.json"
readonly FIRST_RECOVERY_CODE_MANIFEST_SHA256="0c50ae997a80a72450001c1cea2a8bb0939a4b14adc53ba150ae0f1124c77613"
readonly FIRST_RECOVERY_INPUT_ROOT="${FIRST_RECOVERY_ROOT}/formal_input_snapshot_v1"
readonly FIRST_RECOVERY_INPUT_MANIFEST="${FIRST_RECOVERY_INPUT_ROOT}/INPUT_MANIFEST.json"
readonly FIRST_RECOVERY_INPUT_MANIFEST_SHA256="db26157b609d5aeb0a4fe479831cda48e917b0508c33ad7fcc3ca6a55b045eba"
readonly FIRST_RECOVERY_FAILED_RUN_ROOT="${OUTPUT_ROOT}/backbone/nr3d_mcln_joint_butdcls_v99_tier_hard_query_e57_e58_e62_b16a1_recovery1_20260830_154357"
readonly FORMAL_CLAIM="${RECOVERY_ROOT}/formal_recovery_claim.json"
readonly GUARD_SCRIPT_ORIGINAL="/root/mcln_nr3d_tier_hard_query_e58_e62_guard.py"
GUARD_SCRIPT="${GUARD_SCRIPT_ORIGINAL}"
readonly GUARD_SCRIPT_SHA256="8653c116461efc0b2ddf08a4616a9121c5eb7c3b108a14d6e5315cde20a4f806"
readonly WATCHDOG_SCRIPT_ORIGINAL="/root/mcln_nr3d_tier_hard_query_watchdog.py"
WATCHDOG_SCRIPT="${WATCHDOG_SCRIPT_ORIGINAL}"
readonly WATCHDOG_SCRIPT_SHA256="d85e1c0fed6985457386a177852a8414dcf52679b25ae94dcdeb26e152f9a47f"
readonly LANDLOCK_SCRIPT_ORIGINAL="/root/mcln_landlock_snapshot_exec.py"
LANDLOCK_SCRIPT="${LANDLOCK_SCRIPT_ORIGINAL}"
readonly LANDLOCK_SCRIPT_SHA256="ae953c5985549f7c8e47818764237c1db30dd12783367498445879d18a82a28c"
readonly CODE_SNAPSHOT="${RECOVERY_ROOT}/formal_code_snapshot_v2"
readonly INPUT_SNAPSHOT="${RECOVERY_ROOT}/formal_input_snapshot_v2"
readonly SNAPSHOT_OWNER_UID=65532
readonly SNAPSHOT_OWNER_GID=65532
readonly CODE_MANIFEST="${CODE_SNAPSHOT}/CODE_MANIFEST.json"
readonly INPUT_MANIFEST="${INPUT_SNAPSHOT}/INPUT_MANIFEST.json"
readonly GUARD_READY="${RECOVERY_ROOT}/guard_ready.json"
readonly WATCHDOG_HEARTBEAT="${RECOVERY_ROOT}/watchdog_heartbeat.json"
readonly GUARD_LOG="${RECOVERY_ROOT}/guard.log"
readonly WATCHDOG_LOG="${RECOVERY_ROOT}/watchdog.log"
readonly REQUIRED_TRAIN_ENTRY_SHA256="8f78cca50174423d0c4ab0b3c76a1fa6f22bbd1b179bd547013243ad199996f1"
readonly REQUIRED_MAIN_UTILS_SHA256="f677ada134f36bfd194a0694002fa3df37c1d2106d56cbebe59b608a1abbf065"
readonly REQUIRED_LOSSES_SHA256="48de298038ca9996e0c135dfc42ad5d271a0827d1a8c03309cc71d51a4e3082f"
readonly REQUIRED_AUXILIARY_SHA256="67f602ce84c5c5adce98553d65fe58b15f6dbd1dae0a52e430ec2beb29257c2b"
readonly REQUIRED_MODEL_SHA256="3b9a2b88e9fa36c6f94b1595ae3812818ca5fdeb4b1db64c68f467912c04c9b1"
readonly REQUIRED_SELECTOR_SHA256="61211cea91de4c3f3c11e44bc5ff035711307f3381d5a4069d4ab7842bee17dc"
readonly REQUIRED_DETECTOR_FILTER_SHA256="49a43b89a1ff129d09dcbdf0f6b61ff817aca50fb2c0edcb49072c60ded1a7e7"
readonly REQUIRED_SOURCE_MOE_SHA256="f09b2c5a5fb609a1b474baede83e21af0f034ec0fa9b050ced6613f66162fbd3"
readonly REQUIRED_AFFINITY_SHA256="39ecf930684e8936bce3472ea19cad2d59aab37dbb4b0b5b84d2fe842d12039c"
readonly REQUIRED_TRAINING_GROUPS_SHA256="0298531a3adefd2f010cccc65a0724cf9f0521374446cfe7a9081dfacdd437ce"
readonly REQUIRED_GROUNDING_EVALUATOR_SHA256="0173b31a7a818f872c210b01a4e5d17601c4e5f10ec8d97f78c7e537fa44e062"
readonly REQUIRED_MODELS_INIT_SHA256="1ce161d73a37e19ac0a341cd0e4dc50e12449444aa796ac3c58283fedae5a4ee"
readonly REQUIRED_LR_SCHEDULER_SHA256="aa88d4bc7eea87a205bb9c94f8b7a1e54418e9f7335ec232c6c5d50778e245c7"
readonly REQUIRED_UTILS_INIT_SHA256="d0f1d31d6e0207a37dcb8a33315a409226d651670a8d0a238223009a4118f57b"
readonly REQUIRED_DATASET_SHA256="800bac2caf9b7a319bdc200f60386000e4e374a559d9581113a8eb57d525f9f0"
readonly REQUIRED_TEST_SHA256="8e4a92124a8007b11b05a76500ad97d8f5493d56f580418fc8bba00ee72bd0ed"
readonly REQUIRED_PIPELINE_SHA256="edcbfa2bc341b2a57375d948b07c43b6db3eaa70fc9deb3bc44e402ee4c03648"
readonly EXP="nr3d_mcln_joint_butdcls_v99_tier_hard_query_e57_e58_e62_b16a1_recovery2"
readonly TRAIN_SCREEN_NAME="mcln_nr3d_tier_hard_query_recovery2"
readonly BATCH_SIZE=16
readonly MAX_EPOCH=62
readonly EXPECTED_EVAL_SAMPLE_COUNT=7899
readonly MASTER_PORT=5331
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
  --warmup-epoch -1
)
BACKBONE_EXTRA_ARGS=(
  --print_freq 20
  --joint_det
  --butd_cls
  --resume_lr_scale 1.0
  --tier_hard_query_aux_loss_weight 0.10
  --tier_hard_query_aux_candidate_top_k 128
  --tier_hard_query_aux_max_negatives 8
  --tier_hard_query_aux_target_tolerance 0.15
  --tier_hard_query_aux_target_confidence_floor 0.01
  --tier_hard_query_aux_pair_margin 0.05
  --tier_hard_query_aux_preserve_weight 0.25
  --tier_hard_query_aux_acc025_pair_weight 2.0
)

if (($# != 0)); then
  echo "usage: REVIEWED_LAUNCHER_SHA256=<sha256> MODE=preflight|backbone $0" >&2
  exit 2
fi
export MODE="${MODE:-preflight}"
case "${MODE}" in
  preflight|backbone) ;;
  *) echo "MODE must be preflight or backbone" >&2; exit 2 ;;
esac
[[ "${REVIEWED_LAUNCHER_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "REVIEWED_LAUNCHER_SHA256 must be the reviewed 64-character digest" >&2
  exit 2
}
readonly LAUNCHER_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly ACTUAL_LAUNCHER_SHA256="$(sha256sum "${LAUNCHER_PATH}" | awk '{print $1}')"
[[ "${ACTUAL_LAUNCHER_SHA256}" == "${REVIEWED_LAUNCHER_SHA256}" ]] || {
  echo "launcher SHA is not the reviewed fixed point: ${ACTUAL_LAUNCHER_SHA256}" >&2
  exit 3
}

for variable_name in \
    BACKBONE_RESUME_CHECKPOINT BACKBONE_RESUME_SHA256 BACKBONE_RESUME_EPOCH; do
  if [[ -n "${!variable_name:-}" ]]; then
    case "${variable_name}" in
      BACKBONE_RESUME_CHECKPOINT) required_value="${REQUIRED_RESUME_CHECKPOINT}" ;;
      BACKBONE_RESUME_SHA256) required_value="${REQUIRED_RESUME_SHA256}" ;;
      BACKBONE_RESUME_EPOCH) required_value="${REQUIRED_RESUME_EPOCH}" ;;
    esac
    [[ "${!variable_name}" == "${required_value}" ]] || {
      echo "${variable_name} conflicts with the pinned E57 resume" >&2
      exit 2
    }
  fi
done
export BACKBONE_RESUME_CHECKPOINT="${REQUIRED_RESUME_CHECKPOINT}"
export BACKBONE_RESUME_SHA256="${REQUIRED_RESUME_SHA256}"
export BACKBONE_RESUME_EPOCH="${REQUIRED_RESUME_EPOCH}"
export VALIDATE_BACKBONE_RESUME=0
export CLEAN_RECONSTRUCTIBLE_CACHES=1
export PRUNE_NONBEST_BACKBONE_WEIGHTS=1

cd "${ROOT_DIR}"
unset PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/pointnet2"
require_sha256() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; exit 3; }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA changed: ${actual}" >&2
    exit 3
  }
}
require_sha256 "${SOURCE_ROOT}/train_dist_mod.py" "${REQUIRED_TRAIN_ENTRY_SHA256}" "training entrypoint"
require_sha256 "${SOURCE_ROOT}/main_utils.py" "${REQUIRED_MAIN_UTILS_SHA256}" "main_utils"
require_sha256 "${SOURCE_ROOT}/models/losses.py" "${REQUIRED_LOSSES_SHA256}" "loss implementation"
require_sha256 "${SOURCE_ROOT}/models/tier_hard_query_auxiliary.py" "${REQUIRED_AUXILIARY_SHA256}" "tier hard-query auxiliary"
require_sha256 "${SOURCE_ROOT}/models/mcln.py" "${REQUIRED_MODEL_SHA256}" "MCLN model"
require_sha256 "${SOURCE_ROOT}/models/source_choice_selector.py" "${REQUIRED_SELECTOR_SHA256}" "V99 selector"
require_sha256 "${SOURCE_ROOT}/models/rec_evaluator_filter.py" "${REQUIRED_DETECTOR_FILTER_SHA256}" "detector filter"
require_sha256 "${SOURCE_ROOT}/models/source_moe.py" "${REQUIRED_SOURCE_MOE_SHA256}" "box-IoU helper"
require_sha256 "${SOURCE_ROOT}/models/sacr_relation_counterfactual.py" "${REQUIRED_AFFINITY_SHA256}" "affinity helper"
require_sha256 "${SOURCE_ROOT}/models/mcln_training_groups.py" "${REQUIRED_TRAINING_GROUPS_SHA256}" "optimizer grouping implementation"
require_sha256 "${SOURCE_ROOT}/src/grounding_evaluator.py" "${REQUIRED_GROUNDING_EVALUATOR_SHA256}" "formal grounding evaluator"
require_sha256 "${SOURCE_ROOT}/models/__init__.py" "${REQUIRED_MODELS_INIT_SHA256}" "model export surface"
require_sha256 "${SOURCE_ROOT}/utils/lr_scheduler.py" "${REQUIRED_LR_SCHEDULER_SHA256}" "LR scheduler implementation"
require_sha256 "${SOURCE_ROOT}/utils/__init__.py" "${REQUIRED_UTILS_INIT_SHA256}" "scheduler export surface"
require_sha256 "${SOURCE_ROOT}/src/joint_det_dataset.py" "${REQUIRED_DATASET_SHA256}" "dataset implementation"
require_sha256 "${SOURCE_ROOT}/tests/test_tier_hard_query_auxiliary.py" "${REQUIRED_TEST_SHA256}" "regression test"
require_sha256 "${SOURCE_ROOT}/scripts/run_dataset_v99_pipeline_tier_formal.sh" "${REQUIRED_PIPELINE_SHA256}" "formal pipeline"
require_sha256 "${GUARD_SCRIPT_ORIGINAL}" "${GUARD_SCRIPT_SHA256}" "formal patience guard"
require_sha256 "${WATCHDOG_SCRIPT_ORIGINAL}" "${WATCHDOG_SCRIPT_SHA256}" "formal guard watchdog"
require_sha256 "${LANDLOCK_SCRIPT_ORIGINAL}" "${LANDLOCK_SCRIPT_SHA256}" "Landlock snapshot executor"
require_sha256 "${SOURCE_CHECKPOINT_ORIGINAL}" "${SOURCE_SHA256}" "GroupFree checkpoint"
require_sha256 "${REQUIRED_RESUME_CHECKPOINT_ORIGINAL}" "${REQUIRED_RESUME_SHA256}" "protected E57 checkpoint"
require_sha256 "${AUDIT_DECISION_ORIGINAL}" "${AUDIT_DECISION_SHA256}" "audit decision"
require_sha256 "${AUDIT_RECEIPT_ORIGINAL}" "${AUDIT_RECEIPT_SHA256}" "audit receipt"
require_sha256 "${AUDIT_PROVENANCE_ORIGINAL}" "${AUDIT_PROVENANCE_SHA256}" "audit provenance"
require_sha256 "${APPROVAL_ORIGINAL}" "${APPROVAL_SHA256}" "independent approval"

PROOF_CHECKPOINT="${REQUIRED_RESUME_CHECKPOINT}" \
PROOF_CHECKPOINT_SHA256="${REQUIRED_RESUME_SHA256}" \
PROOF_APPROVAL="${APPROVAL}" \
PROOF_DECISION="${AUDIT_DECISION}" \
PROOF_RECEIPT="${AUDIT_RECEIPT}" \
PROOF_PROVENANCE="${AUDIT_PROVENANCE}" \
"${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import math
import os

import torch


def raw_json(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return raw, json.loads(raw.decode("utf-8"))


def sha256_open_file(handle):
    digest = hashlib.sha256()
    handle.seek(0)
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


approval_raw, approval = raw_json(os.environ["PROOF_APPROVAL"])
decision_raw, decision = raw_json(os.environ["PROOF_DECISION"])
receipt_raw, _receipt = raw_json(os.environ["PROOF_RECEIPT"])
provenance_raw, _provenance = raw_json(os.environ["PROOF_PROVENANCE"])
if approval.get("schema") != "mcln-tier-hard-query-independent-density-approval-v1":
    raise SystemExit("independent approval schema mismatch")
authorization = approval.get("authorization", {})
expected_authorization = {
    "authorized": True,
    "dataset": "nr3d",
    "resume_epoch": 57,
    "resume_sha256": "fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655",
    "first_epoch": 58,
    "last_epoch": 62,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "baseline_hits025": 4463,
    "target_hits025": 4724,
    "patience": 2,
    "checkpoint_retention_metrics": ["rec_acc025"],
    "tier_hard_query_aux": {
        "loss_weight": 0.1,
        "candidate_top_k": 128,
        "max_negatives": 8,
        "target_tolerance": 0.15,
        "target_confidence_floor": 0.01,
        "pair_margin": 0.05,
        "preserve_weight": 0.25,
        "acc025_pair_weight": 2.0,
    },
}
if authorization != expected_authorization:
    raise SystemExit("independent approval scope changed")
for label, raw, expected in (
        ("decision", decision_raw, approval["decision"]["sha256"]),
        ("receipt", receipt_raw, approval["receipt"]["sha256"]),
        ("provenance", provenance_raw, approval["provenance"]["sha256"])):
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit("{} no longer matches independent approval".format(label))
if (
        decision.get("schema") != "mcln-tier-hard-query-audit-decision-v1"
        or decision.get("density_gate_passed") is not True
        or decision.get("integrity_passed") is not True
        or decision.get("bounded_receipt_validated") is not True
        or decision.get("no_checkpoint_written") is not True
        or decision.get("post_provenance_verified") is not True
        or decision.get("long_training_authorized") is not False):
    raise SystemExit("bounded audit decision is not eligible for independent approval")
if not all(decision.get("density_checks", {}).values()):
    raise SystemExit("not every preregistered density check passed")
if approval.get("reviewed_density") != {
        "repair_row_ratio": decision["density_values"]["repair_row_ratio"],
        "supervised_row_ratio": decision["density_values"]["supervised_row_ratio"],
        "selected_negative_count_mean": decision["density_values"]["selected_negative_count_mean"],
        "pair_violation_ratio": decision["density_values"]["pair_violation_ratio"],
        "selected_score_gradient_l1": decision["density_values"]["selected_score_gradient_l1"],
        "parent_acc025": decision["stat_means"]["tier_hard_query_aux_parent_acc025"],
        "teacher_oracle_acc025": decision["stat_means"]["tier_hard_query_aux_teacher_oracle_acc025"],
}:
    raise SystemExit("reviewed density values changed")

checkpoint_path = os.environ["PROOF_CHECKPOINT"]
with open(checkpoint_path, "rb") as handle:
    before_sha256 = sha256_open_file(handle)
    if before_sha256 != os.environ["PROOF_CHECKPOINT_SHA256"]:
        raise SystemExit("protected E57 SHA changed")
    handle.seek(0)
    checkpoint = torch.load(handle, map_location="cpu")
    after_sha256 = sha256_open_file(handle)
if after_sha256 != before_sha256:
    raise SystemExit("protected E57 changed while loading")
config = checkpoint.get("config")
config = vars(config) if hasattr(config, "__dict__") else dict(config or {})
optimizer = checkpoint.get("optimizer", {})
scheduler = checkpoint.get("scheduler", {})
groups = optimizer.get("param_groups", [])
expected_current_lrs = [1e-5, 1e-4, 1e-5, 1.25e-5]
expected_initial_lrs = [1e-4, 1e-3, 1e-4, 1.25e-4]
required_config = {
    "dataset": ["nr3d"],
    "test_dataset": "nr3d",
    "batch_size": 16,
    "joint_det": True,
    "butd_cls": True,
    "butd_gt": False,
    "augment_det": False,
    "use_source_choice_selector": True,
    "eval_use_selector_choice_scores": True,
    "source_choice_selector_sources": "default,default_rank_blend_contrastive010",
    "source_choice_selector_default_source": "default",
    "source_choice_selector_hidden_dim": 288,
    "source_choice_selector_lr": 1.25e-4,
    "source_choice_selector_loss_weight": 0.5,
    "source_choice_selector_choice_target": "precision_gain_default_sourcewise_focal_bce",
    "source_choice_selector_min_iou_gap": 0.03,
    "lr_scheduler": "step",
    "warmup_epoch": -1,
}
if int(checkpoint.get("epoch", -1)) != 57:
    raise SystemExit("protected resume is not E57")
for name, expected in required_config.items():
    if config.get(name) != expected:
        raise SystemExit("protected E57 config mismatch {}".format(name))
if int(config.get("gradient_accumulation_steps", 1)) != 1:
    raise SystemExit("protected E57 accumulation mismatch")
if bool(config.get("drop_incomplete_accumulation_group", False)):
    raise SystemExit("protected E57 drop-tail contract changed")
if any(bool(config.get(name, False)) for name in (
        "use_source_moe", "use_sacr_source", "use_sacr_score_refiner",
        "use_joint_query_quality_reranker")):
    raise SystemExit("protected E57 is not the pure V99 branch")
lineage = config.get("resume_lr_scale_lineage", config.get("resume_lr_scale", 1.0))
if not math.isclose(float(lineage), 1.0, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("protected E57 LR lineage mismatch")
if len(groups) != 4 or len(optimizer.get("state", {})) != 716:
    raise SystemExit("protected E57 optimizer topology mismatch")
if [float(group["lr"]) for group in groups] != expected_current_lrs:
    raise SystemExit("protected E57 current LR mismatch")
if [float(group["initial_lr"]) for group in groups] != expected_initial_lrs:
    raise SystemExit("protected E57 initial LR mismatch")
if scheduler.get("base_lrs") != expected_initial_lrs:
    raise SystemExit("protected E57 scheduler base LR mismatch")
if scheduler.get("_last_lr") != expected_current_lrs:
    raise SystemExit("protected E57 scheduler current LR mismatch")
if int(scheduler.get("last_epoch", -1)) != 159942:
    raise SystemExit("protected E57 scheduler progress mismatch")
if dict(scheduler.get("milestones", {})) != {423706: 1}:
    raise SystemExit("protected E57 scheduler milestone mismatch")
print("formal_proof=approved_tier_aux_E57_to_E58_E62_patience2")
PY

if find "${AUDIT_ROOT}" -type f -name '*.pth' -print -quit | grep -q .; then
  echo "bounded audit unexpectedly contains a checkpoint" >&2
  exit 3
fi

verify_failed_startups() {
  ORIGINAL_CLAIM="${ORIGINAL_FORMAL_CLAIM}" \
  ORIGINAL_CLAIM_SHA="${ORIGINAL_FORMAL_CLAIM_SHA256}" \
  ORIGINAL_STATE="${ORIGINAL_GUARD_STATE}" \
  ORIGINAL_STATE_SHA="${ORIGINAL_GUARD_STATE_SHA256}" \
  ORIGINAL_GUARD_LOG_ENV="${ORIGINAL_GUARD_LOG}" \
  ORIGINAL_GUARD_LOG_SHA="${ORIGINAL_GUARD_LOG_SHA256}" \
  ORIGINAL_HEARTBEAT="${ORIGINAL_WATCHDOG_HEARTBEAT}" \
  ORIGINAL_HEARTBEAT_SHA="${ORIGINAL_WATCHDOG_HEARTBEAT_SHA256}" \
  ORIGINAL_LAUNCH_LOG_ENV="${ORIGINAL_LAUNCH_LOG}" \
  ORIGINAL_LAUNCH_LOG_SHA="${ORIGINAL_LAUNCH_LOG_SHA256}" \
  ORIGINAL_CODE_MANIFEST_ENV="${ORIGINAL_CODE_MANIFEST}" \
  ORIGINAL_CODE_MANIFEST_SHA="${ORIGINAL_CODE_MANIFEST_SHA256}" \
  ORIGINAL_INPUT_MANIFEST_ENV="${ORIGINAL_INPUT_MANIFEST}" \
  ORIGINAL_INPUT_MANIFEST_SHA="${ORIGINAL_INPUT_MANIFEST_SHA256}" \
  ORIGINAL_RUN_ROOT="${ORIGINAL_FAILED_RUN_ROOT}" \
  FIRST_RECOVERY_CLAIM_ENV="${FIRST_RECOVERY_CLAIM}" \
  FIRST_RECOVERY_CLAIM_SHA="${FIRST_RECOVERY_CLAIM_SHA256}" \
  FIRST_RECOVERY_STATE_ENV="${FIRST_RECOVERY_GUARD_STATE}" \
  FIRST_RECOVERY_STATE_SHA="${FIRST_RECOVERY_GUARD_STATE_SHA256}" \
  FIRST_RECOVERY_READY_ENV="${FIRST_RECOVERY_GUARD_READY}" \
  FIRST_RECOVERY_READY_SHA="${FIRST_RECOVERY_GUARD_READY_SHA256}" \
  FIRST_RECOVERY_HEARTBEAT_ENV="${FIRST_RECOVERY_HEARTBEAT}" \
  FIRST_RECOVERY_HEARTBEAT_SHA="${FIRST_RECOVERY_HEARTBEAT_SHA256}" \
  FIRST_RECOVERY_GUARD_LOG_ENV="${FIRST_RECOVERY_GUARD_LOG}" \
  FIRST_RECOVERY_GUARD_LOG_SHA="${FIRST_RECOVERY_GUARD_LOG_SHA256}" \
  FIRST_RECOVERY_WATCHDOG_LOG_ENV="${FIRST_RECOVERY_WATCHDOG_LOG}" \
  FIRST_RECOVERY_WATCHDOG_LOG_SHA="${FIRST_RECOVERY_WATCHDOG_LOG_SHA256}" \
  FIRST_RECOVERY_SCREEN_LOG_ENV="${FIRST_RECOVERY_SCREEN_LOG}" \
  FIRST_RECOVERY_SCREEN_LOG_SHA="${FIRST_RECOVERY_SCREEN_LOG_SHA256}" \
  FIRST_RECOVERY_LAUNCH_LOG_ENV="${FIRST_RECOVERY_LAUNCH_LOG}" \
  FIRST_RECOVERY_LAUNCH_LOG_SHA="${FIRST_RECOVERY_LAUNCH_LOG_SHA256}" \
  FIRST_RECOVERY_CODE_MANIFEST_ENV="${FIRST_RECOVERY_CODE_MANIFEST}" \
  FIRST_RECOVERY_CODE_MANIFEST_SHA="${FIRST_RECOVERY_CODE_MANIFEST_SHA256}" \
  FIRST_RECOVERY_INPUT_MANIFEST_ENV="${FIRST_RECOVERY_INPUT_MANIFEST}" \
  FIRST_RECOVERY_INPUT_MANIFEST_SHA="${FIRST_RECOVERY_INPUT_MANIFEST_SHA256}" \
  FIRST_RECOVERY_RUN_ROOT="${FIRST_RECOVERY_FAILED_RUN_ROOT}" \
  RECOVERY_ROOT_PROOF="${RECOVERY_ROOT}" \
  "${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
import pathlib


def read_exact(path_name, sha_name, parse_json=False):
    path = pathlib.Path(os.environ[path_name])
    with path.open("rb") as handle:
        raw = handle.read()
    actual = hashlib.sha256(raw).hexdigest()
    expected = os.environ[sha_name]
    if actual != expected:
        raise SystemExit("startup-failure evidence SHA changed: {}".format(path))
    return json.loads(raw.decode("utf-8")) if parse_json else raw


claim = read_exact("ORIGINAL_CLAIM", "ORIGINAL_CLAIM_SHA", True)
state = read_exact("ORIGINAL_STATE", "ORIGINAL_STATE_SHA", True)
guard_log = read_exact("ORIGINAL_GUARD_LOG_ENV", "ORIGINAL_GUARD_LOG_SHA")
heartbeat = read_exact("ORIGINAL_HEARTBEAT", "ORIGINAL_HEARTBEAT_SHA", True)
launch_log = read_exact("ORIGINAL_LAUNCH_LOG_ENV", "ORIGINAL_LAUNCH_LOG_SHA")
code_manifest = read_exact(
    "ORIGINAL_CODE_MANIFEST_ENV", "ORIGINAL_CODE_MANIFEST_SHA", True)
input_manifest = read_exact(
    "ORIGINAL_INPUT_MANIFEST_ENV", "ORIGINAL_INPUT_MANIFEST_SHA", True)
first_claim = read_exact(
    "FIRST_RECOVERY_CLAIM_ENV", "FIRST_RECOVERY_CLAIM_SHA", True)
first_state = read_exact(
    "FIRST_RECOVERY_STATE_ENV", "FIRST_RECOVERY_STATE_SHA", True)
first_ready = read_exact(
    "FIRST_RECOVERY_READY_ENV", "FIRST_RECOVERY_READY_SHA", True)
first_heartbeat = read_exact(
    "FIRST_RECOVERY_HEARTBEAT_ENV", "FIRST_RECOVERY_HEARTBEAT_SHA", True)
first_guard_log = read_exact(
    "FIRST_RECOVERY_GUARD_LOG_ENV", "FIRST_RECOVERY_GUARD_LOG_SHA")
first_watchdog_log = read_exact(
    "FIRST_RECOVERY_WATCHDOG_LOG_ENV", "FIRST_RECOVERY_WATCHDOG_LOG_SHA")
first_screen_log = read_exact(
    "FIRST_RECOVERY_SCREEN_LOG_ENV", "FIRST_RECOVERY_SCREEN_LOG_SHA")
first_launch_log = read_exact(
    "FIRST_RECOVERY_LAUNCH_LOG_ENV", "FIRST_RECOVERY_LAUNCH_LOG_SHA")
first_code_manifest = read_exact(
    "FIRST_RECOVERY_CODE_MANIFEST_ENV",
    "FIRST_RECOVERY_CODE_MANIFEST_SHA", True)
first_input_manifest = read_exact(
    "FIRST_RECOVERY_INPUT_MANIFEST_ENV",
    "FIRST_RECOVERY_INPUT_MANIFEST_SHA", True)

expected_claim = {
    "schema": "mcln-tier-hard-query-formal-claim-v2",
    "launcher_sha256":
        "59f137cdab7de10da5654b36542689627298dde6eabbe07f214e4168b7ffadb1",
    "bootstrap_sha256":
        "db9ec0e43a45202f1f3d7d3c7d2e3f7d4d61ce4fcc22db7f71bab07e51c329d6",
    "experiment":
        "nr3d_mcln_joint_butdcls_v99_tier_hard_query_e57_e58_e62_b16a1",
    "run_root_parent": os.environ["ORIGINAL_RUN_ROOT"],
    "screen": "590933.mcln_nr3d_tier_hard_query",
    "screen_start_ticks": 4503124505,
    "epochs": [58, 59, 60, 61, 62],
    "baseline_hits025": 4463,
    "target_hits025": 4724,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "checkpoint_retention_metrics": ["rec_acc025"],
    "code_manifest_sha256": os.environ["ORIGINAL_CODE_MANIFEST_SHA"],
    "input_manifest_sha256": os.environ["ORIGINAL_INPUT_MANIFEST_SHA"],
}
for name, value in expected_claim.items():
    if claim.get(name) != value:
        raise SystemExit("original failed claim mismatch: {}".format(name))
if claim.get("guard") != {"pid": 591260, "start_ticks": 4503127514}:
    raise SystemExit("original guard identity mismatch")
if claim.get("watchdog") != {"pid": 591265, "start_ticks": 4503127548}:
    raise SystemExit("original watchdog identity mismatch")
if state != {
        "error": "RuntimeError('watchdog heartbeat is missing')",
        "schema": "mcln-nr3d-tier-hard-query-e58-e62-patience-guard-v2",
        "status": "failed_closed",
        "updated_at": "2026-08-30T15:09:53+0800"}:
    raise SystemExit("original fail-closed state changed")
if b"RuntimeError: watchdog heartbeat is missing" not in guard_log:
    raise SystemExit("original guard failure log is inconsistent")
if b"guard_pid=591260 watchdog_pid=591265" not in launch_log:
    raise SystemExit("original launch log did not reach the guarded pre-train boundary")
if heartbeat.get("guard_pid") != 591260 or heartbeat.get("watchdog_pid") != 591265:
    raise SystemExit("original heartbeat identity mismatch")
if code_manifest.get("schema") != "mcln-tier-hard-query-code-snapshot-v3":
    raise SystemExit("original code manifest schema mismatch")
if input_manifest.get("schema") != "mcln-tier-hard-query-input-snapshot-v3":
    raise SystemExit("original input manifest schema mismatch")
old_guard = input_manifest.get("files", {}).get(
    "mcln_nr3d_tier_hard_query_e58_e62_guard.py", {})
if old_guard.get("sha256") != (
        "ee90d10e99711b7d063b037d5a3b66b43d80fa4be9f086e234265604b0585673"):
    raise SystemExit("original buggy guard was not the claimed input")

run_root = pathlib.Path(os.environ["ORIGINAL_RUN_ROOT"]).resolve()
if not run_root.is_dir():
    raise SystemExit("original failed run root disappeared")
entries = []
for path in run_root.rglob("*"):
    entries.append((str(path.relative_to(run_root)), path.is_dir(), path.is_symlink()))
if entries != [("runtime_home", True, False)]:
    raise SystemExit("original run advanced beyond the pre-training boundary: {!r}".format(
        entries))
if (pathlib.Path(os.environ["ORIGINAL_STATE"]).parent / "decision.json").exists():
    raise SystemExit("original attempt unexpectedly has a formal decision")
if pathlib.Path(os.environ["RECOVERY_ROOT_PROOF"]).exists():
    raise SystemExit("the separately reviewed recovery was already attempted")


def exact_identity_alive(pid, ticks):
    try:
        raw = (pathlib.Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8")
        close_paren = raw.rfind(")")
        return int(raw[close_paren + 2:].split()[19]) == ticks
    except (IOError, OSError, ValueError):
        return False


for pid, ticks in ((590933, 4503124505), (591260, 4503127514),
                   (591265, 4503127548)):
    if exact_identity_alive(pid, ticks):
        raise SystemExit("original failed process identity is still alive: {}".format(pid))

expected_first_claim = {
    "schema": "mcln-tier-hard-query-formal-recovery-claim-v1",
    "created_at": "2026-08-30T15:44:06+0800",
    "launcher":
        "/home/gb/new butd/butd_detr-main/MCLN-main/scripts/"
        "run_nr3d_v99_tier_hard_query_e57_e58_e62.sh",
    "launcher_sha256":
        "c127e32b9f9dbc903744ca50260f2109a0f18d360747e21fd543b72681cb95b3",
    "bootstrap":
        "/home/gb/new butd/butd_detr-main/MCLN-main/scripts/"
        "run_nr3d_v99_tier_hard_query_clean_env.sh",
    "bootstrap_sha256":
        "0366ce64aa4153f9b1abc67da60aaf953797f3e8b23f420c74093fdf0a0e895a",
    "code_manifest_sha256": os.environ["FIRST_RECOVERY_CODE_MANIFEST_SHA"],
    "input_manifest_sha256": os.environ["FIRST_RECOVERY_INPUT_MANIFEST_SHA"],
    "approval_sha256":
        "0a79b4e3a6d7f4a75836c1e3d1f78939bc045c929ad37f08b19c82f710a05f69",
    "landlock_executor_sha256":
        "ae953c5985549f7c8e47818764237c1db30dd12783367498445879d18a82a28c",
    "resume_sha256":
        "fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655",
    "experiment":
        "nr3d_mcln_joint_butdcls_v99_tier_hard_query_"
        "e57_e58_e62_b16a1_recovery1",
    "run_root_parent": os.environ["FIRST_RECOVERY_RUN_ROOT"],
    "screen": "593839.mcln_nr3d_tier_hard_query_recovery1",
    "screen_start_ticks": 4503330545,
    "guard": {"pid": 594174, "start_ticks": 4503333631},
    "watchdog": {"pid": 594178, "start_ticks": 4503333665},
    "epochs": [58, 59, 60, 61, 62],
    "patience": 2,
    "baseline_hits025": 4463,
    "target_hits025": 4724,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "checkpoint_retention_metrics": ["rec_acc025"],
    "recovery_of_claim_sha256": os.environ["ORIGINAL_CLAIM_SHA"],
    "startup_failure_evidence": {
        "guard_state_sha256": os.environ["ORIGINAL_STATE_SHA"],
        "guard_log_sha256": os.environ["ORIGINAL_GUARD_LOG_SHA"],
        "watchdog_heartbeat_sha256": os.environ["ORIGINAL_HEARTBEAT_SHA"],
        "launch_log_sha256": os.environ["ORIGINAL_LAUNCH_LOG_SHA"],
        "training_started": False,
    },
}
if first_claim != expected_first_claim:
    raise SystemExit("first recovery claim changed")
if first_state != {
        "error": "FileNotFoundError(2, 'No such file or directory')",
        "schema": "mcln-nr3d-tier-hard-query-e58-e62-patience-guard-v2",
        "status": "failed_closed",
        "updated_at": "2026-08-30T15:44:16+0800"}:
    raise SystemExit("first recovery fail-closed state changed")
expected_first_ready = {
    "created_at": "2026-08-30T15:44:01+0800",
    "experiment": first_claim["experiment"],
    "pid": 594174,
    "process_start_ticks": 4503333631,
    "run_root_parent": os.environ["FIRST_RECOVERY_RUN_ROOT"],
    "schema": "mcln-tier-hard-query-guard-ready-v2",
    "screen": first_claim["screen"],
    "screen_start_ticks": 4503330545,
    "watchdog_heartbeat": os.environ["FIRST_RECOVERY_HEARTBEAT_ENV"],
}
if first_ready != expected_first_ready:
    raise SystemExit("first recovery guard-ready evidence changed")
for name, value in {
        "schema": "mcln-tier-hard-query-watchdog-v1",
        "experiment": first_claim["experiment"],
        "guard_pid": 594174,
        "guard_start_ticks": 4503333631,
        "watchdog_pid": 594178,
        "watchdog_start_ticks": 4503333665,
        "run_root_parent": os.environ["FIRST_RECOVERY_RUN_ROOT"],
        "screen": first_claim["screen"],
        "screen_start_ticks": 4503330545,
    }.items():
    if first_heartbeat.get(name) != value:
        raise SystemExit("first recovery heartbeat changed: {}".format(name))
if first_heartbeat.get("updated_at_epoch") != 1788075852.2101715:
    raise SystemExit("first recovery heartbeat timestamp changed")
if first_watchdog_log != b"":
    raise SystemExit("first recovery watchdog log is no longer empty")
if b"/proc/594178/stat" not in first_guard_log:
    raise SystemExit("first recovery guard log lost its fail-closed cause")
for label, payload in (("screen", first_screen_log), ("launch", first_launch_log)):
    if b"snapshot_verification=pass" not in payload:
        raise SystemExit("first recovery {} log did not pass snapshot verification".format(
            label))
    if b"Error 304: OS call failed or operation not supported" not in payload:
        raise SystemExit("first recovery {} log lost the CUDA startup failure".format(
            label))
    if b"formal_claim=" not in payload:
        raise SystemExit("first recovery {} log did not reach the claim boundary".format(
            label))
if first_code_manifest.get("schema") != "mcln-tier-hard-query-code-snapshot-v3":
    raise SystemExit("first recovery code manifest schema changed")
if first_input_manifest.get("schema") != "mcln-tier-hard-query-input-snapshot-v3":
    raise SystemExit("first recovery input manifest schema changed")
if first_input_manifest.get("code_manifest_sha256") != os.environ[
        "FIRST_RECOVERY_CODE_MANIFEST_SHA"]:
    raise SystemExit("first recovery snapshot linkage changed")
if first_code_manifest.get("files", {}).get(
        "scripts/run_dataset_v99_pipeline_tier_formal.sh", {}).get(
            "sha256") != (
                "c4781275cfd194c3fd1e916bfb1975913cca0cd4b5525238db8b74572d5cea1e"):
    raise SystemExit("first recovery did not consume the CUDA-incompatible pipeline")
if first_input_manifest.get("files", {}).get(
        "mcln_landlock_snapshot_exec.py", {}).get("sha256") != (
            "ae953c5985549f7c8e47818764237c1db30dd12783367498445879d18a82a28c"):
    raise SystemExit("first recovery Landlock executor identity changed")

first_run_root = pathlib.Path(os.environ["FIRST_RECOVERY_RUN_ROOT"]).resolve()
if not first_run_root.is_dir():
    raise SystemExit("first recovery failed run root disappeared")
first_entries = sorted(
    (str(path.relative_to(first_run_root)), path.is_dir(), path.is_symlink())
    for path in first_run_root.rglob("*")
)
if first_entries != [("runtime_home", True, False)]:
    raise SystemExit("first recovery advanced beyond CUDA initialization: {!r}".format(
        first_entries))
if any((first_run_root / "runtime_home").iterdir()):
    raise SystemExit("first recovery runtime_home is no longer empty")
if (pathlib.Path(os.environ["FIRST_RECOVERY_STATE_ENV"]).parent /
        "decision.json").exists():
    raise SystemExit("first recovery unexpectedly has a formal decision")
for pid, ticks in ((593839, 4503330545), (594174, 4503333631),
                   (594178, 4503333665)):
    if exact_identity_alive(pid, ticks):
        raise SystemExit("first recovery process identity is still alive: {}".format(pid))
print(
    "startup_failure_evidence=verified_two_pre_optimizer_failures;"
    "first=atomic_heartbeat_reader;second=landlock_proc_task_cuda304")
PY
}

verify_failed_startups

prepare_formal_snapshots() {
  SNAP_SOURCE_ROOT="${SOURCE_ROOT}" \
  SNAP_CODE_ROOT="${CODE_SNAPSHOT}" \
  SNAP_INPUT_ROOT="${INPUT_SNAPSHOT}" \
  SNAP_E57="${REQUIRED_RESUME_CHECKPOINT_ORIGINAL}" \
  SNAP_GF="${SOURCE_CHECKPOINT_ORIGINAL}" \
  SNAP_GUARD="${GUARD_SCRIPT_ORIGINAL}" \
  SNAP_WATCHDOG="${WATCHDOG_SCRIPT_ORIGINAL}" \
  SNAP_LANDLOCK="${LANDLOCK_SCRIPT_ORIGINAL}" \
  SNAP_APPROVAL="${APPROVAL_ORIGINAL}" \
  SNAP_DECISION="${AUDIT_DECISION_ORIGINAL}" \
  SNAP_RECEIPT="${AUDIT_RECEIPT_ORIGINAL}" \
  SNAP_PROVENANCE="${AUDIT_PROVENANCE_ORIGINAL}" \
  SNAP_ORIGINAL_CLAIM="${ORIGINAL_FORMAL_CLAIM}" \
  SNAP_ORIGINAL_STATE="${ORIGINAL_GUARD_STATE}" \
  SNAP_ORIGINAL_GUARD_LOG="${ORIGINAL_GUARD_LOG}" \
  SNAP_ORIGINAL_HEARTBEAT="${ORIGINAL_WATCHDOG_HEARTBEAT}" \
  SNAP_ORIGINAL_LAUNCH_LOG="${ORIGINAL_LAUNCH_LOG}" \
  SNAP_FIRST_RECOVERY_CLAIM="${FIRST_RECOVERY_CLAIM}" \
  SNAP_FIRST_RECOVERY_STATE="${FIRST_RECOVERY_GUARD_STATE}" \
  SNAP_FIRST_RECOVERY_READY="${FIRST_RECOVERY_GUARD_READY}" \
  SNAP_FIRST_RECOVERY_HEARTBEAT="${FIRST_RECOVERY_HEARTBEAT}" \
  SNAP_FIRST_RECOVERY_GUARD_LOG="${FIRST_RECOVERY_GUARD_LOG}" \
  SNAP_FIRST_RECOVERY_WATCHDOG_LOG="${FIRST_RECOVERY_WATCHDOG_LOG}" \
  SNAP_FIRST_RECOVERY_SCREEN_LOG="${FIRST_RECOVERY_SCREEN_LOG}" \
  SNAP_FIRST_RECOVERY_LAUNCH_LOG="${FIRST_RECOVERY_LAUNCH_LOG}" \
  SNAP_FIRST_RECOVERY_CODE_MANIFEST="${FIRST_RECOVERY_CODE_MANIFEST}" \
  SNAP_FIRST_RECOVERY_INPUT_MANIFEST="${FIRST_RECOVERY_INPUT_MANIFEST}" \
  SNAP_OWNER_UID_ENV="${SNAPSHOT_OWNER_UID}" \
  SNAP_OWNER_GID_ENV="${SNAPSHOT_OWNER_GID}" \
  SHA_E57="${REQUIRED_RESUME_SHA256}" \
  SHA_GF="${SOURCE_SHA256}" \
  SHA_GUARD="${GUARD_SCRIPT_SHA256}" \
  SHA_WATCHDOG="${WATCHDOG_SCRIPT_SHA256}" \
  SHA_LANDLOCK="${LANDLOCK_SCRIPT_SHA256}" \
  SHA_APPROVAL="${APPROVAL_SHA256}" \
  SHA_DECISION="${AUDIT_DECISION_SHA256}" \
  SHA_RECEIPT="${AUDIT_RECEIPT_SHA256}" \
  SHA_PROVENANCE="${AUDIT_PROVENANCE_SHA256}" \
  SHA_ORIGINAL_CLAIM="${ORIGINAL_FORMAL_CLAIM_SHA256}" \
  SHA_ORIGINAL_STATE="${ORIGINAL_GUARD_STATE_SHA256}" \
  SHA_ORIGINAL_GUARD_LOG="${ORIGINAL_GUARD_LOG_SHA256}" \
  SHA_ORIGINAL_HEARTBEAT="${ORIGINAL_WATCHDOG_HEARTBEAT_SHA256}" \
  SHA_ORIGINAL_LAUNCH_LOG="${ORIGINAL_LAUNCH_LOG_SHA256}" \
  SHA_FIRST_RECOVERY_CLAIM="${FIRST_RECOVERY_CLAIM_SHA256}" \
  SHA_FIRST_RECOVERY_STATE="${FIRST_RECOVERY_GUARD_STATE_SHA256}" \
  SHA_FIRST_RECOVERY_READY="${FIRST_RECOVERY_GUARD_READY_SHA256}" \
  SHA_FIRST_RECOVERY_HEARTBEAT="${FIRST_RECOVERY_HEARTBEAT_SHA256}" \
  SHA_FIRST_RECOVERY_GUARD_LOG="${FIRST_RECOVERY_GUARD_LOG_SHA256}" \
  SHA_FIRST_RECOVERY_WATCHDOG_LOG="${FIRST_RECOVERY_WATCHDOG_LOG_SHA256}" \
  SHA_FIRST_RECOVERY_SCREEN_LOG="${FIRST_RECOVERY_SCREEN_LOG_SHA256}" \
  SHA_FIRST_RECOVERY_LAUNCH_LOG="${FIRST_RECOVERY_LAUNCH_LOG_SHA256}" \
  SHA_FIRST_RECOVERY_CODE_MANIFEST="${FIRST_RECOVERY_CODE_MANIFEST_SHA256}" \
  SHA_FIRST_RECOVERY_INPUT_MANIFEST="${FIRST_RECOVERY_INPUT_MANIFEST_SHA256}" \
  SHA_TRAIN="${REQUIRED_TRAIN_ENTRY_SHA256}" \
  SHA_MAIN="${REQUIRED_MAIN_UTILS_SHA256}" \
  SHA_LOSSES="${REQUIRED_LOSSES_SHA256}" \
  SHA_AUX="${REQUIRED_AUXILIARY_SHA256}" \
  SHA_MODEL="${REQUIRED_MODEL_SHA256}" \
  SHA_SELECTOR="${REQUIRED_SELECTOR_SHA256}" \
  SHA_FILTER="${REQUIRED_DETECTOR_FILTER_SHA256}" \
  SHA_MOE="${REQUIRED_SOURCE_MOE_SHA256}" \
  SHA_AFFINITY="${REQUIRED_AFFINITY_SHA256}" \
  SHA_TRAINING_GROUPS="${REQUIRED_TRAINING_GROUPS_SHA256}" \
  SHA_GROUNDING_EVALUATOR="${REQUIRED_GROUNDING_EVALUATOR_SHA256}" \
  SHA_MODELS_INIT="${REQUIRED_MODELS_INIT_SHA256}" \
  SHA_LR_SCHEDULER="${REQUIRED_LR_SCHEDULER_SHA256}" \
  SHA_UTILS_INIT="${REQUIRED_UTILS_INIT_SHA256}" \
  SHA_DATASET="${REQUIRED_DATASET_SHA256}" \
  SHA_TEST="${REQUIRED_TEST_SHA256}" \
  SHA_PIPELINE="${REQUIRED_PIPELINE_SHA256}" \
  "${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
import pathlib
import shutil
import stat


source_root = pathlib.Path(os.environ["SNAP_SOURCE_ROOT"]).resolve()
code_root = pathlib.Path(os.environ["SNAP_CODE_ROOT"])
input_root = pathlib.Path(os.environ["SNAP_INPUT_ROOT"])
snapshot_owner_uid = int(os.environ["SNAP_OWNER_UID_ENV"])
snapshot_owner_gid = int(os.environ["SNAP_OWNER_GID_ENV"])


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def copy_verified(source, destination, expected):
    source = pathlib.Path(source)
    if source.is_symlink():
        raise SystemExit("snapshot source must not be a symlink: {}".format(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as reader:
        before = os.fstat(reader.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("snapshot source is not regular: {}".format(source))
        with destination.open("xb") as writer:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        after = os.fstat(reader.fileno())
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit("snapshot source SHA changed: {}".format(source))
    if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise SystemExit("snapshot source changed while copying: {}".format(source))
    return {
        "sha256": actual,
        "size": after.st_size,
        "source_dev": before.st_dev,
        "source_ino": before.st_ino,
    }


critical = {
    "train_dist_mod.py": os.environ["SHA_TRAIN"],
    "main_utils.py": os.environ["SHA_MAIN"],
    "models/losses.py": os.environ["SHA_LOSSES"],
    "models/tier_hard_query_auxiliary.py": os.environ["SHA_AUX"],
    "models/mcln.py": os.environ["SHA_MODEL"],
    "models/source_choice_selector.py": os.environ["SHA_SELECTOR"],
    "models/rec_evaluator_filter.py": os.environ["SHA_FILTER"],
    "models/source_moe.py": os.environ["SHA_MOE"],
    "models/sacr_relation_counterfactual.py": os.environ["SHA_AFFINITY"],
    "models/mcln_training_groups.py": os.environ["SHA_TRAINING_GROUPS"],
    "src/grounding_evaluator.py": os.environ["SHA_GROUNDING_EVALUATOR"],
    "models/__init__.py": os.environ["SHA_MODELS_INIT"],
    "utils/lr_scheduler.py": os.environ["SHA_LR_SCHEDULER"],
    "utils/__init__.py": os.environ["SHA_UTILS_INIT"],
    "src/joint_det_dataset.py": os.environ["SHA_DATASET"],
    "tests/test_tier_hard_query_auxiliary.py": os.environ["SHA_TEST"],
    "scripts/run_dataset_v99_pipeline_tier_formal.sh": os.environ["SHA_PIPELINE"],
}
code_directories = (
    "models", "src", "utils", "pointnet2", "sng_parser", "data",
    "scripts", "tests",
)
code_files = (
    "train_dist_mod.py",
    "main_utils.py",
    "mapping_full2rio27.json",
    "experiment_output/historical_e71_geometry/"
    "v97_contextual_listwise_hierarchical_trainonly_v1.json",
)


def source_code_files():
    result = set(code_files)
    for dirname in code_directories:
        root = source_root / dirname
        if not root.is_dir():
            raise SystemExit("missing source directory: {}".format(root))
        for current, dirs, files in os.walk(str(root)):
            dirs[:] = sorted(
                name for name in dirs
                if name not in ("__pycache__", ".pytest_cache")
            )
            current_path = pathlib.Path(current)
            for name in sorted(files):
                if name.endswith((".pyc", ".pyo")):
                    continue
                path = current_path / name
                relative = path.relative_to(source_root).as_posix()
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    raise SystemExit("non-regular code input: {}".format(path))
                result.add(relative)
    return sorted(result)


def verify_code_snapshot():
    manifest_path = code_root / "CODE_MANIFEST.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("schema") != "mcln-tier-hard-query-code-snapshot-v3":
        raise SystemExit("code snapshot schema mismatch")
    records = manifest.get("files", {})
    directories = manifest.get("directories", {})
    if directories.get(".") != {
            "mode": "0555",
            "uid": snapshot_owner_uid,
            "gid": snapshot_owner_gid}:
        raise SystemExit("code snapshot root mode contract changed")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SystemExit("code snapshot manifest is not regular")
    if stat.S_IMODE(manifest_path.stat().st_mode) != 0o444:
        raise SystemExit("code snapshot manifest mode changed")
    if (
            manifest_path.stat().st_uid != snapshot_owner_uid
            or manifest_path.stat().st_gid != snapshot_owner_gid):
        raise SystemExit("code snapshot manifest owner changed")
    observed = set()
    observed_directories = set()
    for current, dirs, files in os.walk(str(code_root)):
        current_path = pathlib.Path(current)
        relative_dir = current_path.relative_to(code_root).as_posix()
        relative_dir = "." if relative_dir == "." else relative_dir
        if current_path.is_symlink() or not current_path.is_dir():
            raise SystemExit("invalid code snapshot directory: {}".format(
                relative_dir
            ))
        contract = directories.get(relative_dir, {})
        expected_mode = int(contract.get("mode", "-1"), 8)
        expected_uid = int(contract.get("uid", -1))
        expected_gid = int(contract.get("gid", -1))
        current_stat = current_path.stat()
        if stat.S_IMODE(current_stat.st_mode) != expected_mode:
            raise SystemExit("code snapshot directory mode changed: {}".format(
                relative_dir
            ))
        if current_stat.st_uid != expected_uid or current_stat.st_gid != expected_gid:
            raise SystemExit("code snapshot directory owner changed: {}".format(
                relative_dir
            ))
        observed_directories.add(relative_dir)
        if relative_dir == "tensorboard_output":
            if contract != {"mode": "0775", "uid": 0, "gid": 0}:
                raise SystemExit("tensorboard directory contract changed")
            if dirs or files:
                raise SystemExit("writable tensorboard snapshot directory is not empty")
            dirs[:] = []
            continue
        if contract != {
                "mode": "0555",
                "uid": snapshot_owner_uid,
                "gid": snapshot_owner_gid}:
            raise SystemExit("read-only code directory contract changed")
        for name in list(dirs):
            child = current_path / name
            if child.is_symlink():
                raise SystemExit("symlink directory in code snapshot: {}".format(
                    child.relative_to(code_root).as_posix()
                ))
        for name in files:
            relative = (current_path / name).relative_to(code_root).as_posix()
            if relative == "CODE_MANIFEST.json":
                continue
            path = code_root / relative
            if not path.is_file() or path.is_symlink():
                raise SystemExit("invalid code snapshot entry: {}".format(relative))
            observed.add(relative)
    if observed != set(records):
        raise SystemExit("code snapshot inventory changed")
    if observed_directories != set(directories):
        raise SystemExit("code snapshot directory inventory changed")
    for relative, record in records.items():
        path = code_root / relative
        if path.stat().st_size != int(record["size"]):
            raise SystemExit("code snapshot size changed: {}".format(relative))
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise SystemExit("code snapshot mode changed: {}".format(relative))
        if (
                path.stat().st_uid != snapshot_owner_uid
                or path.stat().st_gid != snapshot_owner_gid
                or int(record.get("uid", -1)) != snapshot_owner_uid
                or int(record.get("gid", -1)) != snapshot_owner_gid):
            raise SystemExit("code snapshot owner changed: {}".format(relative))
        if digest_file(path) != record["sha256"]:
            raise SystemExit("code snapshot SHA changed: {}".format(relative))
    for relative, expected in critical.items():
        if records.get(relative, {}).get("sha256") != expected:
            raise SystemExit("critical snapshot SHA mismatch: {}".format(relative))
    return hashlib.sha256(raw).hexdigest()


def build_code_snapshot():
    if code_root.exists():
        return verify_code_snapshot()
    temporary = code_root.with_name(
        code_root.name + ".tmp.{}".format(os.getpid())
    )
    if temporary.exists():
        shutil.rmtree(str(temporary))
    temporary.mkdir(parents=True)
    try:
        records = {}
        for relative in source_code_files():
            source = source_root / relative
            destination = temporary / relative
            expected = critical.get(relative, digest_file(source))
            record = copy_verified(source, destination, expected)
            record["mode"] = "0444"
            record["uid"] = snapshot_owner_uid
            record["gid"] = snapshot_owner_gid
            records[relative] = record
            os.chown(str(destination), snapshot_owner_uid, snapshot_owner_gid)
            os.chmod(str(destination), 0o444)
        runtime = temporary / "tensorboard_output"
        runtime.mkdir()
        os.chmod(str(runtime), 0o775)
        directories = {}
        for current, _dirs, _files in os.walk(str(temporary)):
            path = pathlib.Path(current)
            relative = path.relative_to(temporary).as_posix()
            relative = "." if relative == "." else relative
            if relative == "tensorboard_output":
                directories[relative] = {"mode": "0775", "uid": 0, "gid": 0}
            else:
                directories[relative] = {
                    "mode": "0555",
                    "uid": snapshot_owner_uid,
                    "gid": snapshot_owner_gid,
                }
        manifest_path = temporary / "CODE_MANIFEST.json"
        payload = {
            "schema": "mcln-tier-hard-query-code-snapshot-v3",
            "source_root": str(source_root),
            "files": records,
            "directories": directories,
            "writable_runtime_directories": ["tensorboard_output"],
        }
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chown(str(manifest_path), snapshot_owner_uid, snapshot_owner_gid)
        os.chmod(str(manifest_path), 0o444)
        for current, dirs, _files in os.walk(str(temporary), topdown=False):
            path = pathlib.Path(current)
            if path == runtime:
                os.chown(str(path), 0, 0)
                continue
            os.chown(str(path), snapshot_owner_uid, snapshot_owner_gid)
            os.chmod(str(path), 0o555)
        fsync_directory(temporary)
        try:
            os.rename(str(temporary), str(code_root))
        except OSError:
            if not code_root.is_dir():
                raise
            shutil.rmtree(str(temporary))
        fsync_directory(code_root.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(str(temporary))
        raise
    return verify_code_snapshot()


input_sources = {
    "protected_e57.pth": (
        os.environ["SNAP_E57"], os.environ["SHA_E57"], 0o444),
    "gf_detector_l6o256.pth": (
        os.environ["SNAP_GF"], os.environ["SHA_GF"], 0o444),
    "mcln_nr3d_tier_hard_query_e58_e62_guard.py": (
        os.environ["SNAP_GUARD"], os.environ["SHA_GUARD"], 0o555),
    "mcln_nr3d_tier_hard_query_watchdog.py": (
        os.environ["SNAP_WATCHDOG"], os.environ["SHA_WATCHDOG"], 0o555),
    "mcln_landlock_snapshot_exec.py": (
        os.environ["SNAP_LANDLOCK"], os.environ["SHA_LANDLOCK"], 0o555),
    "independent_density_approval.json": (
        os.environ["SNAP_APPROVAL"], os.environ["SHA_APPROVAL"], 0o444),
    "audit_decision.json": (
        os.environ["SNAP_DECISION"], os.environ["SHA_DECISION"], 0o444),
    "train_audit_receipt_epoch_58.json": (
        os.environ["SNAP_RECEIPT"], os.environ["SHA_RECEIPT"], 0o444),
    "audit_provenance.json": (
        os.environ["SNAP_PROVENANCE"], os.environ["SHA_PROVENANCE"], 0o444),
    "failed_formal_claim.json": (
        os.environ["SNAP_ORIGINAL_CLAIM"],
        os.environ["SHA_ORIGINAL_CLAIM"], 0o444),
    "failed_guard_state.json": (
        os.environ["SNAP_ORIGINAL_STATE"],
        os.environ["SHA_ORIGINAL_STATE"], 0o444),
    "failed_guard.log": (
        os.environ["SNAP_ORIGINAL_GUARD_LOG"],
        os.environ["SHA_ORIGINAL_GUARD_LOG"], 0o444),
    "failed_watchdog_heartbeat.json": (
        os.environ["SNAP_ORIGINAL_HEARTBEAT"],
        os.environ["SHA_ORIGINAL_HEARTBEAT"], 0o444),
    "failed_launch.log": (
        os.environ["SNAP_ORIGINAL_LAUNCH_LOG"],
        os.environ["SHA_ORIGINAL_LAUNCH_LOG"], 0o444),
    "failed_recovery1_claim.json": (
        os.environ["SNAP_FIRST_RECOVERY_CLAIM"],
        os.environ["SHA_FIRST_RECOVERY_CLAIM"], 0o444),
    "failed_recovery1_guard_state.json": (
        os.environ["SNAP_FIRST_RECOVERY_STATE"],
        os.environ["SHA_FIRST_RECOVERY_STATE"], 0o444),
    "failed_recovery1_guard_ready.json": (
        os.environ["SNAP_FIRST_RECOVERY_READY"],
        os.environ["SHA_FIRST_RECOVERY_READY"], 0o444),
    "failed_recovery1_watchdog_heartbeat.json": (
        os.environ["SNAP_FIRST_RECOVERY_HEARTBEAT"],
        os.environ["SHA_FIRST_RECOVERY_HEARTBEAT"], 0o444),
    "failed_recovery1_guard.log": (
        os.environ["SNAP_FIRST_RECOVERY_GUARD_LOG"],
        os.environ["SHA_FIRST_RECOVERY_GUARD_LOG"], 0o444),
    "failed_recovery1_watchdog.log": (
        os.environ["SNAP_FIRST_RECOVERY_WATCHDOG_LOG"],
        os.environ["SHA_FIRST_RECOVERY_WATCHDOG_LOG"], 0o444),
    "failed_recovery1_screen.log": (
        os.environ["SNAP_FIRST_RECOVERY_SCREEN_LOG"],
        os.environ["SHA_FIRST_RECOVERY_SCREEN_LOG"], 0o444),
    "failed_recovery1_launch.log": (
        os.environ["SNAP_FIRST_RECOVERY_LAUNCH_LOG"],
        os.environ["SHA_FIRST_RECOVERY_LAUNCH_LOG"], 0o444),
    "failed_recovery1_code_manifest.json": (
        os.environ["SNAP_FIRST_RECOVERY_CODE_MANIFEST"],
        os.environ["SHA_FIRST_RECOVERY_CODE_MANIFEST"], 0o444),
    "failed_recovery1_input_manifest.json": (
        os.environ["SNAP_FIRST_RECOVERY_INPUT_MANIFEST"],
        os.environ["SHA_FIRST_RECOVERY_INPUT_MANIFEST"], 0o444),
}


def verify_input_snapshot(expected_code_manifest_sha256):
    manifest_path = input_root / "INPUT_MANIFEST.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("schema") != "mcln-tier-hard-query-input-snapshot-v3":
        raise SystemExit("input snapshot schema mismatch")
    if input_root.is_symlink() or not input_root.is_dir():
        raise SystemExit("input snapshot root is not a real directory")
    if stat.S_IMODE(input_root.stat().st_mode) != 0o555:
        raise SystemExit("input snapshot root mode changed")
    if (
            input_root.stat().st_uid != snapshot_owner_uid
            or input_root.stat().st_gid != snapshot_owner_gid):
        raise SystemExit("input snapshot root owner changed")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SystemExit("input snapshot manifest is not regular")
    if stat.S_IMODE(manifest_path.stat().st_mode) != 0o444:
        raise SystemExit("input snapshot manifest mode changed")
    if (
            manifest_path.stat().st_uid != snapshot_owner_uid
            or manifest_path.stat().st_gid != snapshot_owner_gid):
        raise SystemExit("input snapshot manifest owner changed")
    if manifest.get("code_manifest_sha256") != expected_code_manifest_sha256:
        raise SystemExit("input snapshot is bound to a different code snapshot")
    records = manifest.get("files", {})
    if set(records) != set(input_sources):
        raise SystemExit("input snapshot inventory changed")
    observed = {
        path.name for path in input_root.iterdir()
        if path.name != "INPUT_MANIFEST.json"
    }
    if observed != set(input_sources):
        raise SystemExit("input snapshot has unregistered files")
    for relative, (source, expected, mode) in input_sources.items():
        path = input_root / relative
        if not path.is_file() or path.is_symlink():
            raise SystemExit("invalid input snapshot entry: {}".format(relative))
        if digest_file(path) != expected:
            raise SystemExit("input snapshot SHA changed: {}".format(relative))
        if path.stat().st_size != int(records[relative]["size"]):
            raise SystemExit("input snapshot size changed: {}".format(relative))
        if stat.S_IMODE(path.stat().st_mode) != mode:
            raise SystemExit("input snapshot mode changed: {}".format(relative))
        if (
                path.stat().st_uid != snapshot_owner_uid
                or path.stat().st_gid != snapshot_owner_gid
                or int(records[relative].get("uid", -1)) != snapshot_owner_uid
                or int(records[relative].get("gid", -1)) != snapshot_owner_gid):
            raise SystemExit("input snapshot owner changed: {}".format(relative))
        source_stat = pathlib.Path(source).stat()
        output_stat = path.stat()
        if (
                output_stat.st_dev == source_stat.st_dev
                and output_stat.st_ino == source_stat.st_ino):
            raise SystemExit("input snapshot inode aliases its source")
    return hashlib.sha256(raw).hexdigest()


def build_input_snapshot(code_manifest_sha256):
    if input_root.exists():
        return verify_input_snapshot(code_manifest_sha256)
    temporary = input_root.with_name(
        input_root.name + ".tmp.{}".format(os.getpid())
    )
    if temporary.exists():
        shutil.rmtree(str(temporary))
    temporary.mkdir(parents=True)
    try:
        records = {}
        for relative, (source, expected, mode) in input_sources.items():
            destination = temporary / relative
            record = copy_verified(source, destination, expected)
            output_stat = destination.stat()
            source_stat = pathlib.Path(source).stat()
            if (
                    output_stat.st_dev == source_stat.st_dev
                    and output_stat.st_ino == source_stat.st_ino):
                raise SystemExit("input snapshot did not receive an independent inode")
            os.chmod(str(destination), mode)
            record["mode"] = "{:04o}".format(mode)
            record["uid"] = snapshot_owner_uid
            record["gid"] = snapshot_owner_gid
            records[relative] = record
            os.chown(str(destination), snapshot_owner_uid, snapshot_owner_gid)
        manifest_path = temporary / "INPUT_MANIFEST.json"
        payload = {
            "schema": "mcln-tier-hard-query-input-snapshot-v3",
            "code_manifest_sha256": code_manifest_sha256,
            "files": records,
        }
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chown(str(manifest_path), snapshot_owner_uid, snapshot_owner_gid)
        os.chmod(str(manifest_path), 0o444)
        fsync_directory(temporary)
        os.chown(str(temporary), snapshot_owner_uid, snapshot_owner_gid)
        os.chmod(str(temporary), 0o555)
        try:
            os.rename(str(temporary), str(input_root))
        except OSError:
            if not input_root.is_dir():
                raise
            os.chmod(str(temporary), 0o755)
            shutil.rmtree(str(temporary))
        fsync_directory(input_root.parent)
    except BaseException:
        if temporary.exists():
            os.chmod(str(temporary), 0o755)
            shutil.rmtree(str(temporary))
        raise
    return verify_input_snapshot(code_manifest_sha256)


code_sha = build_code_snapshot()
input_sha = build_input_snapshot(code_sha)
print("code_manifest_sha256={}".format(code_sha))
print("input_manifest_sha256={}".format(input_sha))
PY
}

verify_landlock_cuda_runtime() {
  local executor="$1" code_root="$2" code_sha="$3"
  local input_root="$4" input_sha="$5"
  local probe_root allowed_root denied_sentinel allowed_output status=0
  probe_root="$(mktemp -d "${DATA_ROOT%/}/.mcln_landlock_cuda_probe.XXXXXX")"
  [[ "${probe_root}" == "${DATA_ROOT%/}/.mcln_landlock_cuda_probe."* ]] || {
    echo "unsafe CUDA probe root: ${probe_root}" >&2
    return 3
  }
  allowed_root="${probe_root}/allowed"
  denied_sentinel="${probe_root}/landlock_denied_sentinel"
  allowed_output="${allowed_root}/cuda_probe_output"
  mkdir "${allowed_root}"
  printf 'landlock-denial-sentinel\n' > "${denied_sentinel}"
  chmod 0666 "${denied_sentinel}"
  /usr/bin/env \
      MCLN_CUDA_PROBE_DENIED="${denied_sentinel}" \
      MCLN_CUDA_PROBE_ALLOWED="${allowed_output}" \
      "${PYTHON_BIN}" "${executor}" \
        --code-root "${code_root}" \
        --code-manifest-sha256 "${code_sha}" \
        --input-root "${input_root}" \
        --input-manifest-sha256 "${input_sha}" \
        --allow-write "${allowed_root}" \
        --allow-write /tmp --allow-write /dev/shm --allow-write /dev \
        --allow-write /proc/self/task \
        -- \
        /usr/bin/env -i \
          HOME=/tmp USER=root LOGNAME=root LANG=C.UTF-8 LC_ALL=C.UTF-8 \
          PATH=/root/miniconda3/envs/bdetr/bin:/usr/local/cuda/bin:/usr/bin:/bin \
          PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
          PYTHONPATH="${code_root}:${code_root}/pointnet2" \
          MCLN_CUDA_PROBE_DENIED="${denied_sentinel}" \
          MCLN_CUDA_PROBE_ALLOWED="${allowed_output}" \
          "${PYTHON_BIN}" -c '
from __future__ import print_function
import os
import pathlib
import torch

torch.cuda.set_device(0)
value = torch.ones(1, device="cuda")
torch.cuda.synchronize()
if float(value.item()) != 1.0:
    raise SystemExit("CUDA probe produced the wrong value")
denied = pathlib.Path(os.environ["MCLN_CUDA_PROBE_DENIED"])
try:
    with denied.open("r+b") as handle:
        handle.write(b"forbidden")
except PermissionError:
    pass
else:
    raise SystemExit("Landlock unexpectedly allowed a denied write")
allowed = pathlib.Path(os.environ["MCLN_CUDA_PROBE_ALLOWED"])
with allowed.open("wb") as handle:
    handle.write(b"allowed")
print("landlock_cuda_runtime_probe=pass device={} denied_write=blocked allowed_write=pass".format(
    torch.cuda.current_device()))
' || status=$?
  [[ ! -e "${allowed_output}" || "$(<"${allowed_output}")" == "allowed" ]] || status=3
  rm -f -- "${allowed_output}" "${denied_sentinel}"
  rmdir -- "${allowed_root}" "${probe_root}"
  if ((status != 0)); then
    echo "Landlock/CUDA runtime probe failed with status ${status}" >&2
    return "${status}"
  fi
}

pgrep -f '/root/mcln_official_rec_monitor.py nr3d ' >/dev/null || {
  echo "Nr3D official REC monitor is not running" >&2
  exit 4
}
if pgrep -f '[t]rain_dist_mod.py.*nr3d_mcln_joint_butdcls' >/dev/null; then
  echo "another Nr3D training process is still running" >&2
  exit 4
fi
[[ ! -e "${FORMAL_CLAIM}" ]] || {
  echo "formal tier hard-query branch was already claimed: ${FORMAL_CLAIM}" >&2
  exit 7
}
free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
((gpu_used < 500)) || { echo "GPU0 is busy (${gpu_used} MiB)" >&2; exit 4; }
((free_gb >= MIN_FREE_GB)) || { echo "need at least ${MIN_FREE_GB} GiB free" >&2; exit 5; }
lock_file="${DATA_ROOT%/}/output/network_v99/single_gpu.lock"
mkdir -p "$(dirname "${lock_file}")"
exec 8>"${lock_file}"
flock -n 8 || { echo "another V99 job owns ${lock_file}" >&2; exit 6; }
if [[ "${MODE}" == "preflight" ]]; then
  verify_landlock_cuda_runtime \
    "${LANDLOCK_SCRIPT_ORIGINAL}" \
    "${FIRST_RECOVERY_CODE_ROOT}" "${FIRST_RECOVERY_CODE_MANIFEST_SHA256}" \
    "${FIRST_RECOVERY_INPUT_ROOT}" "${FIRST_RECOVERY_INPUT_MANIFEST_SHA256}"
  flock -u 8
  exec 8>&-
  echo "preflight=pass formal_tier_aux_recovery2=true training_started=false epochs=58-62 patience=2"
  exit 0
fi
flock -u 8
exec 8>&-
current_screen_name="${STY##*.}"
[[ -n "${STY:-}" && "${current_screen_name}" == "${TRAIN_SCREEN_NAME}" ]] || {
  echo "formal launch must run inside screen ${TRAIN_SCREEN_NAME}" >&2
  exit 7
}
mkdir -p "${CONTROL_ROOT}"
readonly FORMAL_SCREEN_ID="${STY}"
FORMAL_SCREEN_START_TICKS="$(SCREEN_ID="${FORMAL_SCREEN_ID}" "${PYTHON_BIN}" - <<'PY'
import os
import pathlib


screen = os.environ["SCREEN_ID"]
pid_text, name = screen.split(".", 1)
pid = int(pid_text)
raw = (pathlib.Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
close_paren = raw.rfind(")")
if close_paren < 0:
    raise SystemExit("malformed screen process stat")
ticks = int(raw[close_paren + 2:].split()[19])
cmdline = (pathlib.Path("/proc") / str(pid) / "cmdline").read_bytes()
if b"SCREEN" not in cmdline or name.encode("utf-8") not in cmdline:
    raise SystemExit("STY does not identify the active SCREEN leader")
print(ticks)
PY
)"
readonly FORMAL_SCREEN_START_TICKS

snapshot_receipt="$(prepare_formal_snapshots)"
echo "${snapshot_receipt}"
CODE_MANIFEST_SHA256="$(printf '%s\n' "${snapshot_receipt}" | awk -F= '$1=="code_manifest_sha256" {print $2}')"
INPUT_MANIFEST_SHA256="$(printf '%s\n' "${snapshot_receipt}" | awk -F= '$1=="input_manifest_sha256" {print $2}')"
[[ "${CODE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid code snapshot receipt" >&2; exit 3;
}
[[ "${INPUT_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid input snapshot receipt" >&2; exit 3;
}
readonly CODE_MANIFEST_SHA256 INPUT_MANIFEST_SHA256

ROOT_DIR="${CODE_SNAPSHOT}"
SOURCE_CHECKPOINT="${INPUT_SNAPSHOT}/gf_detector_l6o256.pth"
REQUIRED_RESUME_CHECKPOINT="${INPUT_SNAPSHOT}/protected_e57.pth"
GUARD_SCRIPT="${INPUT_SNAPSHOT}/mcln_nr3d_tier_hard_query_e58_e62_guard.py"
WATCHDOG_SCRIPT="${INPUT_SNAPSHOT}/mcln_nr3d_tier_hard_query_watchdog.py"
LANDLOCK_SCRIPT="${INPUT_SNAPSHOT}/mcln_landlock_snapshot_exec.py"
APPROVAL="${INPUT_SNAPSHOT}/independent_density_approval.json"
AUDIT_DECISION="${INPUT_SNAPSHOT}/audit_decision.json"
AUDIT_RECEIPT="${INPUT_SNAPSHOT}/train_audit_receipt_epoch_58.json"
AUDIT_PROVENANCE="${INPUT_SNAPSHOT}/audit_provenance.json"
export BACKBONE_RESUME_CHECKPOINT="${REQUIRED_RESUME_CHECKPOINT}"
export BACKBONE_RESUME_SHA256="${REQUIRED_RESUME_SHA256}"
export BACKBONE_RESUME_EPOCH="${REQUIRED_RESUME_EPOCH}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/pointnet2"
cd "${ROOT_DIR}"

require_sha256 "${CODE_MANIFEST}" "${CODE_MANIFEST_SHA256}" "code snapshot manifest"
require_sha256 "${INPUT_MANIFEST}" "${INPUT_MANIFEST_SHA256}" "input snapshot manifest"
require_sha256 "${ROOT_DIR}/train_dist_mod.py" "${REQUIRED_TRAIN_ENTRY_SHA256}" "snapshot training entrypoint"
require_sha256 "${ROOT_DIR}/main_utils.py" "${REQUIRED_MAIN_UTILS_SHA256}" "snapshot main_utils"
require_sha256 "${ROOT_DIR}/models/losses.py" "${REQUIRED_LOSSES_SHA256}" "snapshot losses"
require_sha256 "${ROOT_DIR}/models/tier_hard_query_auxiliary.py" "${REQUIRED_AUXILIARY_SHA256}" "snapshot tier auxiliary"
require_sha256 "${ROOT_DIR}/models/mcln.py" "${REQUIRED_MODEL_SHA256}" "snapshot MCLN"
require_sha256 "${ROOT_DIR}/models/source_choice_selector.py" "${REQUIRED_SELECTOR_SHA256}" "snapshot selector"
require_sha256 "${ROOT_DIR}/models/mcln_training_groups.py" "${REQUIRED_TRAINING_GROUPS_SHA256}" "snapshot optimizer groups"
require_sha256 "${ROOT_DIR}/src/grounding_evaluator.py" "${REQUIRED_GROUNDING_EVALUATOR_SHA256}" "snapshot grounding evaluator"
require_sha256 "${ROOT_DIR}/utils/lr_scheduler.py" "${REQUIRED_LR_SCHEDULER_SHA256}" "snapshot LR scheduler"
require_sha256 "${ROOT_DIR}/scripts/run_dataset_v99_pipeline_tier_formal.sh" "${REQUIRED_PIPELINE_SHA256}" "snapshot formal pipeline"
require_sha256 "${GUARD_SCRIPT}" "${GUARD_SCRIPT_SHA256}" "snapshot guard"
require_sha256 "${WATCHDOG_SCRIPT}" "${WATCHDOG_SCRIPT_SHA256}" "snapshot watchdog"
require_sha256 "${LANDLOCK_SCRIPT}" "${LANDLOCK_SCRIPT_SHA256}" "snapshot Landlock executor"
require_sha256 "${SOURCE_CHECKPOINT}" "${SOURCE_SHA256}" "snapshot GroupFree checkpoint"
require_sha256 "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" "snapshot protected E57"
require_sha256 "${APPROVAL}" "${APPROVAL_SHA256}" "snapshot approval"
require_sha256 "${AUDIT_DECISION}" "${AUDIT_DECISION_SHA256}" "snapshot audit decision"
require_sha256 "${AUDIT_RECEIPT}" "${AUDIT_RECEIPT_SHA256}" "snapshot audit receipt"
require_sha256 "${AUDIT_PROVENANCE}" "${AUDIT_PROVENANCE_SHA256}" "snapshot audit provenance"
BACKBONE_SANDBOX_SCRIPT="${LANDLOCK_SCRIPT}"
BACKBONE_SANDBOX_CODE_ROOT="${CODE_SNAPSHOT}"
BACKBONE_SANDBOX_CODE_MANIFEST_SHA256="${CODE_MANIFEST_SHA256}"
BACKBONE_SANDBOX_INPUT_ROOT="${INPUT_SNAPSHOT}"
BACKBONE_SANDBOX_INPUT_MANIFEST_SHA256="${INPUT_MANIFEST_SHA256}"

wait_for_guard_ready() {
  READY_PATH="${GUARD_READY}" \
  EXPECTED_PARENT="$1" \
  EXPECTED_SCREEN="${FORMAL_SCREEN_ID}" \
  EXPECTED_SCREEN_TICKS="${FORMAL_SCREEN_START_TICKS}" \
  EXPECTED_EXPERIMENT="${EXP}" \
  EXPECTED_GUARD="${GUARD_SCRIPT}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import pathlib
import time


deadline = time.time() + 30
path = pathlib.Path(os.environ["READY_PATH"])
while time.time() < deadline:
    try:
        first = path.read_bytes()
        time.sleep(0.1)
        second = path.read_bytes()
        if first != second:
            continue
        payload = json.loads(first.decode("utf-8"))
        expected = {
            "schema": "mcln-tier-hard-query-guard-ready-v2",
            "run_root_parent": str(pathlib.Path(os.environ["EXPECTED_PARENT"]).resolve()),
            "screen": os.environ["EXPECTED_SCREEN"],
            "screen_start_ticks": int(os.environ["EXPECTED_SCREEN_TICKS"]),
            "experiment": os.environ["EXPECTED_EXPERIMENT"],
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise SystemExit("guard ready identity mismatch")
        pid = int(payload["pid"])
        ticks = int(payload["process_start_ticks"])
        stat_raw = (pathlib.Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8"
        )
        close_paren = stat_raw.rfind(")")
        actual_ticks = int(stat_raw[close_paren + 2:].split()[19])
        cmdline = (pathlib.Path("/proc") / str(pid) / "cmdline").read_bytes()
        if actual_ticks != ticks:
            raise SystemExit("guard PID identity mismatch")
        if os.environ["EXPECTED_GUARD"].encode("utf-8") not in cmdline:
            raise SystemExit("guard command identity mismatch")
        if os.environ["EXPECTED_EXPERIMENT"].encode("utf-8") not in cmdline:
            raise SystemExit("guard experiment identity mismatch")
        print("{} {}".format(pid, ticks))
        raise SystemExit(0)
    except (IOError, OSError, ValueError, KeyError, json.JSONDecodeError):
        time.sleep(0.2)
raise SystemExit("guard did not publish a valid ready receipt within 30 seconds")
PY
}

wait_for_watchdog_ready() {
  HEARTBEAT_PATH="${WATCHDOG_HEARTBEAT}" \
  EXPECTED_PARENT="$1" \
  EXPECTED_GUARD_PID="$2" \
  EXPECTED_GUARD_TICKS="$3" \
  EXPECTED_SCREEN="${FORMAL_SCREEN_ID}" \
  EXPECTED_SCREEN_TICKS="${FORMAL_SCREEN_START_TICKS}" \
  EXPECTED_EXPERIMENT="${EXP}" \
  EXPECTED_WATCHDOG="${WATCHDOG_SCRIPT}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import pathlib
import time


deadline = time.time() + 30
path = pathlib.Path(os.environ["HEARTBEAT_PATH"])
while time.time() < deadline:
    try:
        first = path.read_bytes()
        time.sleep(0.1)
        second = path.read_bytes()
        if first != second:
            continue
        payload = json.loads(first.decode("utf-8"))
        expected = {
            "schema": "mcln-tier-hard-query-watchdog-v1",
            "run_root_parent": str(pathlib.Path(os.environ["EXPECTED_PARENT"]).resolve()),
            "screen": os.environ["EXPECTED_SCREEN"],
            "screen_start_ticks": int(os.environ["EXPECTED_SCREEN_TICKS"]),
            "experiment": os.environ["EXPECTED_EXPERIMENT"],
            "guard_pid": int(os.environ["EXPECTED_GUARD_PID"]),
            "guard_start_ticks": int(os.environ["EXPECTED_GUARD_TICKS"]),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise SystemExit("watchdog heartbeat identity mismatch")
        pid = int(payload["watchdog_pid"])
        ticks = int(payload["watchdog_start_ticks"])
        stat_raw = (pathlib.Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8"
        )
        close_paren = stat_raw.rfind(")")
        actual_ticks = int(stat_raw[close_paren + 2:].split()[19])
        cmdline = (pathlib.Path("/proc") / str(pid) / "cmdline").read_bytes()
        if actual_ticks != ticks:
            raise SystemExit("watchdog PID identity mismatch")
        if os.environ["EXPECTED_WATCHDOG"].encode("utf-8") not in cmdline:
            raise SystemExit("watchdog command identity mismatch")
        if os.environ["EXPECTED_EXPERIMENT"].encode("utf-8") not in cmdline:
            raise SystemExit("watchdog experiment identity mismatch")
        print("{} {}".format(pid, ticks))
        raise SystemExit(0)
    except (IOError, OSError, ValueError, KeyError, json.JSONDecodeError):
        time.sleep(0.2)
raise SystemExit("watchdog did not publish a valid heartbeat within 30 seconds")
PY
}

start_backbone_guard() {
  local run_root_parent="$1" guard_identity guard_pid guard_ticks
  local watchdog_identity watchdog_pid watchdog_ticks
  [[ "$(readlink -f "/proc/$$/fd/9")" == "$(readlink -f "${lock_file}")" ]] || {
    echo "pipeline GPU lock fd9 is not bound to the expected lock" >&2
    return 8
  }
  verify_landlock_cuda_runtime \
    "${LANDLOCK_SCRIPT}" \
    "${CODE_SNAPSHOT}" "${CODE_MANIFEST_SHA256}" \
    "${INPUT_SNAPSHOT}" "${INPUT_MANIFEST_SHA256}"
  [[ ! -e "${FORMAL_CLAIM}" ]] || {
    echo "formal claim appeared before guarded launch" >&2
    return 8
  }
  READY="${GUARD_READY}" HEARTBEAT="${WATCHDOG_HEARTBEAT}" \
    "${PYTHON_BIN}" - <<'PY'
import os
for name in ("READY", "HEARTBEAT"):
    path = os.environ[name]
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
PY
  : > "${GUARD_LOG}"
  : > "${WATCHDOG_LOG}"
  setsid --fork "${PYTHON_BIN}" "${GUARD_SCRIPT}" \
    --control-root "${RECOVERY_ROOT}" \
    --run-root-parent "${run_root_parent}" \
    --screen "${FORMAL_SCREEN_ID}" \
    --screen-start-ticks "${FORMAL_SCREEN_START_TICKS}" \
    --experiment "${EXP}" \
    --ready-file "${GUARD_READY}" \
    --watchdog-heartbeat "${WATCHDOG_HEARTBEAT}" \
    >> "${GUARD_LOG}" 2>&1 < /dev/null
  guard_identity="$(wait_for_guard_ready "${run_root_parent}")"
  read -r guard_pid guard_ticks <<< "${guard_identity}"

  setsid --fork "${PYTHON_BIN}" "${WATCHDOG_SCRIPT}" \
    --guard-pid "${guard_pid}" \
    --guard-start-ticks "${guard_ticks}" \
    --guard-command-fragment "${GUARD_SCRIPT}" \
    --screen "${FORMAL_SCREEN_ID}" \
    --screen-start-ticks "${FORMAL_SCREEN_START_TICKS}" \
    --experiment "${EXP}" \
    --run-root-parent "${run_root_parent}" \
    --heartbeat "${WATCHDOG_HEARTBEAT}" \
    >> "${WATCHDOG_LOG}" 2>&1 < /dev/null
  watchdog_identity="$(wait_for_watchdog_ready \
    "${run_root_parent}" "${guard_pid}" "${guard_ticks}")"
  read -r watchdog_pid watchdog_ticks <<< "${watchdog_identity}"

  "${PYTHON_BIN}" "${LANDLOCK_SCRIPT}" \
    --code-root "${CODE_SNAPSHOT}" \
    --code-manifest-sha256 "${CODE_MANIFEST_SHA256}" \
    --input-root "${INPUT_SNAPSHOT}" \
    --input-manifest-sha256 "${INPUT_MANIFEST_SHA256}" \
    --verify-only

  CLAIM_PATH="${FORMAL_CLAIM}" \
  CLAIM_LAUNCHER="${LAUNCHER_PATH}" \
  CLAIM_LAUNCHER_SHA256="${ACTUAL_LAUNCHER_SHA256}" \
  CLAIM_CODE_MANIFEST_SHA256="${CODE_MANIFEST_SHA256}" \
  CLAIM_INPUT_MANIFEST_SHA256="${INPUT_MANIFEST_SHA256}" \
  CLAIM_APPROVAL_SHA256="${APPROVAL_SHA256}" \
  CLAIM_BOOTSTRAP="${TRUSTED_BOOTSTRAP_PATH}" \
  CLAIM_BOOTSTRAP_SHA256="${TRUSTED_BOOTSTRAP_SHA256}" \
  CLAIM_LANDLOCK_SHA256="${LANDLOCK_SCRIPT_SHA256}" \
  CLAIM_PIPELINE_SHA256="${REQUIRED_PIPELINE_SHA256}" \
  CLAIM_RESUME_SHA256="${REQUIRED_RESUME_SHA256}" \
  CLAIM_EXPERIMENT="${EXP}" \
  CLAIM_RUN_ROOT_PARENT="${run_root_parent}" \
  CLAIM_SCREEN="${FORMAL_SCREEN_ID}" \
  CLAIM_SCREEN_TICKS="${FORMAL_SCREEN_START_TICKS}" \
  CLAIM_GUARD_PID="${guard_pid}" \
  CLAIM_GUARD_TICKS="${guard_ticks}" \
  CLAIM_WATCHDOG_PID="${watchdog_pid}" \
  CLAIM_WATCHDOG_TICKS="${watchdog_ticks}" \
  CLAIM_RECOVERY_OF_SHA256="${FIRST_RECOVERY_CLAIM_SHA256}" \
  CLAIM_ORIGINAL_CLAIM_SHA256="${ORIGINAL_FORMAL_CLAIM_SHA256}" \
  CLAIM_ORIGINAL_STATE_SHA256="${ORIGINAL_GUARD_STATE_SHA256}" \
  CLAIM_ORIGINAL_GUARD_LOG_SHA256="${ORIGINAL_GUARD_LOG_SHA256}" \
  CLAIM_ORIGINAL_HEARTBEAT_SHA256="${ORIGINAL_WATCHDOG_HEARTBEAT_SHA256}" \
  CLAIM_ORIGINAL_LAUNCH_LOG_SHA256="${ORIGINAL_LAUNCH_LOG_SHA256}" \
  CLAIM_FIRST_STATE_SHA256="${FIRST_RECOVERY_GUARD_STATE_SHA256}" \
  CLAIM_FIRST_READY_SHA256="${FIRST_RECOVERY_GUARD_READY_SHA256}" \
  CLAIM_FIRST_GUARD_LOG_SHA256="${FIRST_RECOVERY_GUARD_LOG_SHA256}" \
  CLAIM_FIRST_WATCHDOG_LOG_SHA256="${FIRST_RECOVERY_WATCHDOG_LOG_SHA256}" \
  CLAIM_FIRST_HEARTBEAT_SHA256="${FIRST_RECOVERY_HEARTBEAT_SHA256}" \
  CLAIM_FIRST_SCREEN_LOG_SHA256="${FIRST_RECOVERY_SCREEN_LOG_SHA256}" \
  CLAIM_FIRST_LAUNCH_LOG_SHA256="${FIRST_RECOVERY_LAUNCH_LOG_SHA256}" \
  CLAIM_FIRST_CODE_MANIFEST_SHA256="${FIRST_RECOVERY_CODE_MANIFEST_SHA256}" \
  CLAIM_FIRST_INPUT_MANIFEST_SHA256="${FIRST_RECOVERY_INPUT_MANIFEST_SHA256}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import time


path = os.environ["CLAIM_PATH"]
payload = {
    "schema": "mcln-tier-hard-query-formal-recovery-claim-v2",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "launcher": os.environ["CLAIM_LAUNCHER"],
    "launcher_sha256": os.environ["CLAIM_LAUNCHER_SHA256"],
    "code_manifest_sha256": os.environ["CLAIM_CODE_MANIFEST_SHA256"],
    "input_manifest_sha256": os.environ["CLAIM_INPUT_MANIFEST_SHA256"],
    "approval_sha256": os.environ["CLAIM_APPROVAL_SHA256"],
    "bootstrap": os.environ["CLAIM_BOOTSTRAP"],
    "bootstrap_sha256": os.environ["CLAIM_BOOTSTRAP_SHA256"],
    "landlock_executor_sha256": os.environ["CLAIM_LANDLOCK_SHA256"],
    "pipeline_sha256": os.environ["CLAIM_PIPELINE_SHA256"],
    "resume_sha256": os.environ["CLAIM_RESUME_SHA256"],
    "experiment": os.environ["CLAIM_EXPERIMENT"],
    "run_root_parent": os.environ["CLAIM_RUN_ROOT_PARENT"],
    "screen": os.environ["CLAIM_SCREEN"],
    "screen_start_ticks": int(os.environ["CLAIM_SCREEN_TICKS"]),
    "guard": {
        "pid": int(os.environ["CLAIM_GUARD_PID"]),
        "start_ticks": int(os.environ["CLAIM_GUARD_TICKS"]),
    },
    "watchdog": {
        "pid": int(os.environ["CLAIM_WATCHDOG_PID"]),
        "start_ticks": int(os.environ["CLAIM_WATCHDOG_TICKS"]),
    },
    "epochs": [58, 59, 60, 61, 62],
    "patience": 2,
    "baseline_hits025": 4463,
    "target_hits025": 4724,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "checkpoint_retention_metrics": ["rec_acc025"],
    "landlock_cuda_runtime_probe": {
        "allowed_proc_path": "/proc/self/task",
        "cuda_set_device": True,
        "denied_write_blocked": True,
    },
    "recovery_of_claim_sha256": os.environ["CLAIM_RECOVERY_OF_SHA256"],
    "prior_startup_failures": {
        "original": {
            "claim_sha256": os.environ["CLAIM_ORIGINAL_CLAIM_SHA256"],
            "guard_state_sha256": os.environ["CLAIM_ORIGINAL_STATE_SHA256"],
            "guard_log_sha256": os.environ["CLAIM_ORIGINAL_GUARD_LOG_SHA256"],
            "watchdog_heartbeat_sha256":
                os.environ["CLAIM_ORIGINAL_HEARTBEAT_SHA256"],
            "launch_log_sha256":
                os.environ["CLAIM_ORIGINAL_LAUNCH_LOG_SHA256"],
            "training_started": False,
            "cause": "atomic_heartbeat_double_read_race",
        },
        "recovery1": {
            "claim_sha256": os.environ["CLAIM_RECOVERY_OF_SHA256"],
            "guard_state_sha256": os.environ["CLAIM_FIRST_STATE_SHA256"],
            "guard_ready_sha256": os.environ["CLAIM_FIRST_READY_SHA256"],
            "guard_log_sha256": os.environ["CLAIM_FIRST_GUARD_LOG_SHA256"],
            "watchdog_log_sha256":
                os.environ["CLAIM_FIRST_WATCHDOG_LOG_SHA256"],
            "watchdog_heartbeat_sha256":
                os.environ["CLAIM_FIRST_HEARTBEAT_SHA256"],
            "screen_log_sha256": os.environ["CLAIM_FIRST_SCREEN_LOG_SHA256"],
            "launch_log_sha256": os.environ["CLAIM_FIRST_LAUNCH_LOG_SHA256"],
            "code_manifest_sha256":
                os.environ["CLAIM_FIRST_CODE_MANIFEST_SHA256"],
            "input_manifest_sha256":
                os.environ["CLAIM_FIRST_INPUT_MANIFEST_SHA256"],
            "training_started": False,
            "cause": "landlock_denied_proc_self_task_cuda_error_304",
        },
    },
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
directory_fd = os.open(os.path.dirname(path), os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
  echo "formal_claim=${FORMAL_CLAIM}"
  echo "guard_pid=${guard_pid} watchdog_pid=${watchdog_pid}"
}
readonly -f start_backbone_guard

echo "guard_script=${GUARD_SCRIPT}"
echo "watchdog_script=${WATCHDOG_SCRIPT}"
echo "guard_contract=recovery2,baseline4463,target4724,E58-E62,patience2,fail_closed"
# shellcheck source=run_dataset_v99_pipeline_tier_formal.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline_tier_formal.sh"
