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
readonly E57_CHECKPOINT="${OUTPUT_ROOT}/audit/nr3d_mcln_joint_butdcls_v99_relation_cf_conservative_anchor_density_v2_audit_e58_b100_b16x1_w4p2_one_shot/resume_e57.pth"
readonly E57_SHA256="fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
readonly E57_EPOCH=57
readonly REPLAY_ROOT="${OUTPUT_ROOT}/control/hard_example_replay_top5_e57"
readonly REPLAY_MANIFEST="${REPLAY_ROOT}/manifest.json"
readonly REPLAY_MANIFEST_SHA256="6a7eaf375dfacd9880c4ee5a3d0611e40c806a792e2a83962ba133bb7a47b988"
readonly BINDING_RECEIPT="${REPLAY_ROOT}/real_dataset_binding_receipt.json"
readonly BINDING_RECEIPT_SHA256="7b5dfa4050fc34f6c38a252077047adb7c227dd946762f8a719e584cd3f8a63b"
readonly CONTROL_ROOT="${OUTPUT_ROOT}/control/hard_example_replay_top5_e57_e58_e59_once"
readonly EXP="nr3d_mcln_joint_butdcls_v99_e57_hard_replay_top5_e58_e59_b16a1"
readonly BATCH_SIZE=16
readonly MAX_EPOCH=59
readonly EXPECTED_EVAL_SAMPLE_COUNT=7899
readonly MASTER_PORT=5339
readonly MIN_FREE_GB=7
readonly BACKBONE_JOINT_TRAINING=1
readonly INFERENCE_USES_GROUND_TRUTH=1
readonly USE_BACKBONE_INITIALIZATION=1
readonly TASK_CHECKPOINT_TRANSFER=0
readonly BACKBONE_AUGMENT_DET=0

readonly REQUIRED_MAIN_UTILS_SHA256="a2a355042b7af61e17576941dec1fe79d3820805f83461e60679e0e3e00182f8"
readonly REQUIRED_REPLAY_MODULE_SHA256="804f25a560807e026cea9fed24f2bc66069804a8d6b35c0943daeee0f8c3d117"
readonly REQUIRED_PIPELINE_SHA256="264eabacb8c034ad51f4fc30ce33ef990408a19e68400069a74c575f58da31a9"
readonly REQUIRED_TRAIN_ENTRY_SHA256="8f78cca50174423d0c4ab0b3c76a1fa6f22bbd1b179bd547013243ad199996f1"
readonly REQUIRED_MCLN_SHA256="3b9a2b88e9fa36c6f94b1595ae3812818ca5fdeb4b1db64c68f467912c04c9b1"
readonly REQUIRED_SELECTOR_SHA256="61211cea91de4c3f3c11e44bc5ff035711307f3381d5a4069d4ab7842bee17dc"
readonly REQUIRED_TRAINING_GROUPS_SHA256="0298531a3adefd2f010cccc65a0724cf9f0521374446cfe7a9081dfacdd437ce"
readonly REQUIRED_LOSSES_SHA256="48de298038ca9996e0c135dfc42ad5d271a0827d1a8c03309cc71d51a4e3082f"
readonly REQUIRED_DATASET_SHA256="800bac2caf9b7a319bdc200f60386000e4e374a559d9581113a8eb57d525f9f0"
readonly REQUIRED_EVALUATOR_SHA256="0173b31a7a818f872c210b01a4e5d17601c4e5f10ec8d97f78c7e537fa44e062"

CHECKPOINT_RETENTION_METRICS=(rec_acc025)
DATASET_LR_ARGS=(
  --lr_backbone 1e-3
  --lr 1e-4
  --lr_decay_epochs 150
  --warmup-epoch -1
)
BACKBONE_EXTRA_ARGS=(
  --print_freq 20
  --joint_det
  --butd_cls
  --hard_example_replay_manifest "${REPLAY_MANIFEST}"
  --hard_example_replay_manifest_sha256 "${REPLAY_MANIFEST_SHA256}"
)

export BACKBONE_RESUME_CHECKPOINT="${E57_CHECKPOINT}"
export BACKBONE_RESUME_SHA256="${E57_SHA256}"
export BACKBONE_RESUME_EPOCH="${E57_EPOCH}"
export VALIDATE_BACKBONE_RESUME=0
export MODE="${MODE:-preflight}"
case "${MODE}" in
  preflight|backbone) ;;
  *) echo "MODE must be preflight or backbone" >&2; exit 2 ;;
esac

: "${REVIEWED_LAUNCHER_SHA256:?set the independently reviewed launcher SHA-256}"
if [[ ! "${REVIEWED_LAUNCHER_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "REVIEWED_LAUNCHER_SHA256 must be lowercase 64-hex" >&2
  exit 2
fi

require_sha256() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || {
    echo "missing ${label}: ${path}" >&2
    exit 3
  }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA-256 changed: ${actual} != ${expected}" >&2
    exit 3
  }
}

require_sha256 "${BASH_SOURCE[0]}" \
  "${REVIEWED_LAUNCHER_SHA256}" "reviewed hard-replay launcher"
require_sha256 "${ROOT_DIR}/main_utils.py" \
  "${REQUIRED_MAIN_UTILS_SHA256}" "hard-replay main_utils"
require_sha256 "${ROOT_DIR}/models/hard_example_replay.py" \
  "${REQUIRED_REPLAY_MODULE_SHA256}" "hard-replay sampler"
require_sha256 "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh" \
  "${REQUIRED_PIPELINE_SHA256}" "shared V99 launcher"
require_sha256 "${ROOT_DIR}/train_dist_mod.py" \
  "${REQUIRED_TRAIN_ENTRY_SHA256}" "training entry point"
require_sha256 "${ROOT_DIR}/models/mcln.py" \
  "${REQUIRED_MCLN_SHA256}" "MCLN network"
require_sha256 "${ROOT_DIR}/models/source_choice_selector.py" \
  "${REQUIRED_SELECTOR_SHA256}" "V99 selector"
require_sha256 "${ROOT_DIR}/models/mcln_training_groups.py" \
  "${REQUIRED_TRAINING_GROUPS_SHA256}" "optimizer grouping"
require_sha256 "${ROOT_DIR}/models/losses.py" \
  "${REQUIRED_LOSSES_SHA256}" "training losses"
require_sha256 "${ROOT_DIR}/src/joint_det_dataset.py" \
  "${REQUIRED_DATASET_SHA256}" "joint dataset"
require_sha256 "${ROOT_DIR}/src/grounding_evaluator.py" \
  "${REQUIRED_EVALUATOR_SHA256}" "grounding evaluator"
require_sha256 "${SOURCE_CHECKPOINT}" "${SOURCE_SHA256}" \
  "GroupFree checkpoint"
require_sha256 "${E57_CHECKPOINT}" "${E57_SHA256}" \
  "protected full-state E57 checkpoint"
require_sha256 "${REPLAY_MANIFEST}" "${REPLAY_MANIFEST_SHA256}" \
  "hard-replay manifest"
require_sha256 "${BINDING_RECEIPT}" "${BINDING_RECEIPT_SHA256}" \
  "real-dataset binding receipt"

if [[ -e "${CONTROL_ROOT}" || -L "${CONTROL_ROOT}" ]]; then
  echo "hard-replay formal attempt already consumed: ${CONTROL_ROOT}" >&2
  exit 3
fi

"${PYTHON_BIN}" - \
  "${E57_CHECKPOINT}" "${REPLAY_MANIFEST}" "${BINDING_RECEIPT}" <<'PY'
import json
import sys

import torch

checkpoint_path, manifest_path, receipt_path = sys.argv[1:]
checkpoint = torch.load(checkpoint_path, map_location="cpu")
config = vars(checkpoint["config"])
optimizer = checkpoint["optimizer"]
scheduler = checkpoint["scheduler"]
if checkpoint.get("epoch") != 57:
    raise SystemExit("protected checkpoint is not E57")
if len(optimizer["param_groups"]) != 4 or len(optimizer["state"]) != 716:
    raise SystemExit("protected E57 optimizer topology mismatch")
current = [float(group["lr"]) for group in optimizer["param_groups"]]
initial = [float(group["initial_lr"]) for group in optimizer["param_groups"]]
if current != [1e-5, 1e-4, 1e-5, 1.25e-5]:
    raise SystemExit("protected E57 current-LR mismatch")
if initial != [1e-4, 1e-3, 1e-4, 1.25e-4]:
    raise SystemExit("protected E57 initial-LR mismatch")
if [float(value) for value in scheduler["base_lrs"]] != initial:
    raise SystemExit("protected E57 scheduler base-LR mismatch")
if [float(value) for value in scheduler["_last_lr"]] != current:
    raise SystemExit("protected E57 scheduler current-LR mismatch")
if int(scheduler["last_epoch"]) != 159942:
    raise SystemExit("protected E57 scheduler progress mismatch")
if int(scheduler["_step_count"]) != 159943:
    raise SystemExit("protected E57 scheduler step-count mismatch")
if dict(scheduler["milestones"]) != {423706: 1}:
    raise SystemExit("protected E57 scheduler milestone mismatch")
required_config = {
    "augment_det": False,
    "batch_size": 16,
    "butd_cls": True,
    "butd_gt": False,
    "dataset": ["nr3d"],
    "eval_use_selector_choice_scores": True,
    "joint_det": True,
    "lr": 1e-4,
    "lr_backbone": 1e-3,
    "lr_decay_epochs": [150],
    "lr_scheduler": "step",
    "source_choice_selector_choice_target": (
        "precision_gain_default_sourcewise_focal_bce"
    ),
    "source_choice_selector_default_source": "default",
    "source_choice_selector_hidden_dim": 288,
    "source_choice_selector_loss_weight": 0.5,
    "source_choice_selector_lr": 1.25e-4,
    "source_choice_selector_min_iou_gap": 0.03,
    "source_choice_selector_sources": (
        "default,default_rank_blend_contrastive010"
    ),
    "test_dataset": "nr3d",
    "use_source_choice_selector": True,
    "warmup_epoch": -1,
}
for name, expected in required_config.items():
    observed = config.get(name)
    if type(observed) is not type(expected) or observed != expected:
        raise SystemExit(
            "protected E57 config mismatch for {}: {!r} != {!r}".format(
                name, observed, expected
            )
        )
if config.get("gradient_accumulation_steps", 1) != 1:
    raise SystemExit("protected E57 accumulation mismatch")
if config.get("drop_incomplete_accumulation_group", False):
    raise SystemExit("protected E57 accumulation-tail mismatch")

with open(manifest_path, "rb") as handle:
    manifest = json.loads(handle.read().decode("utf-8"))
with open(receipt_path, "rb") as handle:
    receipt = json.loads(handle.read().decode("utf-8"))
expected_manifest = {
    "base_dataset_size": 32919,
    "dataset": "nr3d",
    "hard_count": 1548,
    "joint_dataset_size": 44909,
    "repeat_count": 1,
    "schema": "mcln-hard-example-replay-v1",
}
for name, expected in expected_manifest.items():
    if manifest.get(name) != expected:
        raise SystemExit("hard-replay manifest mismatch for {}".format(name))
if manifest.get("criteria") != {
    "default_top1_iou_lte": 0.25,
    "default_topk": 5,
    "topk_oracle_iou_gt": 0.25,
}:
    raise SystemExit("hard-replay criteria changed")
if manifest.get("diagnostics") != {
    "top1_hits025": 30589,
    "top1_hits050": 27220,
    "top5_hits025": 32137,
    "top5_hits050": 30213,
}:
    raise SystemExit("hard-replay diagnostic counts changed")
if manifest.get("candidate_cache", {}).get("checkpoint_sha256") != (
    "fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
):
    raise SystemExit("hard-replay cache checkpoint changed")
expected_receipt = {
    "all_base_seen": True,
    "dataset_size": 44909,
    "hard_count": 1548,
    "hard_total_occurrences": 3103,
    "nonhard_exactly_once": True,
}
for name, expected in expected_receipt.items():
    if receipt.get(name) != expected:
        raise SystemExit("real-dataset binding mismatch for {}".format(name))
plan = receipt.get("plan", {})
for name, expected in {
    "base_dataset_size": 32919,
    "hard_count": 1548,
    "joint_dataset_size": 44909,
    "manifest_sha256": (
        "6a7eaf375dfacd9880c4ee5a3d0611e40c806a792e2a83962ba133bb7a47b988"
    ),
    "padding_size": 7,
    "per_rank_samples": 46464,
    "repeat_count": 1,
    "total_size": 46464,
}.items():
    if plan.get(name) != expected:
        raise SystemExit("real-dataset replay plan mismatch for {}".format(name))
print(
    "hard_replay_preflight=pass epoch57 groups4 states716 "
    "dataset44909 hard1548 total46464 batches2904 padding7"
)
PY

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
lock_file="${DATA_ROOT%/}/output/network_v99/single_gpu.lock"
mkdir -p "$(dirname "${lock_file}")"
exec 8>"${lock_file}"
if ! flock -n 8; then
  echo "another V99 job owns ${lock_file}" >&2
  exit 6
fi
flock -u 8
exec 8>&-

readonly LAUNCHER_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
echo "mode=${MODE} launcher_sha256=${LAUNCHER_SHA256}"
echo "gpu0_memory_used_mib=${gpu_used} free_disk_gib=${free_gb}"
echo "formal_baseline_hits025=4475 strict_target_hits025=4724"
echo "hard_replay_plan=44909+1548+7=46464 samples; 2904 B16 steps"

start_backbone_guard() {
  local backbone_run_dir="$1"
  "${PYTHON_BIN}" - \
    "${CONTROL_ROOT}" "${backbone_run_dir}" "${BASH_SOURCE[0]}" \
    "${LAUNCHER_SHA256}" "${EXP}" "${REPLAY_MANIFEST}" \
    "${REPLAY_MANIFEST_SHA256}" "${BINDING_RECEIPT}" \
    "${BINDING_RECEIPT_SHA256}" "${E57_CHECKPOINT}" \
    "${E57_SHA256}" <<'PY'
import json
import os
import sys
import time

(
    control_root,
    run_dir,
    launcher,
    launcher_sha256,
    experiment,
    manifest,
    manifest_sha256,
    binding_receipt,
    binding_receipt_sha256,
    checkpoint,
    checkpoint_sha256,
) = sys.argv[1:]
parent = os.path.dirname(control_root)
if not os.path.isdir(parent):
    raise SystemExit("hard-replay control parent disappeared")
os.mkdir(control_root, 0o700)
parent_fd = os.open(parent, os.O_RDONLY)
try:
    os.fsync(parent_fd)
finally:
    os.close(parent_fd)
payload = {
    "schema": "mcln-hard-example-replay-formal-claim-v1",
    "claimed_unix_time": time.time(),
    "pid": os.getpid(),
    "experiment": experiment,
    "backbone_run_dir": os.path.realpath(run_dir),
    "launcher": os.path.realpath(launcher),
    "launcher_sha256": launcher_sha256,
    "main_utils_sha256": (
        "a2a355042b7af61e17576941dec1fe79d3820805f83461e60679e0e3e00182f8"
    ),
    "replay_module_sha256": (
        "804f25a560807e026cea9fed24f2bc66069804a8d6b35c0943daeee0f8c3d117"
    ),
    "manifest": os.path.realpath(manifest),
    "manifest_sha256": manifest_sha256,
    "binding_receipt": os.path.realpath(binding_receipt),
    "binding_receipt_sha256": binding_receipt_sha256,
    "checkpoint": os.path.realpath(checkpoint),
    "checkpoint_sha256": checkpoint_sha256,
    "formal_baseline_hits025": 4475,
    "strict_target_hits025": 4724,
    "training_contract": {
        "dataset": ["nr3d"],
        "test_dataset": "nr3d",
        "joint_det": True,
        "butd_cls": True,
        "butd_gt": False,
        "augment_det": False,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "max_epoch": 59,
        "resume_epoch": 57,
        "replay_base_rows": 44909,
        "replay_hard_rows": 1548,
        "replay_padding_rows": 7,
        "samples_per_epoch": 46464,
        "optimizer_steps_per_epoch": 2904,
        "checkpoint_retention_metrics": ["rec_acc025"],
    },
}
temporary = os.path.join(control_root, "claim.json.tmp.{}".format(os.getpid()))
claim_path = os.path.join(control_root, "claim.json")
with open(temporary, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(temporary, 0o444)
os.replace(temporary, claim_path)
root_fd = os.open(control_root, os.O_RDONLY)
try:
    os.fsync(root_fd)
finally:
    os.close(root_fd)
print("hard_replay_formal_claim={}".format(claim_path))
PY
}

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
