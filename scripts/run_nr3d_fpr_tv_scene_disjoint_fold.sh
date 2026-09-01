#!/usr/bin/env bash
set -euo pipefail

readonly MODE="${MODE:-preflight}"
if [[ "${MODE}" != "preflight" && "${MODE}" != "backbone" ]]; then
  echo "MODE must be preflight or backbone" >&2
  exit 2
fi
if [[ "$#" -ne 1 || "$1" != "1" ]]; then
  echo "this preregistered v2 launcher accepts only unconsumed fold 1" >&2
  exit 2
fi
readonly FOLD="$1"

readonly SOURCE_ROOT="/root/autodl-tmp/mcln_fpr_scene_audit_v2_review_20260831"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT"
readonly E57_CHECKPOINT="${DATA_ROOT}/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/official_best_rec025_epoch_57_0p56652741.pth"
readonly E57_SHA256="76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1"
readonly GROUPFREE_CHECKPOINT="${DATA_ROOT}/gf_detector_l6o256.pth"
readonly GROUPFREE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly CONFIG_SHA256="7845c9a73a158f814cb7f8278fbb60abfd49f2bee6e5e1d0b1892c4401ed32c9"
readonly RUNTIME_MANIFEST="${SOURCE_ROOT}/scripts/fpr_scene_disjoint_runtime_manifest_v1.json"
readonly RUNTIME_MANIFEST_SHA256="ed255b26286ef5a9b2b5ed497563ea0221e1bf3ea63ea9d1075bbcd2223459e1"
readonly LOCK_FILE="/root/autodl-tmp/mcln_v99_backbone_gpu0.lock"
readonly OUTPUT_BASE="${DATA_ROOT}/output/network_v99_baseline_gt/nr3d/fpr_tv_scene_disjoint_v2"
readonly RUN_ROOT="${OUTPUT_BASE}/fold_${FOLD}"
readonly RUNTIME_OUTPUT="${RUN_ROOT}/runtime_output"
readonly RUNTIME_HOME="${RUN_ROOT}/runtime_home"
readonly CODE_SNAPSHOT="${RUN_ROOT}/code_snapshot"
readonly INPUT_SNAPSHOT="${RUN_ROOT}/input_snapshot"
readonly EXP="nr3d_v99_fpr_tv_v2_scene_fold${FOLD}_e58"
readonly MASTER_PORT="$((5390 + FOLD))"
readonly V1_OUTPUT_BASE="${DATA_ROOT}/output/network_v99_baseline_gt/nr3d/fpr_tv_scene_disjoint_v1"
readonly V1_FOLD0_DECISION="${V1_OUTPUT_BASE}/fold_0/decision.json"
readonly V1_FOLD0_DECISION_SHA256="02f1951b298f0c059a5adc4bbc2b9abe542c8a03f4202a91b9effd1882a22e67"
readonly V1_FOLD0_RECEIPT="${V1_OUTPUT_BASE}/fold_0/runtime_output/nr3d/nr3d_v99_fpr_tv_scene_fold0_e58/1788159699/fpr_scene_disjoint_audit_fold_0_epoch_58.json"
readonly V1_FOLD0_RECEIPT_SHA256="843bf42ceec16b55677ab7279877c115854fe4270f58f364f885631dc33aa8ff"
readonly V1_FOLD1_ROOT="${V1_OUTPUT_BASE}/fold_1"

readonly -a FIT_SCENES=(402 400 408 417 417)
readonly -a HOLDOUT_SCENES=(109 111 103 94 94)
readonly -a FIT_SAMPLES=(25790 25578 26590 26714 27004)
readonly -a HOLDOUT_SAMPLES=(7129 7341 6329 6205 5915)
readonly EXPECTED_FIT_SCENES="${FIT_SCENES[${FOLD}]}"
readonly EXPECTED_HOLDOUT_SCENES="${HOLDOUT_SCENES[${FOLD}]}"
readonly EXPECTED_FIT_SAMPLES="${FIT_SAMPLES[${FOLD}]}"
readonly EXPECTED_HOLDOUT_SAMPLES="${HOLDOUT_SAMPLES[${FOLD}]}"

readonly -a REVIEWED_PATHS=(
  "main_utils.py"
  "train_dist_mod.py"
  "models/parent_relative_text_verifier.py"
  "models/mcln.py"
  "models/losses.py"
  "models/source_choice_selector.py"
  "models/structured_slots.py"
  "models/sacr_head.py"
  "src/joint_det_dataset.py"
  "src/grounding_evaluator.py"
  "utils/record_tensorboard.py"
  "FPR_TV_SPEC_2026-08-31.md"
)
readonly -a REVIEWED_SHAS=(
  "cdfdc56114ba6840d19c4317c8fda3eca36910959a84185d892e6cc4b3d778d6"
  "2e97c5b422bf5a5713cbda8e631ce0c1b8c2881d5074109ebb7909341e22daf3"
  "a028106e52aef5396a26957c0520578cd2db3bf071e1a5319b141f76b3f99b70"
  "a9301a5fc9bac3b40e4450350a8d2eb4ba11c4763e734c5a09723a5232474db4"
  "474a0d1356a61ef1e7ecdc083273076adc4514643ac1974d03042280ebe09ee4"
  "61211cea91de4c3f3c11e44bc5ff035711307f3381d5a4069d4ab7842bee17dc"
  "78f5c2e3a1e794ebf8876f24126c67fbb0c404707d065f55847ea7d2b2ef3281"
  "1b35e0c1cbb3afe0b543e895ca3614fbff97558df6d4666c90c3e9fd3433a93d"
  "800bac2caf9b7a319bdc200f60386000e4e374a559d9581113a8eb57d525f9f0"
  "0173b31a7a818f872c210b01a4e5d17601c4e5f10ec8d97f78c7e537fa44e062"
  "25c0709e94010c53224ad97f946f24952ae34bdf81d88306e6c51ad4923a89b5"
  "67e4a677e6c44006125e01752e82eb4d8f3c053728606055274f2d5cb7e26a27"
)

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

verify_fold_history() {
  "${PYTHON_BIN}" - \
    "${V1_FOLD0_DECISION}" "${V1_FOLD0_DECISION_SHA256}" \
    "${V1_FOLD0_RECEIPT}" "${V1_FOLD0_RECEIPT_SHA256}" \
    "${V1_FOLD1_ROOT}" "${RUN_ROOT}" <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import stat
import sys

decision_path, decision_sha, receipt_path, receipt_sha = sys.argv[1:5]
absent_roots = sys.argv[5:]


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


decision = load_frozen_json(decision_path, decision_sha)
receipt = load_frozen_json(receipt_path, receipt_sha)
if decision.get("schema") != "mcln-fpr-tv-scene-fold-decision-v1":
    raise SystemExit("unexpected consumed fold decision schema")
if (decision.get("fold") != 0
        or decision.get("fold_gate_pass") is not False
        or decision.get("long_training_authorized") is not False
        or decision.get("receipt_path") != receipt_path
        or decision.get("receipt_sha256") != receipt_sha
        or decision.get("switch_count") != 876
        or decision.get("gate_failures") != [
            "acc025_fix_not_greater_than_break", "acc050_net_negative"
        ]):
    raise SystemExit("consumed fold decision contract mismatch")
thresholds = decision.get("thresholds", {})
if ({key: thresholds.get("025", {}).get(key)
     for key in ("fix_count", "break_count", "net_hits")} != {
         "fix_count": 40, "break_count": 82, "net_hits": -42
     } or
        {key: thresholds.get("050", {}).get(key)
         for key in ("fix_count", "break_count", "net_hits")} != {
             "fix_count": 119, "break_count": 398, "net_hits": -279
         }):
    raise SystemExit("consumed fold transition evidence mismatch")
if (receipt.get("schema") != "mcln-fpr-tv-scene-disjoint-audit-v1"
        or receipt.get("fold_gate_pass") is not False
        or receipt.get("long_training_authorized") is not False
        or receipt.get("split", {}).get("fold") != 0):
    raise SystemExit("consumed fold receipt contract mismatch")
for root in absent_roots:
    if os.path.lexists(root):
        raise SystemExit("unconsumed fold root already exists: {}".format(root))
print("fold_history_verified=consumed_v1_fold0_unconsumed_fold1")
PY
}

readonly LAUNCHER_PATH="$(readlink -f -- "$0")"
readonly LAUNCHER_START_SHA256="$(sha256_file "${LAUNCHER_PATH}")"

verify_or_copy_runtime_closure() {
  local mode="$1"
  local root="$2"
  local destination="${3:--}"
  "${PYTHON_BIN}" - "${RUNTIME_MANIFEST}" \
    "${RUNTIME_MANIFEST_SHA256}" "${mode}" "${root}" \
    "${destination}" <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import stat
import sys

manifest_path, expected_manifest_sha, mode, root, destination = sys.argv[1:]
with open(manifest_path, "rb") as handle:
    manifest_raw = handle.read()
if hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha:
    raise SystemExit("runtime manifest SHA-256 mismatch")
manifest = json.loads(manifest_raw.decode("utf-8"))
if manifest.get("schema") != "mcln-fpr-tv-reviewed-runtime-code-v1":
    raise SystemExit("unexpected runtime manifest schema")
records = manifest.get("files")
if not isinstance(records, dict) or len(records) != manifest.get("file_count"):
    raise SystemExit("runtime manifest record count mismatch")


def checked_relative(value):
    if not isinstance(value, str) or not value:
        raise SystemExit("invalid empty runtime path")
    normalized = os.path.normpath(value)
    if (normalized != value or os.path.isabs(value)
            or value == ".." or value.startswith("../")):
        raise SystemExit("unsafe runtime path: {}".format(value))
    return value


def read_verified(path, record, expected_mode):
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
    actual_mode = format(stat.S_IMODE(after.st_mode), "04o")
    if expected_mode is not None and actual_mode != expected_mode:
        raise SystemExit("runtime mode mismatch: {}".format(path))
    if len(payload) != record.get("size"):
        raise SystemExit("runtime size mismatch: {}".format(path))
    if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
        raise SystemExit("runtime SHA mismatch: {}".format(path))
    return payload


root = os.path.realpath(root)
if (mode in ("verify-source", "copy")
        and manifest.get("source_root") != root):
    raise SystemExit("runtime manifest source_root mismatch")
if mode == "copy":
    if not os.path.isdir(destination) or os.listdir(destination):
        raise SystemExit("runtime copy destination must be an empty directory")
for relative, record in sorted(records.items()):
    relative = checked_relative(relative)
    source = os.path.join(root, relative)
    if mode in ("verify-source", "copy"):
        payload = read_verified(source, record, record.get("mode"))
    elif mode == "verify-snapshot":
        payload = read_verified(source, record, "0444")
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
        os.chmod(target, 0o444)
if mode == "copy":
    for current, directories, _files in os.walk(destination, topdown=False):
        for name in directories:
            os.chmod(os.path.join(current, name), 0o555)
    os.chmod(destination, 0o555)
if mode == "verify-snapshot":
    observed = set()
    for current, directories, files in os.walk(root):
        for name in directories:
            path = os.path.join(current, name)
            if os.path.islink(path):
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

verify_reviewed_tree() {
  local root="$1"
  local index path observed
  for index in "${!REVIEWED_PATHS[@]}"; do
    path="${root}/${REVIEWED_PATHS[${index}]}"
    [[ -f "${path}" && ! -L "${path}" ]] || {
      echo "missing reviewed regular file: ${path}" >&2
      return 1
    }
    observed="$(sha256_file "${path}")"
    [[ "${observed}" == "${REVIEWED_SHAS[${index}]}" ]] || {
      echo "reviewed SHA drift: ${path}: ${observed}" >&2
      return 1
    }
  done
}

verify_inputs() {
  local observed
  observed="$(sha256_file "${E57_CHECKPOINT}")"
  [[ "${observed}" == "${E57_SHA256}" ]] || {
    echo "protected E57 SHA drift: ${observed}" >&2
    return 1
  }
  observed="$(sha256_file "${GROUPFREE_CHECKPOINT}")"
  [[ "${observed}" == "${GROUPFREE_SHA256}" ]] || {
    echo "GroupFree SHA drift: ${observed}" >&2
    return 1
  }
}

verify_resources() {
  local gpu_used free_kib
  gpu_used="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk 'NF {sum += $1} END {print sum + 0}')"
  [[ "${gpu_used}" -lt 500 ]] || {
    echo "GPU is not idle: ${gpu_used} MiB used" >&2
    return 1
  }
  free_kib="$(df -Pk /root/autodl-tmp | awk 'NR == 2 {print $4}')"
  [[ "${free_kib}" -ge 7340032 ]] || {
    echo "less than 7 GiB is free on /root/autodl-tmp" >&2
    return 1
  }
}

verify_fold_history
verify_reviewed_tree "${SOURCE_ROOT}"
verify_or_copy_runtime_closure verify-source "${SOURCE_ROOT}"
verify_inputs
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "one-shot fold root already exists: ${RUN_ROOT}" >&2
  exit 3
}
train_args=(
  --num_target 256 --sampling kps
  --num_encoder_layers 3 --num_decoder_layers 6
  --self_position_embedding loc_learned --query_points_obj_topk 4
  --use_color --weight_decay 0.0005
  --data_root "${DATA_ROOT}/"
  --val_freq 1 --batch_size 16
  --num_workers 4 --dataloader_prefetch_factor 2 --persistent_train_workers
  --save_freq 1 --print_freq 20 --rng_seed 0
  --lr_backbone 1e-3 --lr 1e-4 --lr_decay_epochs 150 --warmup-epoch -1
  --dataset nr3d --test_dataset nr3d
  --joint_det --butd_cls
  --max_train_batches 0 --gradient_accumulation_steps 1
  --local_rank 0
  --detect_intermediate --use_soft_token_loss --use_contrastive_align
  --log_dir "${RUNTIME_OUTPUT}"
  --pp_checkpoint "${INPUT_SNAPSHOT}/gf_detector_l6o256.pth"
  --self_attend --skip_missing_superpoints
  --checkpoint_path "${E57_CHECKPOINT}"
  --start_epoch 58 --max_epoch 58
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
  --fpr_scene_disjoint_fold "${FOLD}"
  --fpr_scene_disjoint_expected_fit_scenes "${EXPECTED_FIT_SCENES}"
  --fpr_scene_disjoint_expected_holdout_scenes "${EXPECTED_HOLDOUT_SCENES}"
  --fpr_scene_disjoint_expected_fit_samples "${EXPECTED_FIT_SAMPLES}"
  --fpr_scene_disjoint_expected_holdout_samples "${EXPECTED_HOLDOUT_SAMPLES}"
  --fpr_scene_disjoint_checkpoint_sha256 "${E57_SHA256}"
  --expected_eval_sample_count "${EXPECTED_HOLDOUT_SAMPLES}"
)

verify_preregistered_config() {
  (
    cd "${SOURCE_ROOT}"
    /usr/bin/env -i \
      PATH=/root/miniconda3/envs/bdetr/bin:/usr/bin:/bin \
      PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
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
    raise SystemExit("preregistered configuration mismatch")
print("preregistered_config_verified={}".format(expected))
PY
  )
}

verify_preregistered_config
if [[ "${MODE}" == "preflight" ]]; then
  verify_resources
  exec 8>"${LOCK_FILE}"
  flock -n 8 || {
    echo "GPU training lock is busy" >&2
    exit 4
  }
  flock -u 8
  exec 8>&-
  echo "preflight_pass fold=${FOLD} fit=${EXPECTED_FIT_SAMPLES}/${EXPECTED_FIT_SCENES} holdout=${EXPECTED_HOLDOUT_SAMPLES}/${EXPECTED_HOLDOUT_SCENES} config_sha256=${CONFIG_SHA256}"
  exit 0
fi

exec 9>"${LOCK_FILE}"
flock -n 9 || {
  echo "GPU training lock is busy" >&2
  exit 4
}
verify_resources
mkdir -p "${OUTPUT_BASE}"
mkdir "${RUN_ROOT}"
mkdir "${RUNTIME_OUTPUT}" "${RUNTIME_HOME}" "${CODE_SNAPSHOT}" "${INPUT_SNAPSHOT}"
exec > >(tee -a "${RUNTIME_OUTPUT}/launch.log") 2>&1

verify_or_copy_runtime_closure copy "${SOURCE_ROOT}" "${CODE_SNAPSHOT}"
verify_or_copy_runtime_closure verify-snapshot "${CODE_SNAPSHOT}"

cp --reflink=auto -- "${GROUPFREE_CHECKPOINT}" "${INPUT_SNAPSHOT}/gf_detector_l6o256.pth"
chmod 0444 "${INPUT_SNAPSHOT}/gf_detector_l6o256.pth"
[[ "$(sha256_file "${INPUT_SNAPSHOT}/gf_detector_l6o256.pth")" == "${GROUPFREE_SHA256}" ]]

mkdir "${RUNTIME_HOME}/hf" "${RUNTIME_HOME}/xdg" "${RUNTIME_HOME}/torch"

cd "${CODE_SNAPSHOT}"
/usr/bin/env -i \
  HOME="${RUNTIME_HOME}" USER=root LOGNAME=root LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PATH=/root/miniconda3/envs/bdetr/bin:/usr/local/cuda/bin:/usr/bin:/bin \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${CODE_SNAPSHOT}:${CODE_SNAPSHOT}/pointnet2" \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  HF_HOME="${RUNTIME_HOME}/hf" TRANSFORMERS_CACHE="${RUNTIME_HOME}/hf" \
  XDG_CACHE_HOME="${RUNTIME_HOME}/xdg" TORCH_HOME="${RUNTIME_HOME}/torch" \
  RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 LOCAL_SIZE=1 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT}" \
  "${PYTHON_BIN}" "${CODE_SNAPSHOT}/train_dist_mod.py" "${train_args[@]}"

verify_or_copy_runtime_closure verify-snapshot "${CODE_SNAPSHOT}"
[[ "$(sha256_file "${LAUNCHER_PATH}")" == "${LAUNCHER_START_SHA256}" ]] || {
  echo "launcher changed during the formal fold" >&2
  exit 6
}
[[ "$(sha256_file "${INPUT_SNAPSHOT}/gf_detector_l6o256.pth")" == "${GROUPFREE_SHA256}" ]] || {
  echo "consumed GroupFree snapshot changed" >&2
  exit 6
}
verify_inputs
mapfile -t receipts < <(find "${RUNTIME_OUTPUT}" -type f -name "fpr_scene_disjoint_audit_fold_${FOLD}_epoch_58.json" -print)
[[ "${#receipts[@]}" -eq 1 ]] || {
  echo "expected exactly one fold receipt, found ${#receipts[@]}" >&2
  exit 6
}
if find "${RUNTIME_OUTPUT}" -type f -name '*.pth' -print -quit | grep -q .; then
  echo "scene audit unexpectedly produced a weight" >&2
  exit 6
fi

"${PYTHON_BIN}" - "${receipts[0]}" "${RUN_ROOT}/decision.json" \
  "${FOLD}" "${EXPECTED_FIT_SAMPLES}" "${EXPECTED_HOLDOUT_SAMPLES}" \
  "${CONFIG_SHA256}" "${E57_SHA256}" "${GROUPFREE_SHA256}" \
  "${RUNTIME_MANIFEST_SHA256}" "${LAUNCHER_START_SHA256}" <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import sys

receipt_path, decision_path = sys.argv[1:3]
fold = int(sys.argv[3])
fit_samples = int(sys.argv[4])
holdout_samples = int(sys.argv[5])
config_sha256 = sys.argv[6]
checkpoint_sha256 = sys.argv[7]
groupfree_sha256 = sys.argv[8]
runtime_manifest_sha256 = sys.argv[9]
launcher_sha256 = sys.argv[10]
with open(receipt_path, "rb") as handle:
    raw = handle.read()
receipt = json.loads(raw.decode("utf-8"))
if receipt.get("schema") != "mcln-fpr-tv-scene-disjoint-audit-v1":
    raise SystemExit("unexpected fold receipt schema")
if receipt.get("epoch") != 58 or receipt.get("checkpoint_epoch") != 57:
    raise SystemExit("unexpected checkpoint/fit epoch")
if receipt.get("checkpoint_sha256") != checkpoint_sha256:
    raise SystemExit("protected checkpoint SHA mismatch")
if receipt.get("split", {}).get("fold") != fold:
    raise SystemExit("fold identity mismatch")
if receipt.get("frozen_config", {}).get("sha256") != config_sha256:
    raise SystemExit("preregistered configuration mismatch")
training = receipt.get("training", {})
if training.get("sample_count") != fit_samples:
    raise SystemExit("fit sample count mismatch")
if training.get("sample_identity_count") != fit_samples:
    raise SystemExit("fit identity count mismatch")
if training.get("sample_identity_unique_count") != fit_samples:
    raise SystemExit("fit identity uniqueness mismatch")
diagnostics = receipt.get("evaluation", {}).get(
    "parent_relative_text_verifier_scene_audit", {}
)
if diagnostics.get("sample_count") != holdout_samples:
    raise SystemExit("held-out sample count mismatch")
if receipt.get("generated_weights") != []:
    raise SystemExit("unexpected generated weights")
if receipt.get("long_training_authorized") is not False:
    raise SystemExit("scene audit must never authorize long training")
decision = {
    "schema": "mcln-fpr-tv-scene-fold-decision-v1",
    "fold": fold,
    "receipt_path": receipt_path,
    "receipt_sha256": hashlib.sha256(raw).hexdigest(),
    "config_sha256": config_sha256,
    "checkpoint_sha256": checkpoint_sha256,
    "groupfree_sha256": groupfree_sha256,
    "runtime_manifest_sha256": runtime_manifest_sha256,
    "launcher_sha256": launcher_sha256,
    "fold_gate_pass": receipt.get("fold_gate_pass") is True,
    "gate_failures": receipt.get("gate_failures", []),
    "switch_count": diagnostics.get("switch_count"),
    "thresholds": diagnostics.get("thresholds"),
    "next_stage": receipt.get("next_stage"),
    "long_training_authorized": False,
}
payload = (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode("utf-8")
fd = os.open(decision_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
try:
    os.write(fd, payload)
    os.fsync(fd)
finally:
    os.close(fd)
print(json.dumps(decision, sort_keys=True))
if not decision["fold_gate_pass"]:
    raise SystemExit(20)
PY

echo "fold_pass fold=${FOLD} decision=${RUN_ROOT}/decision.json"
