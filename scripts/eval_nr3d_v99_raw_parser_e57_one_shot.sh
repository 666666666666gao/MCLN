#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT"
readonly CHECKPOINT="${DATA_ROOT}/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/official_best_rec025_epoch_57_0p56500823.pth"
readonly CHECKPOINT_SHA256="fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
readonly BUNDLE="${DATA_ROOT}/output/network_v99_baseline_gt/nr3d/control/raw_parser_conservative_syntax_v1_20260829_r2/nr3d_conservative_syntax_v1.bundle"
readonly BUNDLE_SHA256="1bb14e411debf1736569cdbf532987311289e10efe6492331cbd73da9c3cbcb6"
readonly GROUPFREE="${DATA_ROOT}/gf_detector_l6o256.pth"
readonly GROUPFREE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly RUN_ROOT="${DATA_ROOT}/output/network_v99_raw_parser_eval/nr3d_e57_conservative_syntax_v1_one_shot"
readonly EXP="nr3d_mcln_joint_butdcls_v99_e57_raw_parser_conservative_syntax_v1_one_shot"
readonly LOCK_FILE="${DATA_ROOT}/output/network_v99/single_gpu.lock"
readonly MASTER_PORT=5391
readonly EXPECTED_SAMPLE_COUNT=7899
readonly BASELINE_HITS025=4463
readonly BASELINE_HITS050=3749
readonly TARGET_MIN_HITS025=4724
readonly MIN_FREE_GB=7
readonly LAUNCHER_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

readonly TRAIN_SHA256="8f78cca50174423d0c4ab0b3c76a1fa6f22bbd1b179bd547013243ad199996f1"
readonly MAIN_SHA256="a80820c6e931eea5a1716082c94457d5776bae627af0796de4db68fa9ebeeeb4"
readonly DATASET_SHA256="800bac2caf9b7a319bdc200f60386000e4e374a559d9581113a8eb57d525f9f0"
readonly CACHE_SHA256="aa4c5949ba017a9f8a44f63caf73669717428eb9725e458027f3053de4d0e749"
readonly PARSER_SHA256="3c8d454022aed8133be5e35c0258db9c2b4d61cc56247ca6d92d5d80f0ea5793"
readonly MODEL_SHA256="3b9a2b88e9fa36c6f94b1595ae3812818ca5fdeb4b1db64c68f467912c04c9b1"
readonly SELECTOR_SHA256="61211cea91de4c3f3c11e44bc5ff035711307f3381d5a4069d4ab7842bee17dc"
readonly ADAPTER_SHA256="dc32c6adfde80af0449b28415a5a4d9ffcb9a5115b8894ab0a2f7c6ab9b11fbb"
readonly LOSSES_SHA256="cb0ba618ea5a126eb41503691a0c2853aceb3a803bd3fae557178b0e81a29816"
readonly EVALUATOR_SHA256="0173b31a7a818f872c210b01a4e5d17601c4e5f10ec8d97f78c7e537fa44e062"

MODE="${MODE:-preflight}"
readonly MODE
case "${MODE}" in
  preflight|eval) ;;
  *) echo "MODE must be preflight or eval" >&2; exit 2 ;;
esac
if (($# != 0)); then
  echo "usage: MODE=preflight|eval $0" >&2
  exit 2
fi
cd "${ROOT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/pointnet2"

require_sha256() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || {
    echo "missing ${label}: ${path}" >&2
    exit 3
  }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA changed: expected ${expected}, got ${actual}" >&2
    exit 3
  }
}

verify_fixed_inputs() {
  require_sha256 "${ROOT_DIR}/train_dist_mod.py" "${TRAIN_SHA256}" "train entry"
  require_sha256 "${ROOT_DIR}/main_utils.py" "${MAIN_SHA256}" "main utils"
  require_sha256 "${ROOT_DIR}/src/joint_det_dataset.py" "${DATASET_SHA256}" "dataset"
  require_sha256 "${ROOT_DIR}/src/legacy_scene_graph_cache.py" "${CACHE_SHA256}" "cache loader"
  require_sha256 "${ROOT_DIR}/sng_parser/backends/spacy_parser.py" "${PARSER_SHA256}" "spaCy parser"
  require_sha256 "${ROOT_DIR}/models/mcln.py" "${MODEL_SHA256}" "MCLN model"
  require_sha256 "${ROOT_DIR}/models/source_choice_selector.py" "${SELECTOR_SHA256}" "selector"
  require_sha256 "${ROOT_DIR}/models/source_choice_adapter.py" "${ADAPTER_SHA256}" "selector adapter"
  require_sha256 "${ROOT_DIR}/models/losses.py" "${LOSSES_SHA256}" "losses"
  require_sha256 "${ROOT_DIR}/src/grounding_evaluator.py" "${EVALUATOR_SHA256}" "evaluator"
  require_sha256 "${GROUPFREE}" "${GROUPFREE_SHA256}" "GroupFree checkpoint"
  require_sha256 "${CHECKPOINT}" "${CHECKPOINT_SHA256}" "E57 checkpoint"
  require_sha256 "${BUNDLE}" "${BUNDLE_SHA256}" "raw parser bundle"
}

verify_fixed_code_root() {
  local root="$1"
  require_sha256 "${root}/train_dist_mod.py" "${TRAIN_SHA256}" "snapshot train entry"
  require_sha256 "${root}/main_utils.py" "${MAIN_SHA256}" "snapshot main utils"
  require_sha256 "${root}/src/joint_det_dataset.py" "${DATASET_SHA256}" "snapshot dataset"
  require_sha256 "${root}/src/legacy_scene_graph_cache.py" "${CACHE_SHA256}" "snapshot cache loader"
  require_sha256 "${root}/sng_parser/backends/spacy_parser.py" "${PARSER_SHA256}" "snapshot spaCy parser"
  require_sha256 "${root}/models/mcln.py" "${MODEL_SHA256}" "snapshot MCLN model"
  require_sha256 "${root}/models/source_choice_selector.py" "${SELECTOR_SHA256}" "snapshot selector"
  require_sha256 "${root}/models/source_choice_adapter.py" "${ADAPTER_SHA256}" "snapshot selector adapter"
  require_sha256 "${root}/models/losses.py" "${LOSSES_SHA256}" "snapshot losses"
  require_sha256 "${root}/src/grounding_evaluator.py" "${EVALUATOR_SHA256}" "snapshot evaluator"
}

copy_verified_snapshot() {
  local source="$1" destination="$2" expected="$3" label="$4"
  local temporary="${destination}.partial.$$"
  cp --reflink=auto --preserve=mode,timestamps -- "${source}" "${temporary}"
  chmod 0444 "${temporary}"
  require_sha256 "${temporary}" "${expected}" "snapshot ${label}"
  if [[ "$(stat -c '%d:%i' "${source}")" == "$(stat -c '%d:%i' "${temporary}")" ]]; then
    echo "snapshot ${label} unexpectedly reuses the source inode" >&2
    exit 8
  fi
  mv -- "${temporary}" "${destination}"
}

verify_code_snapshot_manifest() {
  local code_root="$1" manifest_path="$2"
  "${PYTHON_BIN}" - "${code_root}" "${manifest_path}" <<'PY'
import hashlib
import json
import os
import stat
import sys

root, manifest_path = sys.argv[1:]

def inventory(path):
    rows = []
    for current, directories, files in os.walk(path):
        directories.sort()
        files.sort()
        for name in directories:
            candidate = os.path.join(current, name)
            if os.path.islink(candidate):
                raise SystemExit("code snapshot contains a directory symlink: " + candidate)
        for name in files:
            candidate = os.path.join(current, name)
            relative = os.path.relpath(candidate, path)
            info = os.lstat(candidate)
            if not stat.S_ISREG(info.st_mode):
                raise SystemExit("code snapshot contains a non-regular file: " + candidate)
            digest = hashlib.sha256()
            with open(candidate, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            rows.append({
                "path": relative,
                "sha256": digest.hexdigest(),
                "size": info.st_size,
                "mode": stat.S_IMODE(info.st_mode),
            })
    return rows

with open(manifest_path, "rb") as handle:
    raw = handle.read()
manifest = json.loads(raw.decode("utf-8"))
if manifest.get("schema") != "mcln-immutable-code-snapshot-v1":
    raise SystemExit("unexpected code snapshot manifest schema")
if manifest.get("root") != root:
    raise SystemExit("code snapshot root changed")
if manifest.get("files") != inventory(root):
    raise SystemExit("code snapshot content changed")
print("code_snapshot_manifest_sha256=" + hashlib.sha256(raw).hexdigest())
PY
}

verify_fixed_inputs
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "one-shot run root already exists: ${RUN_ROOT}" >&2
  exit 4
}

"${PYTHON_BIN}" - \
  "${CHECKPOINT}" "${CHECKPOINT_SHA256}" \
  "${BUNDLE}" "${BUNDLE_SHA256}" <<'PY'
import hashlib
import math
import sys

import torch

from src.legacy_scene_graph_cache import load_legacy_scene_graph_cache

checkpoint_path, checkpoint_sha, bundle_path, bundle_sha = sys.argv[1:]
with open(checkpoint_path, "rb") as handle:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    if digest.hexdigest() != checkpoint_sha:
        raise SystemExit("checkpoint SHA changed while opening it")
    handle.seek(0)
    checkpoint = torch.load(handle, map_location="cpu")
config = checkpoint.get("config")
config = vars(config) if hasattr(config, "__dict__") else dict(config or {})
optimizer = checkpoint.get("optimizer", {})
groups = optimizer.get("param_groups", [])
expected_lrs = [1e-5, 1e-4, 1e-5, 1.25e-5]
actual_lrs = [group.get("lr") for group in groups]
if checkpoint.get("epoch") != 57:
    raise SystemExit("checkpoint is not completed E57")
if config.get("batch_size") != 16:
    raise SystemExit("checkpoint batch size is not 16")
if config.get("joint_det") is not True or config.get("butd_cls") is not True:
    raise SystemExit("checkpoint is not joint_det + butd_cls")
if config.get("use_source_choice_selector") is not True:
    raise SystemExit("checkpoint is not the V99 selector model")
if config.get("eval_use_selector_choice_scores") is not True:
    raise SystemExit("checkpoint evaluation selector contract changed")
if config.get("use_sacr_source") is not False:
    raise SystemExit("checkpoint unexpectedly uses SACR")
if len(groups) != 4 or len(optimizer.get("state", {})) != 716:
    raise SystemExit("checkpoint optimizer topology changed")
if len(actual_lrs) != 4 or any(
        value is None or not math.isclose(
            float(value), expected, rel_tol=0.0, abs_tol=1e-12
        ) for value, expected in zip(actual_lrs, expected_lrs)):
    raise SystemExit("checkpoint current LR state changed")
records, manifest = load_legacy_scene_graph_cache(
    bundle_path,
    expected_target_selection="conservative_syntax_v1",
    expected_bundle_sha256=bundle_sha,
)
if len(records) != 40098:
    raise SystemExit("raw bundle unique record count changed")
if manifest.get("records_by_dataset") != {"nr3d": 40098}:
    raise SystemExit("raw bundle dataset scope changed")
print("preflight_provenance=E57_V99_bundle_exact_verified")
PY

free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
if ((gpu_used >= 500)); then
  echo "GPU0 is busy (${gpu_used} MiB)" >&2
  exit 5
fi
if ((free_gb < MIN_FREE_GB)); then
  echo "need at least ${MIN_FREE_GB} GiB free under DATA_ROOT" >&2
  exit 6
fi
mkdir -p "$(dirname "${LOCK_FILE}")"
if [[ "${MODE}" == "preflight" ]]; then
  exec 8>"${LOCK_FILE}"
  flock -n 8 || { echo "another V99 job owns ${LOCK_FILE}" >&2; exit 7; }
  flock -u 8
  exec 8>&-
  echo "preflight=pass eval_type=formal_one_shot run_root=${RUN_ROOT}"
  exit 0
fi

exec 9>"${LOCK_FILE}"
flock -n 9 || { echo "another V99 job owns ${LOCK_FILE}" >&2; exit 7; }
mkdir -p "$(dirname "${RUN_ROOT}")"
mkdir "${RUN_ROOT}"
readonly LAUNCH_LOG="${RUN_ROOT}/launch.log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

readonly SNAPSHOT_ROOT="${RUN_ROOT}/consumed_snapshot"
readonly CODE_SNAPSHOT="${SNAPSHOT_ROOT}/code"
readonly INPUT_SNAPSHOT="${SNAPSHOT_ROOT}/inputs"
readonly SNAPSHOT_CHECKPOINT="${INPUT_SNAPSHOT}/e57_checkpoint.pth"
readonly SNAPSHOT_BUNDLE="${INPUT_SNAPSHOT}/nr3d_conservative_syntax_v1.bundle"
readonly SNAPSHOT_GROUPFREE="${INPUT_SNAPSHOT}/gf_detector_l6o256.pth"
readonly CODE_MANIFEST="${RUN_ROOT}/consumed_code_manifest.json"
readonly CONSUMED_PROVENANCE="${RUN_ROOT}/consumed_provenance.json"
mkdir -p "${CODE_SNAPSHOT}" "${INPUT_SNAPSHOT}"
case "${SNAPSHOT_ROOT}" in
  "${RUN_ROOT}"/*) ;;
  *) echo "snapshot root escaped the one-shot run root" >&2; exit 8 ;;
esac

rsync -a \
  --exclude '/pretained model/' \
  --exclude '/experiment_output/' \
  --exclude '/tensorboard_output/' \
  --exclude '/output/' \
  --exclude '/refine-logs/' \
  --exclude '/reports/' \
  --exclude '/.aris/' \
  --exclude '/.claude/' \
  --exclude '/.pytest_cache/' \
  --exclude '/.v*/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.bak' \
  --exclude '*.orig' \
  -- "${ROOT_DIR}/" "${CODE_SNAPSHOT}/"
verify_fixed_code_root "${CODE_SNAPSHOT}"

copy_verified_snapshot \
  "${CHECKPOINT}" "${SNAPSHOT_CHECKPOINT}" "${CHECKPOINT_SHA256}" "E57 checkpoint"
copy_verified_snapshot \
  "${BUNDLE}" "${SNAPSHOT_BUNDLE}" "${BUNDLE_SHA256}" "raw parser bundle"
copy_verified_snapshot \
  "${GROUPFREE}" "${SNAPSHOT_GROUPFREE}" "${GROUPFREE_SHA256}" "GroupFree checkpoint"

find "${SNAPSHOT_ROOT}" -type d -exec chmod 0555 {} +
find "${SNAPSHOT_ROOT}" -type f -exec chmod a-w {} +
"${PYTHON_BIN}" - "${CODE_SNAPSHOT}" "${CODE_MANIFEST}" <<'PY'
import hashlib
import json
import os
import stat
import sys

root, manifest_path = sys.argv[1:]
rows = []
for current, directories, files in os.walk(root):
    directories.sort()
    files.sort()
    for name in directories:
        candidate = os.path.join(current, name)
        if os.path.islink(candidate):
            raise SystemExit("code snapshot contains a directory symlink: " + candidate)
    for name in files:
        candidate = os.path.join(current, name)
        relative = os.path.relpath(candidate, root)
        info = os.lstat(candidate)
        if not stat.S_ISREG(info.st_mode):
            raise SystemExit("code snapshot contains a non-regular file: " + candidate)
        digest = hashlib.sha256()
        with open(candidate, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append({
            "path": relative,
            "sha256": digest.hexdigest(),
            "size": info.st_size,
            "mode": stat.S_IMODE(info.st_mode),
        })
manifest = {
    "schema": "mcln-immutable-code-snapshot-v1",
    "root": root,
    "files": rows,
}
with open(manifest_path, "x", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
chmod 0444 "${CODE_MANIFEST}"
verify_code_snapshot_manifest "${CODE_SNAPSHOT}" "${CODE_MANIFEST}"
readonly CODE_MANIFEST_SHA256="$(sha256sum "${CODE_MANIFEST}" | awk '{print $1}')"

"${PYTHON_BIN}" - \
  "${CONSUMED_PROVENANCE}" "${LAUNCHER_PATH}" \
  "${CODE_MANIFEST}" "${CODE_SNAPSHOT}" \
  "${SNAPSHOT_CHECKPOINT}" "${CHECKPOINT_SHA256}" \
  "${SNAPSHOT_BUNDLE}" "${BUNDLE_SHA256}" \
  "${SNAPSHOT_GROUPFREE}" "${GROUPFREE_SHA256}" <<'PY'
import hashlib
import json
import sys

(
    output_path, launcher_path, code_manifest_path, code_root,
    checkpoint_path, checkpoint_sha, bundle_path, bundle_sha,
    groupfree_path, groupfree_sha,
) = sys.argv[1:]

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

provenance = {
    "schema": "mcln-e57-raw-parser-consumed-provenance-v1",
    "launcher": {"path": launcher_path, "sha256": sha256_file(launcher_path)},
    "code": {
        "root": code_root,
        "manifest_path": code_manifest_path,
        "manifest_sha256": sha256_file(code_manifest_path),
    },
    "checkpoint": {"path": checkpoint_path, "sha256": checkpoint_sha},
    "bundle": {"path": bundle_path, "sha256": bundle_sha},
    "groupfree": {"path": groupfree_path, "sha256": groupfree_sha},
}
for key in ("checkpoint", "bundle", "groupfree"):
    if sha256_file(provenance[key]["path"]) != provenance[key]["sha256"]:
        raise SystemExit(key + " snapshot SHA changed before launch")
with open(output_path, "x", encoding="utf-8") as handle:
    json.dump(provenance, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
chmod 0444 "${CONSUMED_PROVENANCE}"
readonly CONSUMED_PROVENANCE_SHA256="$(sha256sum "${CONSUMED_PROVENANCE}" | awk '{print $1}')"
readonly LAUNCHER_SHA256="$(sha256sum "${LAUNCHER_PATH}" | awk '{print $1}')"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${CODE_SNAPSHOT}:${CODE_SNAPSHOT}/pointnet2"
cd "${CODE_SNAPSHOT}"

train_args=(
  "${CODE_SNAPSHOT}/train_dist_mod.py"
  --num_target 256 --sampling kps
  --num_encoder_layers 3 --num_decoder_layers 6
  --self_position_embedding loc_learned --query_points_obj_topk 4
  --use_color --weight_decay 0.0005
  --data_root "${DATA_ROOT}/"
  --val_freq 1 --batch_size 16 --num_workers 4
  --dataloader_prefetch_factor 2 --persistent_train_workers
  --save_freq 1 --print_freq 500
  --lr_backbone 1e-3 --lr 1e-4
  --lr_decay_epochs 150 --warmup-epoch -1
  --start_epoch 1 --max_epoch 240
  --dataset nr3d --test_dataset nr3d
  --detect_intermediate --joint_det --butd_cls
  --use_soft_token_loss --use_contrastive_align
  --self_attend --skip_missing_superpoints
  --log_dir "${RUN_ROOT}"
  --pp_checkpoint "${SNAPSHOT_GROUPFREE}"
  --checkpoint_path "${SNAPSHOT_CHECKPOINT}" --checkpoint_start_epoch 57
  --eval --model MCLN --exp "${EXP}"
  --use_source_choice_selector --eval_use_selector_choice_scores
  --source_choice_selector_sources default,default_rank_blend_contrastive010
  --source_choice_selector_default_source default
  --source_choice_selector_hidden_dim 288
  --source_choice_selector_lr 1.25e-4
  --source_choice_selector_loss_weight 0.5
  --source_choice_selector_choice_target precision_gain_default_sourcewise_focal_bce
  --source_choice_selector_min_iou_gap 0.03
  --expected_eval_sample_count "${EXPECTED_SAMPLE_COUNT}"
  --legacy_scene_graph_cache "${SNAPSHOT_BUNDLE}"
  --legacy_scene_graph_cache_strict
  --legacy_scene_graph_cache_expected_target_selection conservative_syntax_v1
  --legacy_scene_graph_cache_expected_sha256 "${BUNDLE_SHA256}"
)

printf 'formal_eval_command='
printf '%q ' "${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node 1 --master_port "${MASTER_PORT}" "${train_args[@]}"
printf '\n'
"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node 1 --master_port "${MASTER_PORT}" "${train_args[@]}"

verify_fixed_inputs
verify_fixed_code_root "${CODE_SNAPSHOT}"
require_sha256 "${SNAPSHOT_CHECKPOINT}" "${CHECKPOINT_SHA256}" "consumed E57 checkpoint"
require_sha256 "${SNAPSHOT_BUNDLE}" "${BUNDLE_SHA256}" "consumed raw parser bundle"
require_sha256 "${SNAPSHOT_GROUPFREE}" "${GROUPFREE_SHA256}" "consumed GroupFree checkpoint"
require_sha256 "${CODE_MANIFEST}" "${CODE_MANIFEST_SHA256}" "consumed code manifest"
require_sha256 "${CONSUMED_PROVENANCE}" "${CONSUMED_PROVENANCE_SHA256}" "consumed provenance"
require_sha256 "${LAUNCHER_PATH}" "${LAUNCHER_SHA256}" "launcher"
verify_code_snapshot_manifest "${CODE_SNAPSHOT}" "${CODE_MANIFEST}"
mapfile -t receipts < <(
  find "${RUN_ROOT}" -path "${SNAPSHOT_ROOT}" -prune -o \
    -type f -name 'eval_metrics_epoch_57.json' -print
)
mapfile -t configs < <(
  find "${RUN_ROOT}" -path "${SNAPSHOT_ROOT}" -prune -o \
    -type f -name 'config.json' -print
)
mapfile -t weights < <(
  find "${RUN_ROOT}" -path "${SNAPSHOT_ROOT}" -prune -o \
    -type f -name '*.pth' -print
)
if ((${#receipts[@]} != 1 || ${#configs[@]} != 1)); then
  echo "expected one metric receipt and one config" >&2
  exit 8
fi
if ((${#weights[@]} != 0)); then
  echo "eval-only run unexpectedly wrote weights" >&2
  exit 8
fi

readonly RECEIPT="${receipts[0]}"
readonly CONFIG="${configs[0]}"
readonly DECISION="${RUN_ROOT}/decision.json"
"${PYTHON_BIN}" - \
  "${RECEIPT}" "${CONFIG}" "${DECISION}" \
  "${CHECKPOINT}" "${CHECKPOINT_SHA256}" \
  "${SNAPSHOT_CHECKPOINT}" "${SNAPSHOT_BUNDLE}" "${BUNDLE_SHA256}" \
  "${SNAPSHOT_GROUPFREE}" "${GROUPFREE_SHA256}" \
  "${CODE_MANIFEST}" "${CODE_MANIFEST_SHA256}" \
  "${CONSUMED_PROVENANCE}" "${CONSUMED_PROVENANCE_SHA256}" \
  "${EXPECTED_SAMPLE_COUNT}" "${BASELINE_HITS025}" \
  "${BASELINE_HITS050}" "${TARGET_MIN_HITS025}" <<'PY'
import hashlib
import json
import os
import sys

(
    receipt_path, config_path, decision_path,
    source_checkpoint_path, checkpoint_sha,
    checkpoint_path, bundle_path, bundle_sha,
    groupfree_path, groupfree_sha,
    code_manifest_path, expected_code_manifest_sha,
    provenance_path, expected_provenance_sha,
    expected_count, baseline025, baseline050, target_min025,
) = sys.argv[1:]

def load_json_with_sha(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()

receipt, receipt_sha = load_json_with_sha(receipt_path)
config, config_sha = load_json_with_sha(config_path)
code_manifest, code_manifest_sha = load_json_with_sha(code_manifest_path)
provenance, provenance_sha = load_json_with_sha(provenance_path)
if code_manifest_sha != expected_code_manifest_sha:
    raise SystemExit("consumed code manifest SHA changed")
if provenance_sha != expected_provenance_sha:
    raise SystemExit("consumed provenance SHA changed")
expected_count = int(expected_count)
baseline025 = int(baseline025)
baseline050 = int(baseline050)
target_min025 = int(target_min025)
multiple = receipt.get("position_subgroups", {}).get("multiple", {})
if receipt.get("schema") != "mcln-retrain-metrics-v1":
    raise SystemExit("unexpected metric receipt schema")
if receipt.get("sample_count") != expected_count:
    raise SystemExit("top-level sample count changed")
if multiple.get("sample_count") != expected_count:
    raise SystemExit("formal multiple sample count changed")
hits025 = multiple.get("hits025")
hits050 = multiple.get("hits050")
if not isinstance(hits025, int) or not isinstance(hits050, int):
    raise SystemExit("formal hit counts are missing")
if abs(float(multiple.get("acc025")) - hits025 / expected_count) > 1e-12:
    raise SystemExit("Acc@0.25 is inconsistent with hits")
if abs(float(multiple.get("acc050")) - hits050 / expected_count) > 1e-12:
    raise SystemExit("Acc@0.50 is inconsistent with hits")
if config.get("legacy_scene_graph_cache") != bundle_path:
    raise SystemExit("config used a different raw bundle")
if config.get("legacy_scene_graph_cache_expected_sha256") != bundle_sha:
    raise SystemExit("config raw bundle SHA changed")
if config.get("legacy_scene_graph_cache_strict") is not True:
    raise SystemExit("config did not use strict raw cache")
if config.get("checkpoint_path") != checkpoint_path:
    raise SystemExit("config used a different checkpoint snapshot")
if config.get("pp_checkpoint") != groupfree_path:
    raise SystemExit("config used a different GroupFree snapshot")
if config.get("use_sacr_source") is not False:
    raise SystemExit("config unexpectedly enabled SACR")
if code_manifest.get("schema") != "mcln-immutable-code-snapshot-v1":
    raise SystemExit("unexpected consumed code manifest")
if provenance.get("schema") != "mcln-e57-raw-parser-consumed-provenance-v1":
    raise SystemExit("unexpected consumed provenance")
if provenance.get("checkpoint") != {"path": checkpoint_path, "sha256": checkpoint_sha}:
    raise SystemExit("consumed checkpoint provenance changed")
if provenance.get("bundle") != {"path": bundle_path, "sha256": bundle_sha}:
    raise SystemExit("consumed bundle provenance changed")
if provenance.get("groupfree") != {"path": groupfree_path, "sha256": groupfree_sha}:
    raise SystemExit("consumed GroupFree provenance changed")
if provenance.get("code", {}).get("manifest_sha256") != code_manifest_sha:
    raise SystemExit("consumed code manifest provenance changed")
decision = {
    "schema": "mcln-nr3d-raw-parser-e57-one-shot-decision-v1",
    "eval_type": "formal_one_shot",
    "source_checkpoint": {"path": source_checkpoint_path, "sha256": checkpoint_sha},
    "checkpoint": {"path": checkpoint_path, "sha256": checkpoint_sha},
    "bundle": {"path": bundle_path, "sha256": bundle_sha},
    "groupfree": {"path": groupfree_path, "sha256": groupfree_sha},
    "code_manifest": {"path": code_manifest_path, "sha256": code_manifest_sha},
    "consumed_provenance": {"path": provenance_path, "sha256": provenance_sha},
    "receipt": {"path": receipt_path, "sha256": receipt_sha},
    "config": {"path": config_path, "sha256": config_sha},
    "baseline": {"hits025": baseline025, "hits050": baseline050},
    "candidate": {
        "sample_count": expected_count,
        "hits025": hits025,
        "hits050": hits050,
        "acc025": hits025 / expected_count,
        "acc050": hits050 / expected_count,
    },
    "delta_hits": {
        "hits025": hits025 - baseline025,
        "hits050": hits050 - baseline050,
    },
    "strict_target": {
        "minimum_hits025": target_min025,
        "achieved": hits025 >= target_min025,
    },
    "official_best_weight_modified": False,
}
with open(decision_path, "x", encoding="utf-8") as handle:
    json.dump(decision, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(decision, indent=2, sort_keys=True))
PY
chmod 0444 \
  "${LAUNCH_LOG}" "${RECEIPT}" "${CONFIG}" "${DECISION}" \
  "${CODE_MANIFEST}" "${CONSUMED_PROVENANCE}"
echo "formal_one_shot_eval=complete decision=${DECISION}"
