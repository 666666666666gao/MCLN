#!/usr/bin/env bash
set -euo pipefail

readonly TRUST_ROOT="/root/mcln_fpr_av4_audit_trust/v1"
readonly TRUSTED_STATIC_EXEC_PATH="${TRUST_ROOT}/mcln_fpr_audit_static_exec.x86_64"
readonly TRUSTED_STATIC_SOURCE_PATH="${TRUST_ROOT}/mcln_fpr_audit_static_exec.c"
readonly TRUSTED_LAUNCHER_PATH="${TRUST_ROOT}/run_nr3d_fpr_tv_density_audit.sh"
readonly TRUSTED_STATIC_EXEC_SHA256="2eafbf471810ddf04ccfcc8ba568dbe1fb03d56658b5bd87e073d6c5051f6773"
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

readonly ROOT_DIR="/root/autodl-tmp/mcln_fpr_av4_audit_review_20260901"
readonly SOURCE_ROOT="/root/autodl-tmp/mcln_fpr_av4_audit_review_20260901"
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
readonly RUNTIME_MANIFEST="${SOURCE_ROOT}/scripts/fpr_tv_counterfactual_parent_runtime_manifest_v2.json"
readonly RUNTIME_MANIFEST_SHA256="f09e490789680a8e7105cb1167f6f6f025a9a83a8998657eda3c9e1b4c9ab807"
readonly REQUIRED_TRAIN_ENTRY_SHA256="add7ad10e5a91248ccf1c593a280df73b6a19c2cdc3b53b898ce2f64be628c64"
readonly REQUIRED_MAIN_UTILS_SHA256="b40c6f6ca83ec68f655feb820de788f7398b0e71574cf73a9f6b22b137fba47e"
readonly REQUIRED_LOSSES_SHA256="54a74ba522d57d74e934f60a6f8a8becdbe0f188ab99c948ea55c436fbd9fe36"
readonly REQUIRED_VERIFIER_SHA256="744b714c339a823c76739685f7d7cea3be0381dad4c7d0d00f4f02c08e58e32a"
readonly REQUIRED_VERIFIER_TEST_SHA256="ed484753a7ba41f8cdcd1cf5ff63112cd4aae3a648db7c7d4780f783a9664cdc"
readonly REQUIRED_SPEC_SHA256="befba370c82b549b0545c1977fac88b578de8f012f4a86943c58d1dd19e10246"
readonly REQUIRED_MODEL_SHA256="5a605328a6d12d610a479ec40f039f91708edcd6c3ce5d559ea4e1494026952c"
readonly REQUIRED_SELECTOR_SHA256="61211cea91de4c3f3c11e44bc5ff035711307f3381d5a4069d4ab7842bee17dc"
readonly REQUIRED_DATASET_SHA256="800bac2caf9b7a319bdc200f60386000e4e374a559d9581113a8eb57d525f9f0"
readonly REQUIRED_SNAPSHOT_EXECUTOR_SHA256="839b6d8479b94e610288723219b0149203dde89ab0a85a6dd3bd9d4776d04c88"
readonly REQUIRED_RESUME_EPOCH=57
readonly AUDIT_EPOCH=58
readonly AUDIT_BATCHES=100
readonly BATCH_SIZE=16
readonly EXPECTED_EVAL_SAMPLE_COUNT=7899
readonly MASTER_PORT=5427
readonly MIN_FREE_GB=6
readonly SNAPSHOT_OWNER_UID=65532
readonly SNAPSHOT_OWNER_GID=65532
readonly EXP="nr3d_v99_fpr_tv_av4_counterfactual_parent_audit_e58_b100_b16x1"
readonly AUDIT_ROOT="${OUTPUT_ROOT}/audit/${EXP}_one_shot"
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
  require_sha256 "${ROOT_DIR}/FPR_TV_COUNTERFACTUAL_PARENT_AUDIT_SPEC_2026-09-01.md" \
    "${REQUIRED_SPEC_SHA256}" "counterfactual Parent audit specification"
  require_sha256 "${ROOT_DIR}/models/mcln.py" \
    "${REQUIRED_MODEL_SHA256}" "MCLN model"
  require_sha256 "${ROOT_DIR}/models/source_choice_selector.py" \
    "${REQUIRED_SELECTOR_SHA256}" "source-choice selector"
  require_sha256 "${ROOT_DIR}/src/joint_det_dataset.py" \
    "${REQUIRED_DATASET_SHA256}" "dataset implementation"
  require_sha256 "${ROOT_DIR}/scripts/mcln_density_audit_snapshot_exec.py" \
    "${REQUIRED_SNAPSHOT_EXECUTOR_SHA256}" "capability-drop executor"
  require_sha256 "${SOURCE_CHECKPOINT}" \
    "${SOURCE_CHECKPOINT_SHA256}" "protected E57 checkpoint"
  require_sha256 "${GROUPFREE_CHECKPOINT}" \
    "${GROUPFREE_SHA256}" "GroupFree checkpoint"
  require_sha256 "${DATA_MANIFEST}" \
    "${DATA_MANIFEST_SHA256}" "Nr3D data manifest"
  require_sha256 "${RUNTIME_MANIFEST}" \
    "${RUNTIME_MANIFEST_SHA256}" "reviewed runtime manifest"
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
        "mcln-fpr-tv-counterfactual-parent-reviewed-runtime-v2"):
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
      tests/test_fpr_av4_audit_contract.py
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

require_fixed_inputs
verify_or_copy_runtime_closure verify-source "${SOURCE_ROOT}"
verify_dataset_manifest "${DATA_MANIFEST}" "${DATA_MANIFEST_SHA256}"
verify_checkpoint_contract
run_default_off_regression
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
  echo "preflight=pass audit_only=true batches=${AUDIT_BATCHES}"
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
readonly RUNTIME_MANIFEST_SNAPSHOT="${INPUT_ROOT}/fpr_tv_counterfactual_parent_runtime_manifest_v2.json"
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
chown "${SNAPSHOT_OWNER_UID}:${SNAPSHOT_OWNER_GID}" "${INPUT_ROOT}"
chmod 0555 "${INPUT_ROOT}"
mkdir "${RUNTIME_HOME}/hf" "${RUNTIME_HOME}/xdg" "${RUNTIME_HOME}/torch"
readonly SNAPSHOT_EXECUTOR="${CODE_SNAPSHOT}/scripts/mcln_density_audit_snapshot_exec.py"

readonly LAUNCH_LOG="${RUNTIME_OUTPUT}/launch.log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1
echo "audit_only=true long_training_authorized=false"
echo "audit_root=${AUDIT_ROOT}"
echo "resume_snapshot=${RESUME_SNAPSHOT}"
echo "audit_epoch=${AUDIT_EPOCH} audit_batches=${AUDIT_BATCHES}"

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
  --max_train_batches "${AUDIT_BATCHES}"
  --gradient_accumulation_steps 1
  --local_rank 0
  --detect_intermediate --use_soft_token_loss --use_contrastive_align
  --log_dir "${RUNTIME_OUTPUT}"
  --pp_checkpoint "${GROUPFREE_SNAPSHOT}"
  --pp_checkpoint_sha256 "${GROUPFREE_SHA256}"
  --self_attend --skip_missing_superpoints
  --checkpoint_path "${RESUME_SNAPSHOT}"
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
  --expected_eval_sample_count "${EXPECTED_EVAL_SAMPLE_COUNT}"
)
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
  "${TRUSTED_STATIC_EXEC_PATH}" "${TRUSTED_STATIC_SOURCE_PATH}" <<'PY'
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
    "counterfactual_parent_audit_spec": os.path.join(
        review_root,
        "FPR_TV_COUNTERFACTUAL_PARENT_AUDIT_SPEC_2026-09-01.md",
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
    "static_executor": os.path.realpath(static_executor),
    "static_source": os.path.realpath(static_source),
    "train_command": os.path.realpath(command),
}
payload = {
    "schema": "mcln-fpr-tv-counterfactual-parent-pre-audit-v1",
    "code_snapshot_root": os.path.realpath(code_root),
    "review_root": os.path.realpath(review_root),
    "paths": paths,
    "observed_sha256": {
        name: sha256_file(path) for name, path in paths.items()
    },
}
with open(output_path, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
chmod 0444 "${PRE_AUDIT}"

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
verify_dataset_manifest "${DATA_MANIFEST_SNAPSHOT}" \
  "${DATA_MANIFEST_SHA256}"
verify_or_copy_runtime_closure verify-source "${SOURCE_ROOT}"
require_fixed_inputs
if [[ "$(sha256_file "${LAUNCHER_PATH}")" != \
      "${LAUNCHER_START_SHA256}" ]]; then
  echo "launcher changed during the bounded audit" >&2
  exit 8
fi

mapfile -t receipts < <(
  find "${RUNTIME_OUTPUT}" -type f \
    -name "train_audit_receipt_epoch_${AUDIT_EPOCH}.json" -print
)
if ((${#receipts[@]} != 1)); then
  echo "expected one bounded-audit receipt, found ${#receipts[@]}" >&2
  exit 8
fi
if find "${AUDIT_ROOT}" -path "${INPUT_ROOT}" -prune -o -type f \
    -name 'eval_metrics*.json' -print -quit \
    | grep -q .; then
  echo "audit unexpectedly evaluated validation data" >&2
  exit 8
fi
if find "${AUDIT_ROOT}" -path "${INPUT_ROOT}" -prune -o -type f \
    -name '*.pth' -print -quit \
    | grep -q .; then
  echo "audit unexpectedly saved a checkpoint" >&2
  exit 8
fi

readonly RECEIPT="${receipts[0]}"
readonly DECISION="${AUDIT_ROOT}/counterfactual_parent_decision.json"
/usr/bin/env -i \
  PATH="${PATH}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "${PYTHON_BIN}" - "${RECEIPT}" "${RESUME_SNAPSHOT}" \
  "${PRE_AUDIT}" "${DECISION}" "${AUDIT_BATCHES}" \
  "${BATCH_SIZE}" "${SOURCE_CHECKPOINT_SHA256}" \
  "${GROUPFREE_SHA256}" "${DATA_MANIFEST_SHA256}" \
  "${RUNTIME_MANIFEST_SHA256}" "${LAUNCHER_START_SHA256}" \
  "${TRUSTED_STATIC_EXEC_SHA256}" "${TRUSTED_STATIC_SOURCE_SHA256}" <<'PY'
import hashlib
import json
import math
import os
import sys

receipt_path, checkpoint_path, pre_path, decision_path = sys.argv[1:5]
audit_batches, batch_size = (int(value) for value in sys.argv[5:7])
(
    expected_checkpoint_sha,
    expected_groupfree_sha,
    expected_data_manifest_sha,
    expected_runtime_manifest_sha,
    expected_launcher_sha,
    expected_static_executor_sha,
    expected_static_source_sha,
) = sys.argv[7:14]

def load_json_with_sha(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )

receipt, receipt_sha = load_json_with_sha(receipt_path)
pre, pre_sha = load_json_with_sha(pre_path)
if pre.get("schema") != "mcln-fpr-tv-counterfactual-parent-pre-audit-v1":
    raise SystemExit("unexpected pre-audit provenance schema")
expected_path_keys = {
    "launcher",
    "train_entry",
    "main_utils",
    "losses",
    "parent_relative_text_verifier",
    "parent_relative_text_verifier_tests",
    "counterfactual_parent_audit_spec",
    "model",
    "source_choice_selector",
    "dataset",
    "snapshot_executor",
    "resume_checkpoint",
    "groupfree_checkpoint",
    "data_manifest",
    "runtime_manifest",
    "static_executor",
    "static_source",
    "train_command",
}
if set(pre.get("paths", {})) != expected_path_keys:
    raise SystemExit("pre-audit provenance path closure changed")
current = {
    name: sha256_file(path) for name, path in pre["paths"].items()
}
if current != pre.get("observed_sha256"):
    raise SystemExit("code or input artifact drifted during audit")
fixed_hashes = {
    "resume_checkpoint": expected_checkpoint_sha,
    "groupfree_checkpoint": expected_groupfree_sha,
    "data_manifest": expected_data_manifest_sha,
    "runtime_manifest": expected_runtime_manifest_sha,
    "launcher": expected_launcher_sha,
    "static_executor": expected_static_executor_sha,
    "static_source": expected_static_source_sha,
}
for name, expected in fixed_hashes.items():
    if current.get(name) != expected:
        raise SystemExit("fixed artifact SHA mismatch: {}".format(name))

if receipt.get("schema") != "mcln-train-loss-epoch-v1":
    raise SystemExit("unexpected bounded-audit receipt schema")
expected_scalars = {
    "epoch": 58,
    "max_train_batches": audit_batches,
    "batch_count": audit_batches,
    "optimizer_step_count": audit_batches,
    "sample_count": audit_batches * batch_size,
    "audit_only": True,
    "formal_validation_accessed": False,
    "long_training_authorized": False,
}
for name, expected in expected_scalars.items():
    if receipt.get(name) != expected:
        raise SystemExit("bounded receipt {} changed".format(name))
if os.path.realpath(receipt.get("checkpoint_path", "")) != os.path.realpath(
        checkpoint_path):
    raise SystemExit("bounded audit resumed a different checkpoint")
for section in ("loss_means", "stat_means"):
    values = receipt.get(section)
    if not isinstance(values, dict) or not values:
        raise SystemExit("missing {}".format(section))
    if any(not finite_number(value) for value in values.values()):
        raise SystemExit("non-finite value in {}".format(section))

state = receipt.get("state_integrity")
outputs = receipt.get("output_integrity")
if (
        not isinstance(state, dict)
        or state.get("frozen_exact") is not True
        or state.get("trainable_changed") is not True):
    raise SystemExit("frozen/trainable state integrity failed")
if (
        not isinstance(outputs, dict)
        or outputs.get("exact") is not True):
    raise SystemExit("frozen output sentinel integrity failed")
for partition in ("frozen", "trainable"):
    before = state.get("before", {}).get(partition, {})
    after = state.get("after", {}).get(partition, {})
    if not all(
            isinstance(record.get("sha256"), str)
            and len(record["sha256"]) == 64
            for record in (before, after)):
        raise SystemExit("state integrity hashes are incomplete")
if (
        state["before"]["frozen"]["sha256"]
        != state["after"]["frozen"]["sha256"]
        or state["before"]["trainable"]["sha256"]
        == state["after"]["trainable"]["sha256"]):
    raise SystemExit("state integrity hashes contradict flags")
if (
        outputs.get("before", {}).get("combined_sha256")
        != outputs.get("after", {}).get("combined_sha256")):
    raise SystemExit("output sentinel hashes contradict exact flag")

stats = receipt["stat_means"]
required_suffixes = (
    "sample_count",
    "supervised_candidate_count",
    "positive_candidate_count",
    "positive_row_count",
    "fix_pair_count",
    "break_pair_count",
    "neutral_pair_count",
    "nonfinite_count",
    "transition_utility_loss",
    "risk_loss",
    "selected_score_gradient_l1",
)
required = ["grad_norm", "parent_relative_text_verifier_counterfactual_view_count"]
for prefix in (
        "parent_relative_text_verifier_actual_",
        "parent_relative_text_verifier_counterfactual_",
):
    required.extend(prefix + suffix for suffix in required_suffixes)
missing = [name for name in required if name not in stats]
if missing:
    raise SystemExit("missing counterfactual audit stats: {}".format(
        ",".join(missing)
    ))
observed = {name: float(stats[name]) for name in required}
actual_prefix = "parent_relative_text_verifier_actual_"
counterfactual_prefix = "parent_relative_text_verifier_counterfactual_"
actual_samples = observed[actual_prefix + "sample_count"]
counterfactual_samples = observed[counterfactual_prefix + "sample_count"]
actual_positive_rows = observed[actual_prefix + "positive_row_count"]
counterfactual_positive_rows = observed[
    counterfactual_prefix + "positive_row_count"
]
actual_positive_ratio = (
    actual_positive_rows / actual_samples if actual_samples > 0.0 else 0.0
)
counterfactual_positive_ratio = (
    counterfactual_positive_rows / counterfactual_samples
    if counterfactual_samples > 0.0 else 0.0
)
checks = {
    "global_gradient_nonzero": observed["grad_norm"] > 0.0,
    "actual_score_gradient_nonzero": (
        observed[actual_prefix + "selected_score_gradient_l1"] > 0.0
    ),
    "counterfactual_score_gradient_nonzero": (
        observed[
            counterfactual_prefix + "selected_score_gradient_l1"
        ] > 0.0
    ),
    "actual_nonfinite_zero": (
        observed[actual_prefix + "nonfinite_count"] == 0.0
    ),
    "counterfactual_nonfinite_zero": (
        observed[counterfactual_prefix + "nonfinite_count"] == 0.0
    ),
    "actual_supervision_present": (
        actual_samples > 0.0
        and observed[actual_prefix + "supervised_candidate_count"] > 0.0
    ),
    "counterfactual_supervision_present": (
        counterfactual_samples > 0.0
        and observed[
            counterfactual_prefix + "supervised_candidate_count"
        ] > 0.0
    ),
    "counterfactual_views_present": (
        observed[
            "parent_relative_text_verifier_counterfactual_view_count"
        ] > 0.0
    ),
    "positive_density_doubled": (
        counterfactual_positive_rows > 0.0
        and counterfactual_positive_ratio
        >= 2.0 * actual_positive_ratio
    ),
    "counterfactual_fix_present": (
        observed[counterfactual_prefix + "fix_pair_count"] > 0.0
    ),
    "counterfactual_break_present": (
        observed[counterfactual_prefix + "break_pair_count"] > 0.0
    ),
    "frozen_state_exact": state["frozen_exact"] is True,
    "trainable_state_changed": state["trainable_changed"] is True,
    "frozen_output_exact": outputs["exact"] is True,
}
passed = all(checks.values())
decision = {
    "schema": "mcln-fpr-tv-counterfactual-parent-audit-v1",
    "audit_only": True,
    "formal_validation_accessed": False,
    "long_training_authorized": False,
    "next_step_if_passed": "independent_scene_disjoint_review_only",
    "audit_batches": audit_batches,
    "sample_count": audit_batches * batch_size,
    "counterfactual_density_gate_passed": passed,
    "checks": checks,
    "observed": observed,
    "actual_positive_row_ratio": actual_positive_ratio,
    "counterfactual_positive_row_ratio": counterfactual_positive_ratio,
    "state_integrity": state,
    "output_integrity": outputs,
    "receipt": os.path.realpath(receipt_path),
    "receipt_sha256": receipt_sha,
    "checkpoint": os.path.realpath(checkpoint_path),
    "checkpoint_sha256": expected_checkpoint_sha,
    "checkpoint_epoch": 57,
    "groupfree_checkpoint_sha256": expected_groupfree_sha,
    "data_manifest_sha256": expected_data_manifest_sha,
    "runtime_manifest_sha256": expected_runtime_manifest_sha,
    "launcher_sha256": expected_launcher_sha,
    "static_executor_sha256": expected_static_executor_sha,
    "static_source_sha256": expected_static_source_sha,
    "code_snapshot_root": pre.get("code_snapshot_root"),
    "pre_audit_provenance": os.path.realpath(pre_path),
    "pre_audit_provenance_sha256": pre_sha,
    "code_and_input_sha256": current,
}
temporary = decision_path + ".tmp.{}".format(os.getpid())
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(decision, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, decision_path)
directory_fd = os.open(
    os.path.dirname(decision_path),
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
print("bounded_audit_receipt=validated")
print("counterfactual_density_gate_passed={}".format(str(passed).lower()))
print("long_training_authorized=false")
PY

chmod 0444 "${RECEIPT}" "${DECISION}" "${LAUNCH_LOG}"
echo "audit_receipt=${RECEIPT}"
echo "audit_receipt_sha256=$(sha256sum "${RECEIPT}" | awk '{print $1}')"
echo "counterfactual_parent_decision=${DECISION}"
echo "counterfactual_parent_decision_sha256=$(sha256sum "${DECISION}" | awk '{print $1}')"
echo "approval_status=pending_independent_scene_disjoint_review"
if ! /usr/bin/env -i PATH="${PATH}" PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" - "${DECISION}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    decision = json.load(handle)
if decision.get("counterfactual_density_gate_passed") is not True:
    raise SystemExit(1)
PY
then
  echo "counterfactual Parent density gate failed" >&2
  exit 20
fi
