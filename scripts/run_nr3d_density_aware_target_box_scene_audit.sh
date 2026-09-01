#!/usr/bin/env bash
set -euo pipefail

readonly TRUST_ROOT="/root/mcln_density_scene_audit_trust/v1"
readonly TRUSTED_STATIC_EXEC_PATH="${TRUST_ROOT}/mcln_fpr_audit_static_exec.x86_64"
readonly TRUSTED_STATIC_SOURCE_PATH="${TRUST_ROOT}/mcln_fpr_audit_static_exec.c"
readonly TRUSTED_LAUNCHER_PATH="${TRUST_ROOT}/run_nr3d_fpr_tv_density_audit.sh"
readonly TRUSTED_STATIC_EXEC_SHA256="d63392f280a6563e6cd8439a44aa5da8eb68c59d71c7a5574aa2763915e02775"
readonly TRUSTED_STATIC_SOURCE_SHA256="0bf6cfcfb015a91474579ba0c0f186c49c6a38695601d904d3216724cc67dcdc"
readonly SOURCE_ROOT="/root/autodl-tmp/mcln_density_target_box_scene_review_20260901"
readonly RUNNER_PATH="${SOURCE_ROOT}/scripts/run_density_target_box_scene_audit.py"
readonly REVIEWED_RUNNER_SHA256="513a03ac1677efc4c3a38dfbfd50fe1110491b73c5c3b0a17e2beaadeae2bd4f"
readonly REVIEWED_RUNTIME_MANIFEST_SHA256="04977c404fb759722d56e8bbeadb383a7113f4cec8e6d7dbde24d35f3f48c354"

[[ "${MCLN_FPR_TRUSTED_CLEAN_ENV:-}" == "1" ]] || {
  echo "scene-audit launcher requires the reviewed static executor" >&2
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
  echo "scene-audit static provenance is incomplete" >&2
  exit 2
}
[[ "${MCLN_FPR_STATIC_PARENT_PID}" == "${PPID}" ]] || {
  echo "scene-audit launcher parent identity drifted" >&2
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
  echo "scene-audit static process identity changed" >&2
  exit 2
}
mapfile -d '' -t parent_argv < "/proc/${PPID}/cmdline"
readonly MODE="${MODE:-preflight}"
[[ "${MODE}" == "preflight" || "${MODE}" == "backbone" ]] || {
  echo "scene audit supports only preflight or backbone" >&2
  exit 2
}
[[ ${#parent_argv[@]} -eq 4
   && "${parent_argv[0]}" == "${TRUSTED_STATIC_EXEC_PATH}"
   && "${parent_argv[1]}" == "${MODE}"
   && "${parent_argv[2]}" == "${TRUSTED_STATIC_EXEC_SHA256}"
   && "${parent_argv[3]}" == "${MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256}" ]] || {
  echo "scene-audit static command identity changed" >&2
  exit 2
}

readonly consumed_launcher_fd="/proc/$$/fd/3"
readonly consumed_launcher_sha256="$(
  /usr/bin/sha256sum "${consumed_launcher_fd}" | /usr/bin/awk '{print $1}'
)"
readonly actual_static_exec_sha256="$(
  /usr/bin/sha256sum "${TRUSTED_STATIC_EXEC_PATH}" | /usr/bin/awk '{print $1}'
)"
readonly actual_static_source_sha256="$(
  /usr/bin/sha256sum "${TRUSTED_STATIC_SOURCE_PATH}" | /usr/bin/awk '{print $1}'
)"
[[ "${consumed_launcher_sha256}" == "${MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256}"
   && "${actual_static_exec_sha256}" == "${TRUSTED_STATIC_EXEC_SHA256}"
   && "${actual_static_source_sha256}" == "${TRUSTED_STATIC_SOURCE_SHA256}"
   && "$(/usr/bin/stat -Lc '%d' "${consumed_launcher_fd}")" == "${MCLN_FPR_LAUNCHER_DEVICE}"
   && "$(/usr/bin/stat -Lc '%i' "${consumed_launcher_fd}")" == "${MCLN_FPR_LAUNCHER_INODE}" ]] || {
  echo "scene-audit trusted artifact changed" >&2
  exit 2
}
exec 3<&-

[[ "$(/usr/bin/stat -Lc '%u:%g:%a:%F' "${TRUSTED_STATIC_EXEC_PATH}")" == \
      "0:0:755:regular file"
   && "$(/usr/bin/stat -Lc '%u:%g:%a:%F' "${TRUSTED_STATIC_SOURCE_PATH}")" == \
      "0:0:644:regular file"
   && "$(/usr/bin/stat -Lc '%u:%g:%a:%F' "${TRUSTED_LAUNCHER_PATH}")" == \
      "0:0:755:regular file" ]] || {
  echo "scene-audit trust-root metadata changed" >&2
  exit 2
}
if /usr/bin/readelf -lW "${TRUSTED_STATIC_EXEC_PATH}" | \
     /usr/bin/grep -Eq '(^|[[:space:]])INTERP([[:space:]]|$)'; then
  echo "scene-audit trust executor is not static" >&2
  exit 2
fi
if [[ -n "${BASH_ENV:-}" || -n "${ENV:-}" || -n "${LD_PRELOAD:-}"
      || -n "${LD_AUDIT:-}" || -n "${LD_LIBRARY_PATH:-}"
      || -n "${PYTHONOPTIMIZE:-}" || -n "${PYTHONWARNINGS:-}"
      || -n "${PYTHONSTARTUP:-}" || -n "${PYTHONHOME:-}"
      || -n "${PYTHONUSERBASE:-}" || -n "${CDPATH:-}"
      || -n "${GLOBIGNORE:-}" ]]; then
  echo "scene-audit ambient injection variable is forbidden" >&2
  exit 2
fi
if /usr/bin/env | /usr/bin/grep -Eq \
     '^(SHELLOPTS|BASHOPTS|PS4|BASH_FUNC_)='; then
  echo "scene-audit exported Bash state is forbidden" >&2
  exit 2
fi
mapfile -t inherited_functions < <(compgen -A function || true)
if ((${#inherited_functions[@]} != 0)); then
  echo "scene-audit inherited shell functions are forbidden" >&2
  exit 2
fi
if (($# != 0)); then
  echo "usage: MODE=preflight|backbone ${TRUSTED_STATIC_EXEC_PATH} ..." >&2
  exit 2
fi

[[ -f "${RUNNER_PATH}" && ! -L "${RUNNER_PATH}" ]] || {
  echo "scene-audit reviewed runner is missing or a symlink" >&2
  exit 3
}
exec 4<"${RUNNER_PATH}"
readonly runner_fd="/proc/$$/fd/4"
readonly observed_runner_sha256="$(
  /usr/bin/sha256sum "${runner_fd}" | /usr/bin/awk '{print $1}'
)"
[[ "${observed_runner_sha256}" == "${REVIEWED_RUNNER_SHA256}"
   && "$(/usr/bin/stat -Lc '%u:%g:%a:%F' "${runner_fd}")" == \
      "0:0:644:regular file" ]] || {
  echo "scene-audit reviewed runner identity changed" >&2
  exit 3
}

cd "${SOURCE_ROOT}"
exec /usr/bin/env -i \
  HOME=/root USER=root LOGNAME=root LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PATH=/root/miniconda3/envs/bdetr/bin:/usr/local/cuda/bin:/usr/bin:/bin \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  MODE="${MODE}" \
  MCLN_FPR_TRUSTED_CLEAN_ENV=1 \
  MCLN_FPR_STATIC_EXEC_PATH="${TRUSTED_STATIC_EXEC_PATH}" \
  MCLN_FPR_STATIC_SOURCE_PATH="${TRUSTED_STATIC_SOURCE_PATH}" \
  MCLN_FPR_LAUNCHER_FD=3 \
  MCLN_FPR_STATIC_PARENT_PID="${MCLN_FPR_STATIC_PARENT_PID}" \
  MCLN_FPR_STATIC_PARENT_START_TICKS="${MCLN_FPR_STATIC_PARENT_START_TICKS}" \
  MCLN_FPR_STATIC_EXEC_SHA256="${MCLN_FPR_STATIC_EXEC_SHA256}" \
  MCLN_FPR_STATIC_SOURCE_SHA256="${MCLN_FPR_STATIC_SOURCE_SHA256}" \
  MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256="${MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256}" \
  MCLN_FPR_LAUNCHER_DEVICE="${MCLN_FPR_LAUNCHER_DEVICE}" \
  MCLN_FPR_LAUNCHER_INODE="${MCLN_FPR_LAUNCHER_INODE}" \
  MCLN_FPR_FORMAL_PGID="${MCLN_FPR_FORMAL_PGID}" \
  MCLN_DENSITY_SCENE_RUNTIME_MANIFEST_SHA256="${REVIEWED_RUNTIME_MANIFEST_SHA256}" \
  /root/miniconda3/envs/bdetr/bin/python "${runner_fd}" --mode "${MODE}"
