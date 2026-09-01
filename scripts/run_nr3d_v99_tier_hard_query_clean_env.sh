#!/bin/bash
set -euo pipefail

readonly LAUNCHER="/home/gb/new butd/butd_detr-main/MCLN-main/scripts/run_nr3d_v99_tier_hard_query_e57_e58_e62.sh"
readonly LAUNCHER_SHA256="aa6802d8d103d978ee9f969bf703d2eabec5f7d897c61f1d0d4a2a30723a5a98"
readonly BOOTSTRAP_PATH="$(/usr/bin/readlink -f "${BASH_SOURCE[0]}")"
readonly ACTUAL_BOOTSTRAP_SHA256="$(/usr/bin/sha256sum "${BOOTSTRAP_PATH}" | /usr/bin/awk '{print $1}')"

[[ "${REVIEWED_BOOTSTRAP_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "REVIEWED_BOOTSTRAP_SHA256 is required" >&2
  exit 2
}
[[ "${ACTUAL_BOOTSTRAP_SHA256}" == "${REVIEWED_BOOTSTRAP_SHA256}" ]] || {
  echo "clean-env bootstrap is not the reviewed fixed point" >&2
  exit 3
}
case "${MODE:-preflight}" in
  preflight|backbone) ;;
  *) echo "MODE must be preflight or backbone" >&2; exit 2 ;;
esac
[[ -f "${LAUNCHER}" ]] || { echo "formal launcher is missing" >&2; exit 3; }
actual_launcher_sha256="$(/usr/bin/sha256sum "${LAUNCHER}" | /usr/bin/awk '{print $1}')"
[[ "${actual_launcher_sha256}" == "${LAUNCHER_SHA256}" ]] || {
  echo "formal launcher is not the bootstrap-pinned fixed point" >&2
  exit 3
}

detected_sty=""
if [[ "${MODE:-preflight}" == "backbone" ]]; then
  detected_sty="$(/root/miniconda3/envs/bdetr/bin/python - <<'PY'
import os
import pathlib


expected_name = "mcln_nr3d_tier_hard_query_recovery2"
pid = os.getppid()
visited = set()
while pid > 1 and pid not in visited:
    visited.add(pid)
    proc = pathlib.Path("/proc") / str(pid)
    raw = (proc / "stat").read_text(encoding="utf-8")
    close_paren = raw.rfind(")")
    fields = raw[close_paren + 2:].split()
    parent = int(fields[1])
    cmdline = (proc / "cmdline").read_bytes()
    if b"SCREEN" in cmdline and expected_name.encode("utf-8") in cmdline:
        print("{}.{}".format(pid, expected_name))
        raise SystemExit(0)
    pid = parent
raise SystemExit("clean-env bootstrap is not a descendant of the formal screen")
PY
)"
fi

exec /usr/bin/env -i \
  HOME=/root USER=root LOGNAME=root LANG=C.UTF-8 LC_ALL=C.UTF-8 TERM=screen \
  PATH=/root/miniconda3/envs/bdetr/bin:/usr/local/cuda/bin:/usr/bin:/bin \
  STY="${detected_sty}" MODE="${MODE:-preflight}" \
  REVIEWED_LAUNCHER_SHA256="${LAUNCHER_SHA256}" \
  MCLN_TIER_TRUSTED_CLEAN_ENV=1 \
  TRUSTED_BOOTSTRAP_PATH="${BOOTSTRAP_PATH}" \
  TRUSTED_BOOTSTRAP_SHA256="${ACTUAL_BOOTSTRAP_SHA256}" \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /bin/bash --noprofile --norc "${LAUNCHER}"
