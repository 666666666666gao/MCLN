#!/usr/bin/env bash
set -euo pipefail

readonly TRUST_ROOT='/root/mcln_fpr_audit_trust/v5'
readonly TRUSTED_STATIC_EXEC_PATH="${TRUST_ROOT}/mcln_fpr_audit_static_exec.x86_64"
readonly TRUSTED_STATIC_SOURCE_PATH="${TRUST_ROOT}/mcln_fpr_audit_static_exec.c"
readonly TRUSTED_LAUNCHER_PATH="${TRUST_ROOT}/run_nr3d_fpr_tv_density_audit.sh"
readonly TRUSTED_STATIC_EXEC_SHA256='82e88919dcebcfbd93dd7371df174264ac92b9cf75a3c68983ba14d79e2ba466'
readonly TRUSTED_STATIC_SOURCE_SHA256='0bf6cfcfb015a91474579ba0c0f186c49c6a38695601d904d3216724cc67dcdc'
[[ "${MCLN_FPR_TRUSTED_CLEAN_ENV:-}" == '1' ]] || {
  echo 'launcher must be entered through the reviewed static executor' >&2
  exit 2
}
[[ "${MCLN_FPR_STATIC_EXEC_PATH:-}" == "${TRUSTED_STATIC_EXEC_PATH}"
   && "${MCLN_FPR_STATIC_SOURCE_PATH:-}" == "${TRUSTED_STATIC_SOURCE_PATH}"
   && "${MCLN_FPR_STATIC_EXEC_SHA256:-}" == "${TRUSTED_STATIC_EXEC_SHA256}"
   && "${MCLN_FPR_STATIC_SOURCE_SHA256:-}" == "${TRUSTED_STATIC_SOURCE_SHA256}"
   && "${MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256:-}" =~ ^[0-9a-f]{64}$
   && "${MCLN_FPR_LAUNCHER_FD:-}" == '3'
   && "${MCLN_FPR_LAUNCHER_DEVICE:-}" =~ ^[0-9]+$
   && "${MCLN_FPR_LAUNCHER_INODE:-}" =~ ^[1-9][0-9]*$
   && "${MCLN_FPR_FORMAL_PGID:-}" =~ ^[1-9][0-9]*$
   && "${MCLN_FPR_STATIC_PARENT_PID:-}" =~ ^[1-9][0-9]*$
   && "${MCLN_FPR_STATIC_PARENT_START_TICKS:-}" =~ ^[1-9][0-9]*$ ]] || {
  echo 'trusted static-executor provenance is incomplete' >&2
  exit 2
}
[[ "${MCLN_FPR_STATIC_PARENT_PID}" == "${PPID}" ]] || {
  echo 'formal launcher parent is not the reviewed static executor' >&2
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
  echo 'static-executor process identity changed' >&2
  exit 2
}
mapfile -d '' -t parent_argv < "/proc/${PPID}/cmdline"
[[ ${#parent_argv[@]} -eq 4
   && "${parent_argv[0]}" == "${TRUSTED_STATIC_EXEC_PATH}"
   && "${parent_argv[1]}" == "${MODE:-preflight}"
   && "${parent_argv[2]}" == "${TRUSTED_STATIC_EXEC_SHA256}"
   && "${parent_argv[3]}" == "${MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256}" ]] || {
  echo 'static-executor command identity changed' >&2
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
  echo 'trusted static-executor artifact changed' >&2
  exit 2
}
exec 3<&-
[[ "$(/usr/bin/stat -Lc '%u:%g:%a:%F' "${TRUSTED_STATIC_EXEC_PATH}")" \
      == '0:0:755:regular file'
   && "$(/usr/bin/stat -Lc '%u:%g:%a:%F' "${TRUSTED_STATIC_SOURCE_PATH}")" \
      == '0:0:644:regular file'
   && "$(/usr/bin/stat -Lc '%u:%g:%a:%F' "${TRUSTED_LAUNCHER_PATH}")" \
      == '0:0:755:regular file' ]] || {
  echo 'trusted static-executor owner or mode changed' >&2
  exit 2
}
readonly static_program_headers="$(
  /usr/bin/readelf -lW "${TRUSTED_STATIC_EXEC_PATH}"
)"
if /usr/bin/grep -Eq '(^|[[:space:]])INTERP([[:space:]]|$)' \
     <<<"${static_program_headers}"; then
  echo 'trusted clean-env executor must be statically linked' >&2
  exit 2
fi
if [[ -n "${BASH_ENV:-}" || -n "${ENV:-}" || -n "${LD_PRELOAD:-}"
      || -n "${LD_AUDIT:-}" || -n "${LD_LIBRARY_PATH:-}"
      || -n "${PYTHONOPTIMIZE:-}" || -n "${PYTHONWARNINGS:-}"
      || -n "${PYTHONSTARTUP:-}" || -n "${PYTHONHOME:-}"
      || -n "${PYTHONUSERBASE:-}" || -n "${CDPATH:-}"
      || -n "${GLOBIGNORE:-}" ]]; then
  echo 'ambient shell, loader, and Python injection variables are forbidden' >&2
  exit 2
fi
if /usr/bin/env | /usr/bin/grep -Eq '^(SHELLOPTS|BASHOPTS|PS4|BASH_FUNC_)='; then
  echo 'exported Bash option, debug, or function variables are forbidden' >&2
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
export PATH='/root/miniconda3/envs/bdetr/bin:/usr/local/cuda/bin:/usr/bin:/bin'
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
hash -r

terminate_formal_group() {
  trap - HUP INT TERM
  kill -TERM -- "-${MCLN_FPR_FORMAL_PGID}" 2>/dev/null || true
  exit 143
}
trap terminate_formal_group HUP INT TERM

readonly ROOT_DIR='/home/gb/new butd/butd_detr-main/MCLN-main'
readonly LAUNCHER_PATH="${TRUSTED_LAUNCHER_PATH}"
readonly PYTHON_BIN='/root/miniconda3/envs/bdetr/bin/python'
readonly DATA_ROOT='/root/autodl-tmp/DATA_ROOT'
readonly DATASET='nr3d'
readonly OUTPUT_ROOT="${DATA_ROOT}/output/network_v99_baseline_gt/nr3d"
readonly ORIGINAL_AUDIT_ROOT="${OUTPUT_ROOT}/audit/nr3d_fpr_tv_e57_e58_b100_b16x1_one_shot_v2"
readonly ORIGINAL_EXP='nr3d_fpr_tv_e57_e58_b100_b16x1_one_shot_v2'
readonly ORIGINAL_LAUNCHER_SHA256='f380dd094ee1dd71500fe895d8cecabf8ea24c8e892eac5858893b28fabf3003'
readonly ORIGINAL_STATIC_EXEC_SHA256='fd4d326cb4498a107761ba039426c5758c0809c09bc67a92ce3b427a93cbbcd7'
readonly ORIGINAL_STATIC_SOURCE_SHA256='0bf6cfcfb015a91474579ba0c0f186c49c6a38695601d904d3216724cc67dcdc'
readonly ORIGINAL_FAILURE_PROVENANCE="${ORIGINAL_AUDIT_ROOT}/consumed_provenance.json"
readonly ORIGINAL_FAILURE_PROVENANCE_SHA256='0424186a248c6a41877148ef22e626d27647fb69ee40883be04a09b7e552a41e'
readonly ORIGINAL_FAILURE_COMMAND="${ORIGINAL_AUDIT_ROOT}/formal_command.json"
readonly ORIGINAL_FAILURE_COMMAND_SHA256='2c351dd7bfecdaf861afd891b7f3a0f9ab287747ced6faa760cb780a89f1efdb'
readonly ORIGINAL_FAILURE_LOG="${ORIGINAL_AUDIT_ROOT}/runtime_output/launch.log"
readonly ORIGINAL_FAILURE_LOG_SHA256='58f88f1b46cb56961c188ec46818fe86fcddb6e2625420316431337c91836ff8'
readonly PRIOR_AUDIT_ROOT="${OUTPUT_ROOT}/audit/nr3d_fpr_tv_e57_e58_b100_b16x1_recovery_v1"
readonly PRIOR_EXP='nr3d_fpr_tv_e57_e58_b100_b16x1_recovery_v1'
readonly PRIOR_LAUNCHER_SHA256='7f4d25b7bb68a5dddb8721937789cafd7d7773a555d5f3d045d88f9dba28bce6'
readonly PRIOR_STATIC_EXEC_SHA256='196db5a6618e8f8aba71898c28dc7287ef35030d7be0d47880f2ec1951488d08'
readonly PRIOR_FAILURE_PROVENANCE="${PRIOR_AUDIT_ROOT}/consumed_provenance.json"
readonly PRIOR_FAILURE_PROVENANCE_SHA256='058a6bc8775976377e53208a1f4786106ce717a0aed2c0c7a955159265a93d63'
readonly PRIOR_FAILURE_COMMAND="${PRIOR_AUDIT_ROOT}/formal_command.json"
readonly PRIOR_FAILURE_COMMAND_SHA256='8ac3cde465bb33a72c40717a013f23ef0813df03e937b889fef5d4d4f00518d9'
readonly PRIOR_FAILURE_LOG="${PRIOR_AUDIT_ROOT}/runtime_output/launch.log"
readonly PRIOR_FAILURE_LOG_SHA256='c0ef01b2c23034cbb9fd1acc7dbccc48cfb68436d8160b1b78ee47464f248881'
readonly PRIOR_FAILURE_TRAIN_LOG="${PRIOR_AUDIT_ROOT}/runtime_output/nr3d/${PRIOR_EXP}/1788140129/log.txt"
readonly PRIOR_FAILURE_TRAIN_LOG_SHA256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
readonly SECOND_AUDIT_ROOT="${OUTPUT_ROOT}/audit/nr3d_fpr_tv_e57_e58_b100_b16x1_recovery_v2"
readonly SECOND_EXP='nr3d_fpr_tv_e57_e58_b100_b16x1_recovery_v2'
readonly SECOND_RUNTIME_TIMESTAMP='1788141845'
readonly SECOND_LAUNCHER_SHA256='78639fd8b50752ac418aee5cabca32e23f1e79da3c1e871b231ba42c083d6a57'
readonly SECOND_STATIC_EXEC_SHA256='3c96b03eb49ff243fef5acd4891ae2571888743cf3f240e8221190d64e89d1ea'
readonly SECOND_REVIEWED_CODE_MANIFEST_SHA256='590ea40219879fa73aad46f4a1d6095b6d209aafd8890ea43943e5854b9295d7'
readonly SECOND_FAILURE_PROVENANCE="${SECOND_AUDIT_ROOT}/consumed_provenance.json"
readonly SECOND_FAILURE_PROVENANCE_SHA256='0b7d492751c97212ca2e5875ea7cf294c4a46af38d902aa47b5ecfe5609e8514'
readonly SECOND_FAILURE_COMMAND="${SECOND_AUDIT_ROOT}/formal_command.json"
readonly SECOND_FAILURE_COMMAND_SHA256='d5ead64a6a94dbf8970a0035724cf58cad229337cb531288ab35dc74b02a7609'
readonly SECOND_FAILURE_LOG="${SECOND_AUDIT_ROOT}/runtime_output/launch.log"
readonly SECOND_FAILURE_LOG_SHA256='f95d850e3cb278d73b6c6775f54ede4656d03584579cbea80be67dad44eec85e'
readonly SECOND_RUN_DIR="${SECOND_AUDIT_ROOT}/runtime_output/nr3d/${SECOND_EXP}/${SECOND_RUNTIME_TIMESTAMP}"
readonly SECOND_FAILURE_CONFIG="${SECOND_RUN_DIR}/config.json"
readonly SECOND_FAILURE_CONFIG_SHA256='256b521cd22910d6c0359bf8502306fcf03a2d41cf3d1ee0f5e286eb1c5091aa'
readonly SECOND_FAILURE_TRAIN_LOG="${SECOND_RUN_DIR}/log.txt"
readonly SECOND_FAILURE_TRAIN_LOG_SHA256='9a2977039ae6c41386ff94e2adf9d48e40aadb93d716460786688fb83137bc81'
readonly SECOND_FAILURE_TENSORBOARD_TRAIN="${SECOND_RUN_DIR}/tensorboard/train/events.out.tfevents.1788141845.autodl-container-c7cb4299a4-24929f53"
readonly SECOND_FAILURE_TENSORBOARD_VAL="${SECOND_RUN_DIR}/tensorboard/val/events.out.tfevents.1788141845.autodl-container-c7cb4299a4-24929f53"
readonly SECOND_FAILURE_EMPTY_SHA256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
readonly SECOND_CODE_MANIFEST="${SECOND_AUDIT_ROOT}/consumed_snapshot/code/CODE_MANIFEST.json"
readonly SECOND_CODE_MANIFEST_SHA256='63ea6ea0509144129882198a91f7af0ff6fac6bc4a84030a2402fe1c6a100823'
readonly SECOND_INPUT_MANIFEST="${SECOND_AUDIT_ROOT}/consumed_snapshot/inputs/INPUT_MANIFEST.json"
readonly SECOND_INPUT_MANIFEST_SHA256='27cddd0b105059183fe93b7aff5bcf3ebb151582089af67449de6f6b82cfa3d7'
readonly THIRD_AUDIT_ROOT="${OUTPUT_ROOT}/audit/nr3d_fpr_tv_e57_e58_b100_b16x1_recovery_v3"
readonly THIRD_EXP='nr3d_fpr_tv_e57_e58_b100_b16x1_recovery_v3'
readonly THIRD_RUNTIME_TIMESTAMP='1788147434'
readonly THIRD_LAUNCHER_SHA256='af347fc47299c75ed27dcc8b098843f3b44ebb1cd398f4ed74dcc0546e504bad'
readonly THIRD_STATIC_EXEC_SHA256='56dbf9cf7b7299577e6bc2fee9aa5e2b183b0d1fe6c5a625951204d0ae5fc216'
readonly THIRD_REVIEWED_CODE_MANIFEST_SHA256='590ea40219879fa73aad46f4a1d6095b6d209aafd8890ea43943e5854b9295d7'
readonly THIRD_FAILURE_EVIDENCE="${OUTPUT_ROOT}/control/fpr_tv_audit/recovery_v3_collate_failure_evidence_v1.json"
readonly THIRD_FAILURE_EVIDENCE_SHA256='0e90395d3925858a9931dc2ef5e6bc4759e9ebb288e11c921708ba3b55887bd7'
readonly THIRD_FIRST_BATCH_REPLAY_SCRIPT="${OUTPUT_ROOT}/control/fpr_tv_audit/replay_recovery_v3_first_batch_v1.py"
readonly THIRD_FIRST_BATCH_REPLAY_SCRIPT_SHA256='6733b921c2bb6fe41a3781b08562df4252315a7ec693dee366058a90eeb78957'
readonly THIRD_FIRST_BATCH_REPLAY_RECEIPT="${OUTPUT_ROOT}/control/fpr_tv_audit/recovery_v3_first_batch_replay_receipt_v1.json"
readonly THIRD_FIRST_BATCH_REPLAY_RECEIPT_SHA256='418d3381c5a6348b3a469c6457bcff7a148ce36ec3b15ea7910cadfcc12596a5'
readonly THIRD_FAILURE_PROVENANCE="${THIRD_AUDIT_ROOT}/consumed_provenance.json"
readonly THIRD_FAILURE_PROVENANCE_SHA256='7b9f6e05b9ad80e809270336ef0964a6d3983372e5d3c5bd729c1cf77c914180'
readonly THIRD_FAILURE_COMMAND="${THIRD_AUDIT_ROOT}/formal_command.json"
readonly THIRD_FAILURE_COMMAND_SHA256='54c5ef1697f0ab7d7b69892504d0421439d4ee330422c566777c5510dbb739a6'
readonly THIRD_FAILURE_LOG="${THIRD_AUDIT_ROOT}/runtime_output/launch.log"
readonly THIRD_FAILURE_LOG_SHA256='8c39ec7271051ed8f8149735cee328af49da6b20cccbdca0c0d64eebf6f453c1'
readonly THIRD_RUN_DIR="${THIRD_AUDIT_ROOT}/runtime_output/nr3d/${THIRD_EXP}/${THIRD_RUNTIME_TIMESTAMP}"
readonly THIRD_FAILURE_CONFIG="${THIRD_RUN_DIR}/config.json"
readonly THIRD_FAILURE_CONFIG_SHA256='9c04246f7de1a0314def0feb8520338a55c60908cb7558d2fb9e3b45f0af291b'
readonly THIRD_FAILURE_TRAIN_LOG="${THIRD_RUN_DIR}/log.txt"
readonly THIRD_FAILURE_TRAIN_LOG_SHA256='9ac7298628e6c97cb1526df4a75662d52aa35caa80be405ea040fce4aa47cce8'
readonly THIRD_FAILURE_TENSORBOARD_TRAIN="${THIRD_RUN_DIR}/tensorboard/train/events.out.tfevents.1788147434.autodl-container-c7cb4299a4-24929f53"
readonly THIRD_FAILURE_TENSORBOARD_TRAIN_SHA256='78a12bc231c24294544ecfb9382a15002fa39dfbe9a08498fc3d2c6ef6adb7f3'
readonly THIRD_FAILURE_TENSORBOARD_VAL="${THIRD_RUN_DIR}/tensorboard/val/events.out.tfevents.1788147434.autodl-container-c7cb4299a4-24929f53"
readonly THIRD_FAILURE_TENSORBOARD_VAL_SHA256='6708c1bbb32f99ef71021fb7b2bcb61ceb300fb9979e3a7f4ade58dba684eb7c'
readonly THIRD_CODE_MANIFEST="${THIRD_AUDIT_ROOT}/consumed_snapshot/code/CODE_MANIFEST.json"
readonly THIRD_CODE_MANIFEST_SHA256='63ea6ea0509144129882198a91f7af0ff6fac6bc4a84030a2402fe1c6a100823'
readonly THIRD_INPUT_MANIFEST="${THIRD_AUDIT_ROOT}/consumed_snapshot/inputs/INPUT_MANIFEST.json"
readonly THIRD_INPUT_MANIFEST_SHA256='7eac4a1686916780a9c7171522c1dbefb03653f95849ae367a6bf33836ec46b5'
readonly AUDIT_ROOT="${OUTPUT_ROOT}/audit/nr3d_fpr_tv_e57_e58_b100_b16x1_recovery_v4"
readonly EXP='nr3d_fpr_tv_e57_e58_b100_b16x1_recovery_v4'
readonly CHECKPOINT="${OUTPUT_ROOT}/control/official_rec_monitor/official_best_rec025_epoch_57_0p56652741.pth"
readonly CHECKPOINT_SHA256='76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1'
readonly GROUPFREE_CHECKPOINT="${DATA_ROOT}/gf_detector_l6o256.pth"
readonly GROUPFREE_SHA256='9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2'
readonly DATA_MANIFEST="${OUTPUT_ROOT}/control/fpr_tv_audit/nr3d_train_input_manifest_v1.json"
readonly DATA_MANIFEST_SHA256='ce0e287856363fce2c6cb119617798ff98470aaba331d8239fbc32ffcdc93259'
readonly REVIEWED_CODE_MANIFEST="${OUTPUT_ROOT}/control/fpr_tv_audit/runtime_code_manifest_fpr_collate_v3.json"
readonly REVIEWED_CODE_MANIFEST_SHA256='d96f120735159c6d6c93a4677dd4c6f14ffa48789f9267132aec2ad9e1bd6e25'
readonly LANDLOCK_EXECUTOR='/root/mcln_landlock_snapshot_exec.py'
readonly LANDLOCK_EXECUTOR_SHA256='ae953c5985549f7c8e47818764237c1db30dd12783367498445879d18a82a28c'
readonly SNAPSHOT_OWNER_UID=65532
readonly SNAPSHOT_OWNER_GID=65532
readonly BATCH_SIZE=16
readonly AUDIT_BATCHES=100
readonly AUDIT_EPOCH=58
readonly LOCAL_RANK=0
readonly MASTER_PORT=5317
readonly MIN_FREE_GB=7

# These are preregistered density/safety gates, not decision thresholds.  A
# passing audit authorizes only a later scene-disjoint audit, never long train.
readonly MIN_DEPLOYABLE_ROW_RATIO='0.50'
readonly MIN_DETECTOR_CANDIDATE_RATIO='0.002'
readonly MIN_RELIABLE_ROW_RATIO='0.02'
readonly MIN_FEASIBLE_CANDIDATE_RATIO='0.01'
readonly MIN_POSITIVE_ROW_RATIO='0.002'
readonly MIN_CANDIDATE_POSITIVE_RATIO='0.0005'

readonly REQUIRED_TRAIN_ENTRY_SHA256='ef66b0f1775eba71e0fcba6db9465bc4feeeb6b4c3536c97bb34b63fb5887534'
readonly REQUIRED_MAIN_UTILS_SHA256='fded0db08e3b7b7d4c8fc9c3cc8ffb6ed92ecb42d0884949472378a6cde806e3'
readonly REQUIRED_LOSSES_SHA256='474a0d1356a61ef1e7ecdc083273076adc4514643ac1974d03042280ebe09ee4'
readonly REQUIRED_MCLN_SHA256='a9301a5fc9bac3b40e4450350a8d2eb4ba11c4763e734c5a09723a5232474db4'
readonly REQUIRED_FPR_SHA256='bb797968c027f55722e34803617b8ae192919e2d92bc3c166727c0841faaae61'
readonly REQUIRED_FILTER_SHA256='49a43b89a1ff129d09dcbdf0f6b61ff817aca50fb2c0edcb49072c60ded1a7e7'
readonly REQUIRED_SELECTOR_SHA256='61211cea91de4c3f3c11e44bc5ff035711307f3381d5a4069d4ab7842bee17dc'
readonly REQUIRED_SOURCE_ADAPTER_SHA256='dc32c6adfde80af0449b28415a5a4d9ffcb9a5115b8894ab0a2f7c6ab9b11fbb'
readonly REQUIRED_REC_ADAPTER_SHA256='dfc5afaa6ca4feabc67417707660f4f881594f9fad11663a14f56b9be26c10a3'
readonly REQUIRED_STRUCTURED_SLOTS_SHA256='78f5c2e3a1e794ebf8876f24126c67fbb0c404707d065f55847ea7d2b2ef3281'
readonly REQUIRED_SACR_HEAD_SHA256='1b35e0c1cbb3afe0b543e895ca3614fbff97558df6d4666c90c3e9fd3433a93d'
readonly REQUIRED_STRUCTURED_SHA256='de8f32e3afc6a5c198a87f7a0b74838c073d7fe1b01093da6791b461f1d3a716'
readonly REQUIRED_GROUPS_SHA256='fd7de7565600645ac82b7d0812fe433b04eb8ee99702c5494b2030247a09d738'
readonly REQUIRED_DATASET_SHA256='800bac2caf9b7a319bdc200f60386000e4e374a559d9581113a8eb57d525f9f0'
readonly REQUIRED_TENSORBOARD_SHA256='25c0709e94010c53224ad97f946f24952ae34bdf81d88306e6c51ad4923a89b5'

MODE="${MODE:-preflight}"
readonly MODE
case "${MODE}" in
  preflight|backbone) ;;
  *) echo 'audit-only launcher supports MODE=preflight or MODE=backbone' >&2; exit 2 ;;
esac
if (($# != 0)); then
  echo 'formal launcher accepts no argv; invoke the reviewed static executor' >&2
  exit 2
fi
readonly EXPECTED_LAUNCHER_SHA256="${MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256:-}"
if [[ ! "${EXPECTED_LAUNCHER_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo 'MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256 must be the reviewed launcher SHA' >&2
  exit 2
fi

cd "${ROOT_DIR}"
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

verify_no_inheritable_descriptors() {
  local descriptor_path descriptor flags
  local -a unsafe=()
  for descriptor_path in /proc/$$/fd/[3-9] /proc/$$/fd/[1-9][0-9]*; do
    [[ -e "${descriptor_path}" ]] || continue
    descriptor="${descriptor_path##*/}"
    flags="$(/usr/bin/awk '$1 == "flags:" {print $2}' \
      "/proc/$$/fdinfo/${descriptor}")"
    [[ "${flags}" =~ ^[0-7]+$ ]] || {
      echo "cannot audit descriptor ${descriptor}" >&2
      exit 8
    }
    if (( (8#${flags} & 8#2000000) == 0 )); then
      unsafe+=("${descriptor}")
    fi
  done
  if ((${#unsafe[@]} != 0)); then
    echo "non-CLOEXEC descriptors would escape Landlock: ${unsafe[*]}" >&2
    exit 8
  fi
}

verify_fixed_code_root() {
  local root="$1"
  require_sha256 "${root}/train_dist_mod.py" "${REQUIRED_TRAIN_ENTRY_SHA256}" 'training entrypoint'
  require_sha256 "${root}/main_utils.py" "${REQUIRED_MAIN_UTILS_SHA256}" 'main_utils'
  require_sha256 "${root}/models/losses.py" "${REQUIRED_LOSSES_SHA256}" 'loss implementation'
  require_sha256 "${root}/models/mcln.py" "${REQUIRED_MCLN_SHA256}" 'MCLN implementation'
  require_sha256 "${root}/models/parent_relative_text_verifier.py" "${REQUIRED_FPR_SHA256}" 'FPR-TV implementation'
  require_sha256 "${root}/models/rec_evaluator_filter.py" "${REQUIRED_FILTER_SHA256}" 'formal detector filter'
  require_sha256 "${root}/models/source_choice_selector.py" "${REQUIRED_SELECTOR_SHA256}" 'V99 selector'
  require_sha256 "${root}/models/source_choice_adapter.py" "${REQUIRED_SOURCE_ADAPTER_SHA256}" 'source adapter'
  require_sha256 "${root}/models/rec_candidate_adapter.py" "${REQUIRED_REC_ADAPTER_SHA256}" 'REC adapter'
  require_sha256 "${root}/models/structured_slots.py" "${REQUIRED_STRUCTURED_SLOTS_SHA256}" 'structured slots'
  require_sha256 "${root}/models/sacr_head.py" "${REQUIRED_SACR_HEAD_SHA256}" 'SACR head'
  require_sha256 "${root}/models/structured_source.py" "${REQUIRED_STRUCTURED_SHA256}" 'structured source builder'
  require_sha256 "${root}/models/mcln_training_groups.py" "${REQUIRED_GROUPS_SHA256}" 'optimizer grouping'
  require_sha256 "${root}/src/joint_det_dataset.py" "${REQUIRED_DATASET_SHA256}" 'joint dataset'
  require_sha256 "${root}/utils/record_tensorboard.py" "${REQUIRED_TENSORBOARD_SHA256}" 'TensorBoard output scoping'
}

verify_fixed_inputs() {
  require_sha256 "${LAUNCHER_PATH}" "${EXPECTED_LAUNCHER_SHA256}" 'reviewed audit launcher'
  verify_fixed_code_root "${ROOT_DIR}"
  require_sha256 "${CHECKPOINT}" "${CHECKPOINT_SHA256}" 'protected Nr3D E57'
  require_sha256 "${GROUPFREE_CHECKPOINT}" "${GROUPFREE_SHA256}" 'GroupFree checkpoint'
  require_sha256 "${DATA_MANIFEST}" "${DATA_MANIFEST_SHA256}" 'Nr3D data manifest'
  require_sha256 "${REVIEWED_CODE_MANIFEST}" "${REVIEWED_CODE_MANIFEST_SHA256}" 'reviewed runtime-code manifest'
  require_sha256 "${LANDLOCK_EXECUTOR}" "${LANDLOCK_EXECUTOR_SHA256}" 'Landlock executor'
  require_sha256 "${TRUSTED_STATIC_EXEC_PATH}" "${TRUSTED_STATIC_EXEC_SHA256}" \
    'trusted static clean-env executor'
  require_sha256 "${TRUSTED_STATIC_SOURCE_PATH}" "${TRUSTED_STATIC_SOURCE_SHA256}" \
    'trusted static clean-env source'
}

verify_dataset_manifest() {
  local manifest_path="$1" expected_sha="$2"
  DATA_ROOT_ENV="${DATA_ROOT}" DATA_MANIFEST_ENV="${manifest_path}" \
  DATA_MANIFEST_SHA_ENV="${expected_sha}" "${PYTHON_BIN}" - <<'PY'
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
    "val_v3scans.pkl",
    "refer_it_3d/nr3d.csv",
    "roberta-base",
    "superpoints/train",
    "superpoints/val",
    "group_free_pred_bboxes/group_free_pred_bboxes_train",
]

if os.path.realpath(root) != root or os.path.islink(root):
    raise SystemExit("DATA_ROOT must be a real canonical directory")

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
if manifest.get("schema") != "mcln-nr3d-fpr-tv-audit-data-manifest-v1":
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
                candidate = os.path.join(current, name)
                if os.path.islink(candidate):
                    raise SystemExit("dataset directory symlink: " + candidate)
            current_paths.extend(
                os.path.join(current, name) for name in files
            )
    else:
        current_paths.append(source)

rows = manifest.get("files")
if not isinstance(rows, list):
    raise SystemExit("dataset manifest lacks files")
expected_paths = [row.get("path") for row in rows]
actual_paths = [os.path.relpath(path, root) for path in current_paths]
if actual_paths != expected_paths:
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

verify_reviewed_code_manifest() {
  local source_root="$1" manifest_path="$2" expected_sha="$3"
  SOURCE_ROOT_ENV="${source_root}" REVIEWED_CODE_MANIFEST_ENV="${manifest_path}" \
  REVIEWED_CODE_MANIFEST_SHA_ENV="${expected_sha}" "${PYTHON_BIN}" - <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import stat

root = os.environ["SOURCE_ROOT_ENV"]
manifest_path = os.environ["REVIEWED_CODE_MANIFEST_ENV"]
expected_sha = os.environ["REVIEWED_CODE_MANIFEST_SHA_ENV"]

if os.path.realpath(root) != root or os.path.islink(root):
    raise SystemExit("runtime code root must be real and canonical")

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

with open(manifest_path, "rb") as handle:
    raw = handle.read()
if hashlib.sha256(raw).hexdigest() != expected_sha:
    raise SystemExit("reviewed runtime-code manifest SHA changed")
manifest = json.loads(raw.decode("utf-8"))
if manifest.get("schema") != "mcln-fpr-tv-reviewed-runtime-code-v1":
    raise SystemExit("reviewed runtime-code manifest schema changed")
if manifest.get("source_root") != root:
    raise SystemExit("reviewed runtime-code root changed")
files = manifest.get("files")
if not isinstance(files, dict) or not files:
    raise SystemExit("reviewed runtime-code manifest lacks files")
if manifest.get("file_count") != len(files):
    raise SystemExit("reviewed runtime-code file count changed")
if manifest.get("total_size") != sum(
        record.get("size", -1) for record in files.values()):
    raise SystemExit("reviewed runtime-code total size changed")
for relative, record in sorted(files.items()):
    components = relative.split("/") if isinstance(relative, str) else []
    if (
            not components
            or os.path.isabs(relative)
            or any(component in ("", ".", "..") for component in components)):
        raise SystemExit("unsafe runtime-code path: {!r}".format(relative))
    current = root
    for component in components:
        current = os.path.join(current, component)
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise SystemExit("runtime-code path contains symlink: " + current)
    info = os.lstat(current)
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit("runtime-code source is not regular: " + current)
    if int(info.st_size) != record.get("size"):
        raise SystemExit("runtime-code size changed: " + relative)
    if "{:04o}".format(stat.S_IMODE(info.st_mode)) != record.get("mode"):
        raise SystemExit("runtime-code mode changed: " + relative)
    if sha256_file(current) != record.get("sha256"):
        raise SystemExit("runtime-code SHA changed: " + relative)
print("reviewed_runtime_code_verified={} files={} bytes={}".format(
    expected_sha, len(files), manifest["total_size"]
))
PY
}

verify_original_failed_startup() {
  if screen -ls | grep -Fq '.mcln_fpr_tv_audit_100b'; then
    echo 'original failed-audit screen is unexpectedly alive' >&2
    exit 6
  fi
  ORIGINAL_ROOT_ENV="${ORIGINAL_AUDIT_ROOT}" \
  ORIGINAL_EXP_ENV="${ORIGINAL_EXP}" \
  ORIGINAL_PROVENANCE_ENV="${ORIGINAL_FAILURE_PROVENANCE}" \
  ORIGINAL_PROVENANCE_SHA_ENV="${ORIGINAL_FAILURE_PROVENANCE_SHA256}" \
  ORIGINAL_COMMAND_ENV="${ORIGINAL_FAILURE_COMMAND}" \
  ORIGINAL_COMMAND_SHA_ENV="${ORIGINAL_FAILURE_COMMAND_SHA256}" \
  ORIGINAL_LOG_ENV="${ORIGINAL_FAILURE_LOG}" \
  ORIGINAL_LOG_SHA_ENV="${ORIGINAL_FAILURE_LOG_SHA256}" \
  ORIGINAL_LAUNCHER_SHA_ENV="${ORIGINAL_LAUNCHER_SHA256}" \
  ORIGINAL_STATIC_EXEC_SHA_ENV="${ORIGINAL_STATIC_EXEC_SHA256}" \
  ORIGINAL_STATIC_SOURCE_SHA_ENV="${ORIGINAL_STATIC_SOURCE_SHA256}" \
  "${PYTHON_BIN}" - <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import pathlib
import stat


root = pathlib.Path(os.environ["ORIGINAL_ROOT_ENV"])
original_exp = os.environ["ORIGINAL_EXP_ENV"]
expected_files = {
    "consumed_provenance.json": {
        "path": os.environ["ORIGINAL_PROVENANCE_ENV"],
        "sha256": os.environ["ORIGINAL_PROVENANCE_SHA_ENV"],
        "size": 3517,
        "mode": 0o444,
    },
    "formal_command.json": {
        "path": os.environ["ORIGINAL_COMMAND_ENV"],
        "sha256": os.environ["ORIGINAL_COMMAND_SHA_ENV"],
        "size": 6857,
        "mode": 0o444,
    },
    "runtime_output/launch.log": {
        "path": os.environ["ORIGINAL_LOG_ENV"],
        "sha256": os.environ["ORIGINAL_LOG_SHA_ENV"],
        "size": 6422,
        "mode": 0o600,
    },
}
expected_directories = {
    ".",
    "runtime_output",
    "runtime_output/runtime_home",
    "runtime_output/runtime_home/.nv",
    "runtime_output/runtime_home/.nv/ComputeCache",
    "runtime_output/runtime_home/hf",
    "runtime_output/runtime_home/torch",
    "runtime_output/runtime_home/xdg",
}


def read_regular(path, expected_sha):
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("failed-audit evidence is not regular: " + str(path))
        chunks = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise SystemExit("failed-audit evidence changed while reading: " + str(path))
    if digest.hexdigest() != expected_sha:
        raise SystemExit("failed-audit evidence SHA changed: " + str(path))
    return b"".join(chunks), before


if root.is_symlink() or root.resolve() != root or not root.is_dir():
    raise SystemExit("original failed-audit root identity changed")
root_info = root.stat()
if (
        root_info.st_uid != 0
        or root_info.st_gid != 0
        or stat.S_IMODE(root_info.st_mode) != 0o700):
    raise SystemExit("original failed-audit root owner changed")
snapshot_root = root / "consumed_snapshot"
if snapshot_root.is_symlink() or not snapshot_root.is_dir():
    raise SystemExit("original failed-audit snapshot is missing or unsafe")

actual_files = {}
actual_directories = set()
for current, directory_names, file_names in os.walk(str(root)):
    current_path = pathlib.Path(current)
    relative_directory = current_path.relative_to(root).as_posix()
    relative_directory = "." if relative_directory == "." else relative_directory
    if relative_directory == "." and "consumed_snapshot" in directory_names:
        directory_names.remove("consumed_snapshot")
    current_info = current_path.stat()
    if (
            current_info.st_uid != 0
            or current_info.st_gid != 0
            or stat.S_IMODE(current_info.st_mode) != 0o700):
        raise SystemExit("failed-audit output directory metadata changed")
    for directory_name in directory_names:
        candidate = current_path / directory_name
        if candidate.is_symlink():
            raise SystemExit("failed-audit output contains a directory symlink")
    actual_directories.add(relative_directory)
    for file_name in file_names:
        candidate = current_path / file_name
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise SystemExit("failed-audit output contains a file symlink")
        actual_files[relative] = candidate
if set(actual_files) != set(expected_files):
    raise SystemExit("failed-audit non-snapshot file inventory changed")
if actual_directories != expected_directories:
    raise SystemExit("failed-audit non-snapshot directory inventory changed")

raw_by_name = {}
for relative, expected in expected_files.items():
    candidate = actual_files[relative]
    if str(candidate) != expected["path"]:
        raise SystemExit("failed-audit evidence path changed: " + relative)
    raw, info = read_regular(candidate, expected["sha256"])
    if (
            info.st_size != expected["size"]
            or stat.S_IMODE(info.st_mode) != expected["mode"]
            or info.st_uid != 0
            or info.st_gid != 0):
        raise SystemExit("failed-audit evidence metadata changed: " + relative)
    raw_by_name[relative] = raw

command = json.loads(raw_by_name["formal_command.json"].decode("utf-8"))
provenance = json.loads(raw_by_name["consumed_provenance.json"].decode("utf-8"))
log_text = raw_by_name["runtime_output/launch.log"].decode("utf-8")
if command.get("schema") != "mcln-fpr-tv-density-audit-command-v2":
    raise SystemExit("failed formal-command schema changed")
argv = command.get("argv")
if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
    raise SystemExit("failed formal-command argv changed")
if argv.count("LOCAL_RANK=0") != 1:
    raise SystemExit("failed formal command did not pin LOCAL_RANK=0")
if "--local_rank" in argv or "--local-rank" in argv:
    raise SystemExit("failed formal command unexpectedly pinned argparse local rank")
for flag, expected in (
        ("--exp", original_exp),
        ("--max_train_batches", "100"),
        ("--gradient_accumulation_steps", "1"),
        ("--start_epoch", "58"),
        ("--max_epoch", "58")):
    if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != expected:
        raise SystemExit("failed formal-command contract changed: " + flag)
if "--eval" in argv:
    raise SystemExit("failed formal command unexpectedly requested evaluation")
required_log_fragments = (
    "Traceback (most recent call last):",
    "torch.cuda.set_device(opt.local_rank)",
    "RuntimeError: CUDA error: invalid device ordinal",
)
if any(fragment not in log_text for fragment in required_log_fragments):
    raise SystemExit("failed startup traceback changed")
for forbidden in (
        "Epoch: [58]", "train_audit_receipt_epoch_58.json",
        "audit_complete=true", "bounded_train_audit"):
    if forbidden in log_text:
        raise SystemExit("failed startup unexpectedly reached training: " + forbidden)
if provenance.get("schema") != "mcln-fpr-tv-density-audit-consumed-provenance-v3":
    raise SystemExit("failed consumed-provenance schema changed")
inputs = provenance.get("inputs")
if not isinstance(inputs, dict):
    raise SystemExit("failed consumed provenance lacks inputs")
expected_input_hashes = {
    "launcher": os.environ["ORIGINAL_LAUNCHER_SHA_ENV"],
    "static_clean_env_executor": os.environ["ORIGINAL_STATIC_EXEC_SHA_ENV"],
    "static_clean_env_source": os.environ["ORIGINAL_STATIC_SOURCE_SHA_ENV"],
    "command": os.environ["ORIGINAL_COMMAND_SHA_ENV"],
}
for label, expected_sha in expected_input_hashes.items():
    item = inputs.get(label)
    if not isinstance(item, dict) or item.get("sha256") != expected_sha:
        raise SystemExit("failed consumed provenance changed: " + label)
if provenance.get("code", {}).get("manifest_sha256") != (
        "4d157ba04b390d5a753f0973ce9c16c7b45b6ce91d54e458fda5730bf8d6abd6"):
    raise SystemExit("failed code-snapshot manifest changed")
if provenance.get("input_snapshot", {}).get("manifest_sha256") != (
        "e93de0d573af490c9d663bdc67f9e3ac8733ad79b00c55320e3b46d363e3089b"):
    raise SystemExit("failed input-snapshot manifest changed")
for section, expected_path, expected_sha in (
        (
            "code",
            snapshot_root / "code" / "CODE_MANIFEST.json",
            "4d157ba04b390d5a753f0973ce9c16c7b45b6ce91d54e458fda5730bf8d6abd6",
        ),
        (
            "input_snapshot",
            snapshot_root / "inputs" / "INPUT_MANIFEST.json",
            "e93de0d573af490c9d663bdc67f9e3ac8733ad79b00c55320e3b46d363e3089b",
        )):
    if provenance.get(section, {}).get("manifest_path") != str(expected_path):
        raise SystemExit("failed snapshot manifest path changed: " + section)
    read_regular(expected_path, expected_sha)

needle = original_exp.encode("utf-8")
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        command_line = (entry / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if needle in command_line:
        raise SystemExit("original failed-audit process is still alive: " + entry.name)
print("original_failed_startup_verified=zero_batches_zero_receipts_zero_weights")
PY
}

verify_original_failed_startup

verify_prior_failed_startup() {
  if screen -ls | grep -Fq '.mcln_fpr_tv_audit_recovery_100b'; then
    echo 'prior recovery-audit screen is unexpectedly alive' >&2
    exit 6
  fi
  PRIOR_ROOT_ENV="${PRIOR_AUDIT_ROOT}" \
  PRIOR_EXP_ENV="${PRIOR_EXP}" \
  PRIOR_PROVENANCE_ENV="${PRIOR_FAILURE_PROVENANCE}" \
  PRIOR_PROVENANCE_SHA_ENV="${PRIOR_FAILURE_PROVENANCE_SHA256}" \
  PRIOR_COMMAND_ENV="${PRIOR_FAILURE_COMMAND}" \
  PRIOR_COMMAND_SHA_ENV="${PRIOR_FAILURE_COMMAND_SHA256}" \
  PRIOR_LOG_ENV="${PRIOR_FAILURE_LOG}" \
  PRIOR_LOG_SHA_ENV="${PRIOR_FAILURE_LOG_SHA256}" \
  PRIOR_TRAIN_LOG_ENV="${PRIOR_FAILURE_TRAIN_LOG}" \
  PRIOR_TRAIN_LOG_SHA_ENV="${PRIOR_FAILURE_TRAIN_LOG_SHA256}" \
  PRIOR_LAUNCHER_SHA_ENV="${PRIOR_LAUNCHER_SHA256}" \
  PRIOR_STATIC_EXEC_SHA_ENV="${PRIOR_STATIC_EXEC_SHA256}" \
  STATIC_SOURCE_SHA_ENV="${ORIGINAL_STATIC_SOURCE_SHA256}" \
  ORIGINAL_PROVENANCE_SHA_ENV="${ORIGINAL_FAILURE_PROVENANCE_SHA256}" \
  ORIGINAL_COMMAND_SHA_ENV="${ORIGINAL_FAILURE_COMMAND_SHA256}" \
  ORIGINAL_LOG_SHA_ENV="${ORIGINAL_FAILURE_LOG_SHA256}" \
  "${PYTHON_BIN}" - <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import pathlib
import stat


root = pathlib.Path(os.environ["PRIOR_ROOT_ENV"])
prior_exp = os.environ["PRIOR_EXP_ENV"]
train_log_relative = (
    "runtime_output/nr3d/{}/1788140129/log.txt".format(prior_exp)
)
expected_files = {
    "consumed_provenance.json": {
        "path": os.environ["PRIOR_PROVENANCE_ENV"],
        "sha256": os.environ["PRIOR_PROVENANCE_SHA_ENV"],
        "size": 4474,
        "mode": 0o444,
    },
    "formal_command.json": {
        "path": os.environ["PRIOR_COMMAND_ENV"],
        "sha256": os.environ["PRIOR_COMMAND_SHA_ENV"],
        "size": 6886,
        "mode": 0o444,
    },
    "runtime_output/launch.log": {
        "path": os.environ["PRIOR_LOG_ENV"],
        "sha256": os.environ["PRIOR_LOG_SHA_ENV"],
        "size": 8318,
        "mode": 0o600,
    },
    train_log_relative: {
        "path": os.environ["PRIOR_TRAIN_LOG_ENV"],
        "sha256": os.environ["PRIOR_TRAIN_LOG_SHA_ENV"],
        "size": 0,
        "mode": 0o600,
    },
}
expected_directories = {
    ".",
    "runtime_output",
    "runtime_output/nr3d",
    "runtime_output/nr3d/{}".format(prior_exp),
    "runtime_output/nr3d/{}/1788140129".format(prior_exp),
    "runtime_output/runtime_home",
    "runtime_output/runtime_home/.nv",
    "runtime_output/runtime_home/.nv/ComputeCache",
    "runtime_output/runtime_home/hf",
    "runtime_output/runtime_home/torch",
    "runtime_output/runtime_home/xdg",
}


def read_regular(path, expected_sha):
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("prior failure evidence is not regular: " + str(path))
        chunks = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise SystemExit("prior failure evidence changed while reading: " + str(path))
    if digest.hexdigest() != expected_sha:
        raise SystemExit("prior failure evidence SHA changed: " + str(path))
    return b"".join(chunks), before


if root.is_symlink() or root.resolve() != root or not root.is_dir():
    raise SystemExit("prior failure root identity changed")
root_info = root.stat()
if (
        root_info.st_uid != 0
        or root_info.st_gid != 0
        or stat.S_IMODE(root_info.st_mode) != 0o700):
    raise SystemExit("prior failure root owner changed")
snapshot_root = root / "consumed_snapshot"
if snapshot_root.is_symlink() or not snapshot_root.is_dir():
    raise SystemExit("prior failure snapshot is missing or unsafe")

actual_files = {}
actual_directories = set()
for current, directory_names, file_names in os.walk(str(root)):
    current_path = pathlib.Path(current)
    relative_directory = current_path.relative_to(root).as_posix()
    relative_directory = "." if relative_directory == "." else relative_directory
    if relative_directory == "." and "consumed_snapshot" in directory_names:
        directory_names.remove("consumed_snapshot")
    current_info = current_path.stat()
    if (
            current_info.st_uid != 0
            or current_info.st_gid != 0
            or stat.S_IMODE(current_info.st_mode) != 0o700):
        raise SystemExit("prior failure output directory metadata changed")
    for directory_name in directory_names:
        if (current_path / directory_name).is_symlink():
            raise SystemExit("prior failure output contains a directory symlink")
    actual_directories.add(relative_directory)
    for file_name in file_names:
        candidate = current_path / file_name
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise SystemExit("prior failure output contains a file symlink")
        actual_files[relative] = candidate
if set(actual_files) != set(expected_files):
    raise SystemExit("prior failure non-snapshot file inventory changed")
if actual_directories != expected_directories:
    raise SystemExit("prior failure non-snapshot directory inventory changed")

raw_by_name = {}
for relative, expected in expected_files.items():
    candidate = actual_files[relative]
    if str(candidate) != expected["path"]:
        raise SystemExit("prior failure evidence path changed: " + relative)
    raw, info = read_regular(candidate, expected["sha256"])
    if (
            info.st_size != expected["size"]
            or stat.S_IMODE(info.st_mode) != expected["mode"]
            or info.st_uid != 0
            or info.st_gid != 0):
        raise SystemExit("prior failure evidence metadata changed: " + relative)
    raw_by_name[relative] = raw

command = json.loads(raw_by_name["formal_command.json"].decode("utf-8"))
provenance = json.loads(raw_by_name["consumed_provenance.json"].decode("utf-8"))
log_text = raw_by_name["runtime_output/launch.log"].decode("utf-8")
if raw_by_name[train_log_relative] != b"":
    raise SystemExit("prior failure unexpectedly wrote a training log")
if command.get("schema") != "mcln-fpr-tv-density-audit-command-v3":
    raise SystemExit("prior formal-command schema changed")
argv = command.get("argv")
if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
    raise SystemExit("prior formal-command argv changed")
if argv.count("LOCAL_RANK=0") != 1:
    raise SystemExit("prior formal command did not pin LOCAL_RANK=0")
if argv.count("--local_rank") != 1:
    raise SystemExit("prior formal command local-rank flag changed")
local_rank_index = argv.index("--local_rank")
if local_rank_index + 1 >= len(argv) or argv[local_rank_index + 1] != "0":
    raise SystemExit("prior formal command did not use local rank zero")
for flag, expected in (
        ("--exp", prior_exp),
        ("--max_train_batches", "100"),
        ("--gradient_accumulation_steps", "1"),
        ("--start_epoch", "58"),
        ("--max_epoch", "58")):
    if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != expected:
        raise SystemExit("prior formal-command contract changed: " + flag)
if "--eval" in argv:
    raise SystemExit("prior formal command unexpectedly requested evaluation")
required_log_fragments = (
    "snapshot_verification=pass",
    "SummaryWriter(os.path.join('tensorboard_output/', \"tensorboard/train\"))",
    "PermissionError: [Errno 13] Permission denied: 'tensorboard_output'",
)
if any(fragment not in log_text for fragment in required_log_fragments):
    raise SystemExit("prior startup traceback changed")
for forbidden in (
        "Epoch: [58]", "train_audit_receipt_epoch_58.json",
        "audit_complete=true", "bounded_train_audit"):
    if forbidden in log_text:
        raise SystemExit("prior startup unexpectedly reached training: " + forbidden)
if provenance.get("schema") != "mcln-fpr-tv-density-audit-consumed-provenance-v4":
    raise SystemExit("prior consumed-provenance schema changed")
inputs = provenance.get("inputs")
if not isinstance(inputs, dict):
    raise SystemExit("prior consumed provenance lacks inputs")
expected_input_hashes = {
    "launcher": os.environ["PRIOR_LAUNCHER_SHA_ENV"],
    "static_clean_env_executor": os.environ["PRIOR_STATIC_EXEC_SHA_ENV"],
    "static_clean_env_source": os.environ["STATIC_SOURCE_SHA_ENV"],
    "command": os.environ["PRIOR_COMMAND_SHA_ENV"],
    "original_failed_consumed_provenance": os.environ[
        "ORIGINAL_PROVENANCE_SHA_ENV"
    ],
    "original_failed_formal_command": os.environ["ORIGINAL_COMMAND_SHA_ENV"],
    "original_failed_launch_log": os.environ["ORIGINAL_LOG_SHA_ENV"],
}
for label, expected_sha in expected_input_hashes.items():
    item = inputs.get(label)
    if not isinstance(item, dict) or item.get("sha256") != expected_sha:
        raise SystemExit("prior consumed provenance changed: " + label)
if provenance.get("code", {}).get("manifest_sha256") != (
        "4d157ba04b390d5a753f0973ce9c16c7b45b6ce91d54e458fda5730bf8d6abd6"):
    raise SystemExit("prior code-snapshot manifest changed")
if provenance.get("input_snapshot", {}).get("manifest_sha256") != (
        "4346562d4553a3df6afd0ef778ac3bfdb4bbb8fb688a5c47304b145ac6fc0bb5"):
    raise SystemExit("prior input-snapshot manifest changed")
for section, expected_path, expected_sha in (
        (
            "code",
            snapshot_root / "code" / "CODE_MANIFEST.json",
            "4d157ba04b390d5a753f0973ce9c16c7b45b6ce91d54e458fda5730bf8d6abd6",
        ),
        (
            "input_snapshot",
            snapshot_root / "inputs" / "INPUT_MANIFEST.json",
            "4346562d4553a3df6afd0ef778ac3bfdb4bbb8fb688a5c47304b145ac6fc0bb5",
        )):
    if provenance.get(section, {}).get("manifest_path") != str(expected_path):
        raise SystemExit("prior snapshot manifest path changed: " + section)
    read_regular(expected_path, expected_sha)

needle = prior_exp.encode("utf-8")
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        command_line = (entry / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if needle in command_line:
        raise SystemExit("prior failed-audit process is still alive: " + entry.name)
print("prior_failed_startup_verified=zero_batches_zero_receipts_zero_weights")
PY
}

verify_prior_failed_startup

verify_second_failed_postflight_contract() {
  if screen -ls | grep -Fq '.mcln_fpr_tv_audit_recovery2_100b'; then
    echo 'second recovery-audit screen is unexpectedly alive' >&2
    exit 6
  fi
  SECOND_ROOT_ENV="${SECOND_AUDIT_ROOT}" \
  SECOND_EXP_ENV="${SECOND_EXP}" \
  SECOND_TIMESTAMP_ENV="${SECOND_RUNTIME_TIMESTAMP}" \
  SECOND_PROVENANCE_ENV="${SECOND_FAILURE_PROVENANCE}" \
  SECOND_PROVENANCE_SHA_ENV="${SECOND_FAILURE_PROVENANCE_SHA256}" \
  SECOND_COMMAND_ENV="${SECOND_FAILURE_COMMAND}" \
  SECOND_COMMAND_SHA_ENV="${SECOND_FAILURE_COMMAND_SHA256}" \
  SECOND_LOG_ENV="${SECOND_FAILURE_LOG}" \
  SECOND_LOG_SHA_ENV="${SECOND_FAILURE_LOG_SHA256}" \
  SECOND_CONFIG_ENV="${SECOND_FAILURE_CONFIG}" \
  SECOND_CONFIG_SHA_ENV="${SECOND_FAILURE_CONFIG_SHA256}" \
  SECOND_TRAIN_LOG_ENV="${SECOND_FAILURE_TRAIN_LOG}" \
  SECOND_TRAIN_LOG_SHA_ENV="${SECOND_FAILURE_TRAIN_LOG_SHA256}" \
  SECOND_TB_TRAIN_ENV="${SECOND_FAILURE_TENSORBOARD_TRAIN}" \
  SECOND_TB_VAL_ENV="${SECOND_FAILURE_TENSORBOARD_VAL}" \
  SECOND_EMPTY_SHA_ENV="${SECOND_FAILURE_EMPTY_SHA256}" \
  SECOND_CODE_MANIFEST_ENV="${SECOND_CODE_MANIFEST}" \
  SECOND_CODE_MANIFEST_SHA_ENV="${SECOND_CODE_MANIFEST_SHA256}" \
  SECOND_INPUT_MANIFEST_ENV="${SECOND_INPUT_MANIFEST}" \
  SECOND_INPUT_MANIFEST_SHA_ENV="${SECOND_INPUT_MANIFEST_SHA256}" \
  SECOND_LAUNCHER_SHA_ENV="${SECOND_LAUNCHER_SHA256}" \
  SECOND_STATIC_EXEC_SHA_ENV="${SECOND_STATIC_EXEC_SHA256}" \
  STATIC_SOURCE_SHA_ENV="${ORIGINAL_STATIC_SOURCE_SHA256}" \
  E57_SHA_ENV="${CHECKPOINT_SHA256}" GF_SHA_ENV="${GROUPFREE_SHA256}" \
  DATA_SHA_ENV="${DATA_MANIFEST_SHA256}" \
  REVIEWED_SHA_ENV="${SECOND_REVIEWED_CODE_MANIFEST_SHA256}" \
  LANDLOCK_SHA_ENV="${LANDLOCK_EXECUTOR_SHA256}" \
  ORIGINAL_PROVENANCE_SHA_ENV="${ORIGINAL_FAILURE_PROVENANCE_SHA256}" \
  ORIGINAL_COMMAND_SHA_ENV="${ORIGINAL_FAILURE_COMMAND_SHA256}" \
  ORIGINAL_LOG_SHA_ENV="${ORIGINAL_FAILURE_LOG_SHA256}" \
  PRIOR_PROVENANCE_SHA_ENV="${PRIOR_FAILURE_PROVENANCE_SHA256}" \
  PRIOR_COMMAND_SHA_ENV="${PRIOR_FAILURE_COMMAND_SHA256}" \
  PRIOR_LOG_SHA_ENV="${PRIOR_FAILURE_LOG_SHA256}" \
  PRIOR_TRAIN_LOG_SHA_ENV="${PRIOR_FAILURE_TRAIN_LOG_SHA256}" \
  "${PYTHON_BIN}" - <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import pathlib
import stat


root = pathlib.Path(os.environ["SECOND_ROOT_ENV"])
exp = os.environ["SECOND_EXP_ENV"]
timestamp = os.environ["SECOND_TIMESTAMP_ENV"]
run_relative = "runtime_output/nr3d/{}/{}".format(exp, timestamp)
config_relative = run_relative + "/config.json"
train_log_relative = run_relative + "/log.txt"
train_event_relative = pathlib.Path(
    os.environ["SECOND_TB_TRAIN_ENV"]
).relative_to(root).as_posix()
val_event_relative = pathlib.Path(
    os.environ["SECOND_TB_VAL_ENV"]
).relative_to(root).as_posix()
expected_files = {
    "consumed_provenance.json": {
        "path": os.environ["SECOND_PROVENANCE_ENV"],
        "sha256": os.environ["SECOND_PROVENANCE_SHA_ENV"],
        "size": 5724,
        "mode": 0o444,
    },
    "formal_command.json": {
        "path": os.environ["SECOND_COMMAND_ENV"],
        "sha256": os.environ["SECOND_COMMAND_SHA_ENV"],
        "size": 6886,
        "mode": 0o444,
    },
    "runtime_output/launch.log": {
        "path": os.environ["SECOND_LOG_ENV"],
        "sha256": os.environ["SECOND_LOG_SHA_ENV"],
        "size": 20053,
        "mode": 0o600,
    },
    config_relative: {
        "path": os.environ["SECOND_CONFIG_ENV"],
        "sha256": os.environ["SECOND_CONFIG_SHA_ENV"],
        "size": 14650,
        "mode": 0o600,
    },
    train_log_relative: {
        "path": os.environ["SECOND_TRAIN_LOG_ENV"],
        "sha256": os.environ["SECOND_TRAIN_LOG_SHA_ENV"],
        "size": 14244,
        "mode": 0o600,
    },
    train_event_relative: {
        "path": os.environ["SECOND_TB_TRAIN_ENV"],
        "sha256": os.environ["SECOND_EMPTY_SHA_ENV"],
        "size": 0,
        "mode": 0o600,
    },
    val_event_relative: {
        "path": os.environ["SECOND_TB_VAL_ENV"],
        "sha256": os.environ["SECOND_EMPTY_SHA_ENV"],
        "size": 0,
        "mode": 0o600,
    },
}
expected_directories = {
    ".",
    "runtime_output",
    "runtime_output/nr3d",
    "runtime_output/nr3d/{}".format(exp),
    run_relative,
    run_relative + "/tensorboard",
    run_relative + "/tensorboard/train",
    run_relative + "/tensorboard/val",
    "runtime_output/runtime_home",
    "runtime_output/runtime_home/.nv",
    "runtime_output/runtime_home/.nv/ComputeCache",
    "runtime_output/runtime_home/hf",
    "runtime_output/runtime_home/torch",
    "runtime_output/runtime_home/xdg",
}


def read_regular(path, expected_sha):
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("second failure evidence is not regular: " + str(path))
        chunks = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise SystemExit("second failure evidence changed while reading: " + str(path))
    if digest.hexdigest() != expected_sha:
        raise SystemExit("second failure evidence SHA changed: " + str(path))
    return b"".join(chunks), before


if root.is_symlink() or root.resolve() != root or not root.is_dir():
    raise SystemExit("second failure root identity changed")
root_info = root.stat()
if (
        root_info.st_uid != 0
        or root_info.st_gid != 0
        or stat.S_IMODE(root_info.st_mode) != 0o700):
    raise SystemExit("second failure root owner changed")
snapshot_root = root / "consumed_snapshot"
if snapshot_root.is_symlink() or not snapshot_root.is_dir():
    raise SystemExit("second failure snapshot is missing or unsafe")

actual_files = {}
actual_directories = set()
for current, directory_names, file_names in os.walk(str(root)):
    current_path = pathlib.Path(current)
    relative_directory = current_path.relative_to(root).as_posix()
    relative_directory = "." if relative_directory == "." else relative_directory
    if relative_directory == "." and "consumed_snapshot" in directory_names:
        directory_names.remove("consumed_snapshot")
    current_info = current_path.stat()
    if (
            current_info.st_uid != 0
            or current_info.st_gid != 0
            or stat.S_IMODE(current_info.st_mode) != 0o700):
        raise SystemExit("second failure output directory metadata changed")
    for directory_name in directory_names:
        if (current_path / directory_name).is_symlink():
            raise SystemExit("second failure output contains a directory symlink")
    actual_directories.add(relative_directory)
    for file_name in file_names:
        candidate = current_path / file_name
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise SystemExit("second failure output contains a file symlink")
        actual_files[relative] = candidate
if set(actual_files) != set(expected_files):
    raise SystemExit("second failure non-snapshot file inventory changed")
if actual_directories != expected_directories:
    raise SystemExit("second failure non-snapshot directory inventory changed")

raw_by_name = {}
for relative, expected in expected_files.items():
    candidate = actual_files[relative]
    if str(candidate) != expected["path"]:
        raise SystemExit("second failure evidence path changed: " + relative)
    raw, info = read_regular(candidate, expected["sha256"])
    if (
            info.st_size != expected["size"]
            or stat.S_IMODE(info.st_mode) != expected["mode"]
            or info.st_uid != 0
            or info.st_gid != 0):
        raise SystemExit("second failure evidence metadata changed: " + relative)
    raw_by_name[relative] = raw

command = json.loads(raw_by_name["formal_command.json"].decode("utf-8"))
provenance = json.loads(raw_by_name["consumed_provenance.json"].decode("utf-8"))
config = json.loads(raw_by_name[config_relative].decode("utf-8"))
launch_text = raw_by_name["runtime_output/launch.log"].decode("utf-8")
train_text = raw_by_name[train_log_relative].decode("utf-8")
if raw_by_name[train_event_relative] != b"" or raw_by_name[val_event_relative] != b"":
    raise SystemExit("second failure unexpectedly wrote TensorBoard events")
if command.get("schema") != "mcln-fpr-tv-density-audit-command-v3":
    raise SystemExit("second formal-command schema changed")
argv = command.get("argv")
if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
    raise SystemExit("second formal-command argv changed")
for flag, expected in (
        ("--exp", exp),
        ("--log_dir", str(root / "runtime_output")),
        ("--max_train_batches", "100"),
        ("--gradient_accumulation_steps", "1"),
        ("--local_rank", "0"),
        ("--start_epoch", "58"),
        ("--max_epoch", "58")):
    if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != expected:
        raise SystemExit("second formal-command contract changed: " + flag)
if "--eval" in argv:
    raise SystemExit("second formal command unexpectedly requested evaluation")
expected_run_dir = str(root / run_relative)
expected_config = {
    "log_dir": expected_run_dir,
    "exp": exp,
    "dataset": ["nr3d"],
    "test_dataset": "nr3d",
    "start_epoch": 58,
    "max_epoch": 58,
    "max_train_batches": 100,
    "eval": False,
    "local_rank": 0,
}
for key, value in expected_config.items():
    if type(config.get(key)) is not type(value) or config.get(key) != value:
        raise SystemExit("second failure config changed: " + key)
if str(pathlib.Path(os.environ["SECOND_CONFIG_ENV"]).parent) != expected_run_dir:
    raise SystemExit("second config was not written to normalized run directory")
required_log_fragments = (
    "snapshot_verification=pass",
    "Full config saved to " + os.environ["SECOND_CONFIG_ENV"],
    "'log_dir': '" + expected_run_dir + "'",
)
if any(fragment not in launch_text for fragment in required_log_fragments):
    raise SystemExit("second pre-training log changed")

# BaseTrainTester.__init__ writes exactly these two durable log records before
# train_dist_mod.py calls BaseTrainTester.main.  main() then constructs both
# loaders and durably logs their sizes (plus the accumulation plan) before it
# can build/load the model or execute a single training micro-batch.  Pinning
# this pre-main boundary rules out the otherwise silent first 1--19 batches.
train_lines = train_text.splitlines()
if len(train_lines) != 2:
    raise SystemExit("second train log no longer stops at the config boundary")
if "Full config saved to " + os.environ["SECOND_CONFIG_ENV"] not in train_lines[0]:
    raise SystemExit("second train log first config record changed")
if "'log_dir': '" + expected_run_dir + "'" not in train_lines[1]:
    raise SystemExit("second train log config payload changed")
for forbidden in (
        "length of training dataset:", "gradient accumulation:",
        "length of testing dataset:", "=> loading checkpoint '",
        "=> loaded successfully '", "Train: [58]",
        "train_audit_receipt_epoch_58.json", "audit_complete=true",
        "bounded_train_audit", "Traceback (most recent call last):"):
    if forbidden in launch_text or forbidden in train_text:
        raise SystemExit(
            "second failure unexpectedly crossed the pre-loader boundary: "
            + forbidden
        )

if provenance.get("schema") != "mcln-fpr-tv-density-audit-consumed-provenance-v5":
    raise SystemExit("second consumed-provenance schema changed")
inputs = provenance.get("inputs")
if not isinstance(inputs, dict):
    raise SystemExit("second consumed provenance lacks inputs")
expected_input_hashes = {
    "launcher": os.environ["SECOND_LAUNCHER_SHA_ENV"],
    "checkpoint": os.environ["E57_SHA_ENV"],
    "groupfree": os.environ["GF_SHA_ENV"],
    "data_manifest": os.environ["DATA_SHA_ENV"],
    "reviewed_code_manifest": os.environ["REVIEWED_SHA_ENV"],
    "landlock_executor": os.environ["LANDLOCK_SHA_ENV"],
    "static_clean_env_executor": os.environ["SECOND_STATIC_EXEC_SHA_ENV"],
    "static_clean_env_source": os.environ["STATIC_SOURCE_SHA_ENV"],
    "command": os.environ["SECOND_COMMAND_SHA_ENV"],
    "original_failed_consumed_provenance": os.environ[
        "ORIGINAL_PROVENANCE_SHA_ENV"
    ],
    "original_failed_formal_command": os.environ["ORIGINAL_COMMAND_SHA_ENV"],
    "original_failed_launch_log": os.environ["ORIGINAL_LOG_SHA_ENV"],
    "prior_failed_consumed_provenance": os.environ["PRIOR_PROVENANCE_SHA_ENV"],
    "prior_failed_formal_command": os.environ["PRIOR_COMMAND_SHA_ENV"],
    "prior_failed_launch_log": os.environ["PRIOR_LOG_SHA_ENV"],
    "prior_failed_empty_train_log": os.environ["PRIOR_TRAIN_LOG_SHA_ENV"],
}
if set(inputs) != set(expected_input_hashes):
    raise SystemExit("second consumed-provenance input set changed")
for label, expected_sha in expected_input_hashes.items():
    item = inputs.get(label)
    if not isinstance(item, dict) or item.get("sha256") != expected_sha:
        raise SystemExit("second consumed provenance changed: " + label)

expected_manifests = (
    (
        "code", pathlib.Path(os.environ["SECOND_CODE_MANIFEST_ENV"]),
        os.environ["SECOND_CODE_MANIFEST_SHA_ENV"], "code",
    ),
    (
        "input_snapshot", pathlib.Path(os.environ["SECOND_INPUT_MANIFEST_ENV"]),
        os.environ["SECOND_INPUT_MANIFEST_SHA_ENV"], "inputs",
    ),
)
for section, manifest_path, manifest_sha, leaf in expected_manifests:
    expected_root = snapshot_root / leaf
    expected = {
        "root": str(expected_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
    }
    if provenance.get(section) != expected:
        raise SystemExit("second snapshot provenance changed: " + section)
    read_regular(manifest_path, manifest_sha)

needle = exp.encode("utf-8")
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        command_line = (entry / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if needle in command_line:
        raise SystemExit("second failed-audit process is still alive: " + entry.name)
print(
    "second_failed_postflight_contract_verified="
    "normalized_nested_log_dir_pre_loader_zero_batches_zero_receipts_zero_weights"
)
PY
}

verify_second_failed_postflight_contract

verify_third_failed_collate_contract() {
  if screen -ls | grep -Fq '.mcln_fpr_tv_audit_recovery3_100b'; then
    echo 'third recovery-audit screen is unexpectedly alive' >&2
    exit 6
  fi
  THIRD_ROOT_ENV="${THIRD_AUDIT_ROOT}" \
  THIRD_EXP_ENV="${THIRD_EXP}" \
  THIRD_TIMESTAMP_ENV="${THIRD_RUNTIME_TIMESTAMP}" \
  THIRD_EVIDENCE_ENV="${THIRD_FAILURE_EVIDENCE}" \
  THIRD_EVIDENCE_SHA_ENV="${THIRD_FAILURE_EVIDENCE_SHA256}" \
  THIRD_REPLAY_SCRIPT_ENV="${THIRD_FIRST_BATCH_REPLAY_SCRIPT}" \
  THIRD_REPLAY_SCRIPT_SHA_ENV="${THIRD_FIRST_BATCH_REPLAY_SCRIPT_SHA256}" \
  THIRD_REPLAY_RECEIPT_ENV="${THIRD_FIRST_BATCH_REPLAY_RECEIPT}" \
  THIRD_REPLAY_RECEIPT_SHA_ENV="${THIRD_FIRST_BATCH_REPLAY_RECEIPT_SHA256}" \
  THIRD_CODE_MANIFEST_ENV="${THIRD_CODE_MANIFEST}" \
  THIRD_CODE_MANIFEST_SHA_ENV="${THIRD_CODE_MANIFEST_SHA256}" \
  THIRD_INPUT_MANIFEST_ENV="${THIRD_INPUT_MANIFEST}" \
  THIRD_INPUT_MANIFEST_SHA_ENV="${THIRD_INPUT_MANIFEST_SHA256}" \
  THIRD_LAUNCHER_SHA_ENV="${THIRD_LAUNCHER_SHA256}" \
  THIRD_STATIC_EXEC_SHA_ENV="${THIRD_STATIC_EXEC_SHA256}" \
  THIRD_REVIEWED_MANIFEST_SHA_ENV="${THIRD_REVIEWED_CODE_MANIFEST_SHA256}" \
  THIRD_OLD_MAIN_SHA_ENV='df1780a6ed0c8678759f33d060ad1e0aff25f39b6787f2bc0536540bd2da1ea5' \
  E57_SHA_ENV="${CHECKPOINT_SHA256}" GF_SHA_ENV="${GROUPFREE_SHA256}" \
  DATA_SHA_ENV="${DATA_MANIFEST_SHA256}" \
  LANDLOCK_SHA_ENV="${LANDLOCK_EXECUTOR_SHA256}" \
  STATIC_SOURCE_SHA_ENV="${TRUSTED_STATIC_SOURCE_SHA256}" \
  "${PYTHON_BIN}" - <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import pathlib
import stat


root = pathlib.Path(os.environ["THIRD_ROOT_ENV"])
exp = os.environ["THIRD_EXP_ENV"]
timestamp = os.environ["THIRD_TIMESTAMP_ENV"]
evidence_path = pathlib.Path(os.environ["THIRD_EVIDENCE_ENV"])


def read_regular(path, expected_sha):
    path = pathlib.Path(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("third failure evidence is not regular: " + str(path))
        chunks = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise SystemExit("third failure evidence changed while reading: " + str(path))
    if digest.hexdigest() != expected_sha:
        raise SystemExit("third failure evidence SHA changed: " + str(path))
    return b"".join(chunks), before


evidence_raw, evidence_info = read_regular(
    evidence_path, os.environ["THIRD_EVIDENCE_SHA_ENV"]
)
if (
        evidence_info.st_uid != 0
        or evidence_info.st_gid != 0
        or stat.S_IMODE(evidence_info.st_mode) != 0o444
        or evidence_info.st_size != 4831):
    raise SystemExit("third failure evidence manifest metadata changed")
evidence = json.loads(evidence_raw.decode("utf-8"))
if evidence.get("schema") != "mcln-fpr-tv-density-audit-failure-evidence-v1":
    raise SystemExit("third failure evidence schema changed")
if (
        evidence.get("root") != str(root)
        or evidence.get("experiment") != exp
        or evidence.get("runtime_timestamp") != timestamp
        or evidence.get("failure_stage") != "dataloader_collate_before_first_batch_yield"
        or evidence.get("optimizer_steps") != 0
        or evidence.get("receipts") != 0
        or evidence.get("weights") != 0):
    raise SystemExit("third failure evidence contract changed")

replay = evidence.get("first_batch_replay")
if (
        not isinstance(replay, dict)
        or replay.get("conclusion")
        != "first_batch_fails_before_training_loop_body"):
    raise SystemExit("third failure lacks first-batch replay evidence")
replay_raw = {}
for label, path_environment, sha_environment in (
        ("script", "THIRD_REPLAY_SCRIPT_ENV", "THIRD_REPLAY_SCRIPT_SHA_ENV"),
        ("receipt", "THIRD_REPLAY_RECEIPT_ENV", "THIRD_REPLAY_RECEIPT_SHA_ENV")):
    record = replay.get(label)
    path = pathlib.Path(os.environ[path_environment])
    expected_sha = os.environ[sha_environment]
    if (
            not isinstance(record, dict)
            or record.get("path") != str(path)
            or record.get("sha256") != expected_sha
            or record.get("owner") != "0:0"
            or record.get("mode") != "0444"):
        raise SystemExit("third first-batch replay record changed: " + label)
    raw, info = read_regular(path, expected_sha)
    if (
            info.st_uid != 0
            or info.st_gid != 0
            or stat.S_IMODE(info.st_mode) != 0o444
            or info.st_size != record.get("size")):
        raise SystemExit("third first-batch replay metadata changed: " + label)
    replay_raw[label] = raw
replay_receipt = json.loads(replay_raw["receipt"].decode("utf-8"))
if (
        replay_receipt.get("schema")
        != "mcln-fpr-tv-first-batch-collate-replay-v1"
        or replay_receipt.get("audit_root") != str(root)
        or replay_receipt.get("epoch") != 58
        or replay_receipt.get("batch_size") != 16
        or replay_receipt.get("dataset_length") != 44909
        or replay_receipt.get("failed_main_utils_sha256")
        != os.environ["THIRD_OLD_MAIN_SHA_ENV"]
        or replay_receipt.get("conclusion")
        != "first_batch_fails_before_training_loop_body"):
    raise SystemExit("third first-batch replay receipt contract changed")
indices = replay_receipt.get("first_batch_indices")
variable_lengths = replay_receipt.get("variable_structured_lengths")
default_result = replay_receipt.get("default_collate")
structured_result = replay_receipt.get("structured_collate")
if (
        not isinstance(indices, list)
        or len(indices) != 16
        or any(not isinstance(index, int) for index in indices)
        or len(set(indices)) != 16
        or not isinstance(variable_lengths, dict)
        or not variable_lengths
        or not isinstance(default_result, dict)
        or default_result.get("failed") is not True
        or default_result.get("exception_type") != "RuntimeError"
        or default_result.get("exception_message")
        != "each element in list of batch should be of equal size"
        or not isinstance(structured_result, dict)
        or structured_result.get("succeeded") is not True):
    raise SystemExit("third first-batch replay did not reproduce the collate failure")

if root.is_symlink() or root.resolve() != root or not root.is_dir():
    raise SystemExit("third failure root identity changed")
root_info = root.stat()
if (
        root_info.st_uid != 0
        or root_info.st_gid != 0
        or stat.S_IMODE(root_info.st_mode) != 0o700
        or evidence.get("root_mode") != "0700"):
    raise SystemExit("third failure root metadata changed")
snapshot_root = root / "consumed_snapshot"
if snapshot_root.is_symlink() or not snapshot_root.is_dir():
    raise SystemExit("third failure snapshot is missing or unsafe")

actual_files = {}
actual_directories = set()
for current, directory_names, file_names in os.walk(str(root)):
    current_path = pathlib.Path(current)
    relative_directory = current_path.relative_to(root).as_posix()
    relative_directory = "." if relative_directory == "." else relative_directory
    if relative_directory == "." and "consumed_snapshot" in directory_names:
        directory_names.remove("consumed_snapshot")
    current_info = current_path.stat()
    if (
            current_info.st_uid != 0
            or current_info.st_gid != 0
            or stat.S_IMODE(current_info.st_mode) != 0o700):
        raise SystemExit("third failure output directory metadata changed")
    for directory_name in directory_names:
        if (current_path / directory_name).is_symlink():
            raise SystemExit("third failure output contains a directory symlink")
    actual_directories.add(relative_directory)
    for file_name in file_names:
        candidate = current_path / file_name
        if candidate.is_symlink():
            raise SystemExit("third failure output contains a file symlink")
        actual_files[candidate.relative_to(root).as_posix()] = candidate

expected_files = evidence.get("files")
expected_directories = evidence.get("directories")
if not isinstance(expected_files, dict) or not isinstance(expected_directories, list):
    raise SystemExit("third failure evidence inventory changed")
if set(actual_files) != set(expected_files):
    raise SystemExit("third failure non-snapshot file inventory changed")
if actual_directories != set(expected_directories) | {"."}:
    raise SystemExit("third failure non-snapshot directory inventory changed")

raw_by_name = {}
for relative, record in expected_files.items():
    if not isinstance(record, dict):
        raise SystemExit("third failure file record changed: " + relative)
    raw, info = read_regular(actual_files[relative], record.get("sha256"))
    if (
            info.st_uid != 0
            or info.st_gid != 0
            or stat.S_IMODE(info.st_mode) != int(record.get("mode"), 8)
            or info.st_size != record.get("size")):
        raise SystemExit("third failure file metadata changed: " + relative)
    # The launch log contains tqdm carriage returns; the frozen evidence uses
    # POSIX newline count (wc -l), not Python's broader splitlines semantics.
    if "line_count" in record and raw.count(b"\n") != record["line_count"]:
        raise SystemExit("third failure line count changed: " + relative)
    raw_by_name[relative] = raw

run_relative = "runtime_output/nr3d/{}/{}".format(exp, timestamp)
config_relative = run_relative + "/config.json"
train_log_relative = run_relative + "/log.txt"
command = json.loads(raw_by_name["formal_command.json"].decode("utf-8"))
provenance = json.loads(raw_by_name["consumed_provenance.json"].decode("utf-8"))
config = json.loads(raw_by_name[config_relative].decode("utf-8"))
launch_text = raw_by_name["runtime_output/launch.log"].decode("utf-8")
train_text = raw_by_name[train_log_relative].decode("utf-8")

if command.get("schema") != "mcln-fpr-tv-density-audit-command-v3":
    raise SystemExit("third formal-command schema changed")
argv = command.get("argv")
if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
    raise SystemExit("third formal-command argv changed")
for flag, expected in (
        ("--exp", exp),
        ("--log_dir", str(root / "runtime_output")),
        ("--batch_size", "16"),
        ("--gradient_accumulation_steps", "1"),
        ("--max_train_batches", "100"),
        ("--local_rank", "0"),
        ("--start_epoch", "58"),
        ("--max_epoch", "58")):
    if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != expected:
        raise SystemExit("third formal-command contract changed: " + flag)
if "--eval" in argv:
    raise SystemExit("third formal command unexpectedly requested evaluation")

expected_run_dir = str(root / run_relative)
for key, value in {
        "log_dir": expected_run_dir,
        "exp": exp,
        "dataset": ["nr3d"],
        "test_dataset": "nr3d",
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "start_epoch": 58,
        "max_epoch": 58,
        "max_train_batches": 100,
        "eval": False,
        "local_rank": 0,
        "use_parent_relative_text_verifier": True,
}.items():
    if type(config.get(key)) is not type(value) or config.get(key) != value:
        raise SystemExit("third failure config changed: " + key)
if str(pathlib.Path(actual_files[config_relative]).parent) != expected_run_dir:
    raise SystemExit("third config was not written to normalized run directory")

for fragment in evidence.get("required_pre_failure_fragments", []):
    if fragment not in launch_text and fragment not in train_text:
        raise SystemExit("third pre-failure evidence changed: " + fragment)
for fragment in evidence.get("required_trace_fragments", []):
    if fragment not in launch_text:
        raise SystemExit("third collate traceback changed: " + fragment)
for fragment in evidence.get("forbidden_fragments", []):
    if fragment in launch_text or fragment in train_text:
        raise SystemExit("third failure crossed the first-batch boundary: " + fragment)

if provenance.get("schema") != "mcln-fpr-tv-density-audit-consumed-provenance-v6":
    raise SystemExit("third consumed-provenance schema changed")
inputs = provenance.get("inputs")
for label, expected_sha in {
        "launcher": os.environ["THIRD_LAUNCHER_SHA_ENV"],
        "checkpoint": os.environ["E57_SHA_ENV"],
        "groupfree": os.environ["GF_SHA_ENV"],
        "data_manifest": os.environ["DATA_SHA_ENV"],
        "reviewed_code_manifest": os.environ["THIRD_REVIEWED_MANIFEST_SHA_ENV"],
        "landlock_executor": os.environ["LANDLOCK_SHA_ENV"],
        "static_clean_env_executor": os.environ["THIRD_STATIC_EXEC_SHA_ENV"],
        "static_clean_env_source": os.environ["STATIC_SOURCE_SHA_ENV"],
        "command": evidence["files"]["formal_command.json"]["sha256"],
}.items():
    item = inputs.get(label) if isinstance(inputs, dict) else None
    if not isinstance(item, dict) or item.get("sha256") != expected_sha:
        raise SystemExit("third consumed provenance changed: " + label)

snapshot_manifests = evidence.get("snapshot_manifests")
for section, environment_path, environment_sha, leaf in (
        ("code", "THIRD_CODE_MANIFEST_ENV", "THIRD_CODE_MANIFEST_SHA_ENV", "code"),
        ("inputs", "THIRD_INPUT_MANIFEST_ENV", "THIRD_INPUT_MANIFEST_SHA_ENV", "inputs")):
    record = snapshot_manifests.get(section) if isinstance(snapshot_manifests, dict) else None
    manifest_path = pathlib.Path(os.environ[environment_path])
    manifest_sha = os.environ[environment_sha]
    if (
            not isinstance(record, dict)
            or record.get("path") != manifest_path.relative_to(root).as_posix()
            or record.get("sha256") != manifest_sha):
        raise SystemExit("third snapshot evidence changed: " + section)
    manifest_raw, _ = read_regular(manifest_path, manifest_sha)
    manifest = json.loads(manifest_raw.decode("utf-8"))
    provenance_section = "input_snapshot" if section == "inputs" else "code"
    if provenance.get(provenance_section) != {
            "root": str(snapshot_root / leaf),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha}:
        raise SystemExit("third snapshot provenance changed: " + section)
    if section == "code":
        main_record = manifest.get("files", {}).get("main_utils.py")
        if (
                not isinstance(main_record, dict)
                or main_record.get("sha256") != os.environ["THIRD_OLD_MAIN_SHA_ENV"]):
            raise SystemExit("third failed code did not contain the reviewed pre-fix main_utils")
    else:
        reviewed_record = manifest.get("files", {}).get(
            "reviewed_runtime_code_manifest.json"
        )
        if (
                not isinstance(reviewed_record, dict)
                or reviewed_record.get("sha256") != os.environ[
                    "THIRD_REVIEWED_MANIFEST_SHA_ENV"
                ]):
            raise SystemExit("third failed input snapshot changed")

for relative in expected_files:
    if relative.endswith(".pth") or "receipt" in relative:
        raise SystemExit("third failure unexpectedly contains a result artifact")
needle = exp.encode("utf-8")
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        command_line = (entry / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if needle in command_line:
        raise SystemExit("third failed-audit process is still alive: " + entry.name)
print(
    "third_failed_collate_contract_verified="
    "pre_first_batch_zero_optimizer_steps_zero_receipts_zero_weights"
)
PY
}

verify_third_failed_collate_contract
[[ ! -e "${AUDIT_ROOT}" ]] || {
  echo "one-shot root exists: ${AUDIT_ROOT}" >&2
  exit 6
}
verify_fixed_inputs
verify_reviewed_code_manifest \
  "${ROOT_DIR}" "${REVIEWED_CODE_MANIFEST}" \
  "${REVIEWED_CODE_MANIFEST_SHA256}"
verify_dataset_manifest "${DATA_MANIFEST}" "${DATA_MANIFEST_SHA256}"

CHECKPOINT_ENV="${CHECKPOINT}" CHECKPOINT_SHA_ENV="${CHECKPOINT_SHA256}" \
"${PYTHON_BIN}" - <<'PY'
from __future__ import print_function

import hashlib
import os

import torch

path = os.environ["CHECKPOINT_ENV"]
expected_sha = os.environ["CHECKPOINT_SHA_ENV"]
with open(path, "rb") as handle:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        raise SystemExit("protected E57 SHA changed before load")
    handle.seek(0)
    checkpoint = torch.load(handle, map_location="cpu")
    handle.seek(0)
    post = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        post.update(chunk)
    if post.hexdigest() != expected_sha:
        raise SystemExit("protected E57 changed during load")

config = checkpoint.get("config")
config = vars(config) if hasattr(config, "__dict__") else dict(config or {})
expected = {
    "batch_size": 16,
    "joint_det": True,
    "butd_cls": True,
    "test_dataset": "nr3d",
    "use_source_choice_selector": True,
    "eval_use_selector_choice_scores": True,
    "source_choice_selector_sources": "default,default_rank_blend_contrastive010",
    "source_choice_selector_default_source": "default",
    "source_choice_selector_hidden_dim": 288,
    "source_choice_selector_lr": 1.25e-4,
    "source_choice_selector_loss_weight": 0.5,
    "source_choice_selector_choice_target": "precision_gain_default_sourcewise_focal_bce",
    "source_choice_selector_min_iou_gap": 0.03,
}
if checkpoint.get("epoch") != 57:
    raise SystemExit("protected checkpoint is not E57")
if config.get("dataset") != ["nr3d"]:
    raise SystemExit("protected checkpoint dataset changed")
if config.get("gradient_accumulation_steps", 1) != 1:
    raise SystemExit("protected checkpoint accumulation changed")
for key, value in expected.items():
    if type(config.get(key)) is not type(value) or config.get(key) != value:
        raise SystemExit("protected checkpoint {} changed".format(key))
if config.get("use_source_moe", False) is not False:
    raise SystemExit("protected checkpoint unexpectedly enables SourceMoE")
if len(checkpoint.get("model", {})) != 1144:
    raise SystemExit("protected checkpoint model topology changed")
if "optimizer" in checkpoint or "scheduler" in checkpoint:
    raise SystemExit("protected official checkpoint is no longer model-only")
print("checkpoint_provenance=protected_E57_V99_model_only_verified")
PY

free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
if ((gpu_used >= 500)); then echo "GPU0 busy: ${gpu_used} MiB" >&2; exit 4; fi
if ((free_gb < MIN_FREE_GB)); then echo 'insufficient DATA_ROOT space' >&2; exit 5; fi
/usr/bin/env -i HOME=/root USER=root LOGNAME=root LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PATH=/root/miniconda3/envs/bdetr/bin:/usr/local/cuda/bin:/usr/bin:/bin \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 \
  LOCAL_RANK="${LOCAL_RANK}" "${PYTHON_BIN}" -c \
  'import os, torch; rank = int(os.environ["LOCAL_RANK"]); assert torch.cuda.device_count() == 1; torch.cuda.set_device(rank); assert torch.cuda.current_device() == rank; print("cuda_local_rank_preflight=pass rank={}".format(rank))'

if [[ "${MODE}" == 'preflight' ]]; then
  echo 'preflight=pass audit_only=true long_training_authorized=false'
  exit 0
fi

mkdir -p "$(dirname "${AUDIT_ROOT}")"
mkdir "${AUDIT_ROOT}"
chmod 0700 "${AUDIT_ROOT}"
readonly RUNTIME_OUTPUT="${AUDIT_ROOT}/runtime_output"
readonly RUNTIME_HOME="${RUNTIME_OUTPUT}/runtime_home"
mkdir "${RUNTIME_OUTPUT}"
chmod 0700 "${RUNTIME_OUTPUT}"
readonly LAUNCH_LOG="${RUNTIME_OUTPUT}/launch.log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1
echo 'audit_only=true long_training_authorized=false'
echo "audit_root=${AUDIT_ROOT}"

readonly SNAPSHOT_ROOT="${AUDIT_ROOT}/consumed_snapshot"
readonly CODE_SNAPSHOT="${SNAPSHOT_ROOT}/code"
readonly INPUT_SNAPSHOT="${SNAPSHOT_ROOT}/inputs"
readonly INPUT_CHECKPOINT="${INPUT_SNAPSHOT}/e57_checkpoint.pth"
readonly INPUT_GROUPFREE="${INPUT_SNAPSHOT}/gf_detector_l6o256.pth"
readonly INPUT_DATA_MANIFEST="${INPUT_SNAPSHOT}/nr3d_train_input_manifest_v1.json"
readonly INPUT_REVIEWED_CODE_MANIFEST="${INPUT_SNAPSHOT}/reviewed_runtime_code_manifest.json"
readonly INPUT_LANDLOCK_EXECUTOR="${INPUT_SNAPSHOT}/mcln_landlock_snapshot_exec.py"
readonly INPUT_STATIC_EXECUTOR="${INPUT_SNAPSHOT}/mcln_fpr_audit_static_exec.x86_64"
readonly INPUT_STATIC_SOURCE="${INPUT_SNAPSHOT}/mcln_fpr_audit_static_exec.c"
readonly INPUT_ORIGINAL_FAILURE_PROVENANCE="${INPUT_SNAPSHOT}/original_failed_consumed_provenance.json"
readonly INPUT_ORIGINAL_FAILURE_COMMAND="${INPUT_SNAPSHOT}/original_failed_formal_command.json"
readonly INPUT_ORIGINAL_FAILURE_LOG="${INPUT_SNAPSHOT}/original_failed_launch.log"
readonly INPUT_PRIOR_FAILURE_PROVENANCE="${INPUT_SNAPSHOT}/prior_failed_consumed_provenance.json"
readonly INPUT_PRIOR_FAILURE_COMMAND="${INPUT_SNAPSHOT}/prior_failed_formal_command.json"
readonly INPUT_PRIOR_FAILURE_LOG="${INPUT_SNAPSHOT}/prior_failed_launch.log"
readonly INPUT_PRIOR_FAILURE_TRAIN_LOG="${INPUT_SNAPSHOT}/prior_failed_empty_train.log"
readonly INPUT_SECOND_FAILURE_PROVENANCE="${INPUT_SNAPSHOT}/second_failed_consumed_provenance.json"
readonly INPUT_SECOND_FAILURE_COMMAND="${INPUT_SNAPSHOT}/second_failed_formal_command.json"
readonly INPUT_SECOND_FAILURE_LOG="${INPUT_SNAPSHOT}/second_failed_launch.log"
readonly INPUT_SECOND_FAILURE_CONFIG="${INPUT_SNAPSHOT}/second_failed_config.json"
readonly INPUT_SECOND_FAILURE_TRAIN_LOG="${INPUT_SNAPSHOT}/second_failed_train.log"
readonly INPUT_SECOND_FAILURE_TENSORBOARD_TRAIN="${INPUT_SNAPSHOT}/second_failed_tensorboard_train.events"
readonly INPUT_SECOND_FAILURE_TENSORBOARD_VAL="${INPUT_SNAPSHOT}/second_failed_tensorboard_val.events"
readonly INPUT_SECOND_CODE_MANIFEST="${INPUT_SNAPSHOT}/second_failed_code_manifest.json"
readonly INPUT_SECOND_INPUT_MANIFEST="${INPUT_SNAPSHOT}/second_failed_input_manifest.json"
readonly INPUT_THIRD_FAILURE_EVIDENCE_MANIFEST="${INPUT_SNAPSHOT}/third_failed_collate_evidence.json"
readonly INPUT_THIRD_FIRST_BATCH_REPLAY_SCRIPT="${INPUT_SNAPSHOT}/third_failed_first_batch_replay.py"
readonly INPUT_THIRD_FIRST_BATCH_REPLAY_RECEIPT="${INPUT_SNAPSHOT}/third_failed_first_batch_replay_receipt.json"
readonly INPUT_THIRD_FAILURE_PROVENANCE="${INPUT_SNAPSHOT}/third_failed_consumed_provenance.json"
readonly INPUT_THIRD_FAILURE_COMMAND="${INPUT_SNAPSHOT}/third_failed_formal_command.json"
readonly INPUT_THIRD_FAILURE_LOG="${INPUT_SNAPSHOT}/third_failed_launch.log"
readonly INPUT_THIRD_FAILURE_CONFIG="${INPUT_SNAPSHOT}/third_failed_config.json"
readonly INPUT_THIRD_FAILURE_TRAIN_LOG="${INPUT_SNAPSHOT}/third_failed_train.log"
readonly INPUT_THIRD_FAILURE_TENSORBOARD_TRAIN="${INPUT_SNAPSHOT}/third_failed_tensorboard_train.events"
readonly INPUT_THIRD_FAILURE_TENSORBOARD_VAL="${INPUT_SNAPSHOT}/third_failed_tensorboard_val.events"
readonly INPUT_THIRD_CODE_MANIFEST="${INPUT_SNAPSHOT}/third_failed_code_manifest.json"
readonly INPUT_THIRD_INPUT_MANIFEST="${INPUT_SNAPSHOT}/third_failed_input_manifest.json"
readonly CODE_MANIFEST="${CODE_SNAPSHOT}/CODE_MANIFEST.json"
readonly INPUT_MANIFEST="${INPUT_SNAPSHOT}/INPUT_MANIFEST.json"
readonly COMMAND_MANIFEST="${AUDIT_ROOT}/formal_command.json"
readonly CONSUMED_PROVENANCE="${AUDIT_ROOT}/consumed_provenance.json"
mkdir "${SNAPSHOT_ROOT}"
case "${SNAPSHOT_ROOT}" in
  "${AUDIT_ROOT}"/*) ;;
  *) echo 'snapshot root escaped one-shot root' >&2; exit 8 ;;
esac

# Build both immutable snapshots from the separately reviewed runtime closure.
# Every copied byte is hashed from the same opened source descriptor that is
# consumed, then the resulting trees are owned by an unprivileged UID.  The
# Landlock executor verifies these manifests again before dropping all root
# capabilities and executing training in the same process.
snapshot_build_output="$(
  SNAP_SOURCE_ROOT_ENV="${ROOT_DIR}" \
  SNAP_CODE_ROOT_ENV="${CODE_SNAPSHOT}" \
  SNAP_INPUT_ROOT_ENV="${INPUT_SNAPSHOT}" \
  SNAP_REVIEWED_MANIFEST_ENV="${REVIEWED_CODE_MANIFEST}" \
  SNAP_REVIEWED_MANIFEST_SHA_ENV="${REVIEWED_CODE_MANIFEST_SHA256}" \
  SNAP_E57_ENV="${CHECKPOINT}" SNAP_E57_SHA_ENV="${CHECKPOINT_SHA256}" \
  SNAP_GF_ENV="${GROUPFREE_CHECKPOINT}" SNAP_GF_SHA_ENV="${GROUPFREE_SHA256}" \
  SNAP_DATA_MANIFEST_ENV="${DATA_MANIFEST}" \
  SNAP_DATA_MANIFEST_SHA_ENV="${DATA_MANIFEST_SHA256}" \
  SNAP_LANDLOCK_ENV="${LANDLOCK_EXECUTOR}" \
  SNAP_LANDLOCK_SHA_ENV="${LANDLOCK_EXECUTOR_SHA256}" \
  SNAP_STATIC_EXEC_ENV="${TRUSTED_STATIC_EXEC_PATH}" \
  SNAP_STATIC_EXEC_SHA_ENV="${TRUSTED_STATIC_EXEC_SHA256}" \
  SNAP_STATIC_SOURCE_ENV="${TRUSTED_STATIC_SOURCE_PATH}" \
  SNAP_STATIC_SOURCE_SHA_ENV="${TRUSTED_STATIC_SOURCE_SHA256}" \
  SNAP_ORIGINAL_PROVENANCE_ENV="${ORIGINAL_FAILURE_PROVENANCE}" \
  SNAP_ORIGINAL_PROVENANCE_SHA_ENV="${ORIGINAL_FAILURE_PROVENANCE_SHA256}" \
  SNAP_ORIGINAL_COMMAND_ENV="${ORIGINAL_FAILURE_COMMAND}" \
  SNAP_ORIGINAL_COMMAND_SHA_ENV="${ORIGINAL_FAILURE_COMMAND_SHA256}" \
  SNAP_ORIGINAL_LOG_ENV="${ORIGINAL_FAILURE_LOG}" \
  SNAP_ORIGINAL_LOG_SHA_ENV="${ORIGINAL_FAILURE_LOG_SHA256}" \
  SNAP_PRIOR_PROVENANCE_ENV="${PRIOR_FAILURE_PROVENANCE}" \
  SNAP_PRIOR_PROVENANCE_SHA_ENV="${PRIOR_FAILURE_PROVENANCE_SHA256}" \
  SNAP_PRIOR_COMMAND_ENV="${PRIOR_FAILURE_COMMAND}" \
  SNAP_PRIOR_COMMAND_SHA_ENV="${PRIOR_FAILURE_COMMAND_SHA256}" \
  SNAP_PRIOR_LOG_ENV="${PRIOR_FAILURE_LOG}" \
  SNAP_PRIOR_LOG_SHA_ENV="${PRIOR_FAILURE_LOG_SHA256}" \
  SNAP_PRIOR_TRAIN_LOG_ENV="${PRIOR_FAILURE_TRAIN_LOG}" \
  SNAP_PRIOR_TRAIN_LOG_SHA_ENV="${PRIOR_FAILURE_TRAIN_LOG_SHA256}" \
  SNAP_SECOND_PROVENANCE_ENV="${SECOND_FAILURE_PROVENANCE}" \
  SNAP_SECOND_PROVENANCE_SHA_ENV="${SECOND_FAILURE_PROVENANCE_SHA256}" \
  SNAP_SECOND_COMMAND_ENV="${SECOND_FAILURE_COMMAND}" \
  SNAP_SECOND_COMMAND_SHA_ENV="${SECOND_FAILURE_COMMAND_SHA256}" \
  SNAP_SECOND_LOG_ENV="${SECOND_FAILURE_LOG}" \
  SNAP_SECOND_LOG_SHA_ENV="${SECOND_FAILURE_LOG_SHA256}" \
  SNAP_SECOND_CONFIG_ENV="${SECOND_FAILURE_CONFIG}" \
  SNAP_SECOND_CONFIG_SHA_ENV="${SECOND_FAILURE_CONFIG_SHA256}" \
  SNAP_SECOND_TRAIN_LOG_ENV="${SECOND_FAILURE_TRAIN_LOG}" \
  SNAP_SECOND_TRAIN_LOG_SHA_ENV="${SECOND_FAILURE_TRAIN_LOG_SHA256}" \
  SNAP_SECOND_TB_TRAIN_ENV="${SECOND_FAILURE_TENSORBOARD_TRAIN}" \
  SNAP_SECOND_TB_VAL_ENV="${SECOND_FAILURE_TENSORBOARD_VAL}" \
  SNAP_SECOND_EMPTY_SHA_ENV="${SECOND_FAILURE_EMPTY_SHA256}" \
  SNAP_SECOND_CODE_MANIFEST_ENV="${SECOND_CODE_MANIFEST}" \
  SNAP_SECOND_CODE_MANIFEST_SHA_ENV="${SECOND_CODE_MANIFEST_SHA256}" \
  SNAP_SECOND_INPUT_MANIFEST_ENV="${SECOND_INPUT_MANIFEST}" \
  SNAP_SECOND_INPUT_MANIFEST_SHA_ENV="${SECOND_INPUT_MANIFEST_SHA256}" \
  SNAP_THIRD_EVIDENCE_ENV="${THIRD_FAILURE_EVIDENCE}" \
  SNAP_THIRD_EVIDENCE_SHA_ENV="${THIRD_FAILURE_EVIDENCE_SHA256}" \
  SNAP_THIRD_REPLAY_SCRIPT_ENV="${THIRD_FIRST_BATCH_REPLAY_SCRIPT}" \
  SNAP_THIRD_REPLAY_SCRIPT_SHA_ENV="${THIRD_FIRST_BATCH_REPLAY_SCRIPT_SHA256}" \
  SNAP_THIRD_REPLAY_RECEIPT_ENV="${THIRD_FIRST_BATCH_REPLAY_RECEIPT}" \
  SNAP_THIRD_REPLAY_RECEIPT_SHA_ENV="${THIRD_FIRST_BATCH_REPLAY_RECEIPT_SHA256}" \
  SNAP_THIRD_PROVENANCE_ENV="${THIRD_FAILURE_PROVENANCE}" \
  SNAP_THIRD_PROVENANCE_SHA_ENV="${THIRD_FAILURE_PROVENANCE_SHA256}" \
  SNAP_THIRD_COMMAND_ENV="${THIRD_FAILURE_COMMAND}" \
  SNAP_THIRD_COMMAND_SHA_ENV="${THIRD_FAILURE_COMMAND_SHA256}" \
  SNAP_THIRD_LOG_ENV="${THIRD_FAILURE_LOG}" \
  SNAP_THIRD_LOG_SHA_ENV="${THIRD_FAILURE_LOG_SHA256}" \
  SNAP_THIRD_CONFIG_ENV="${THIRD_FAILURE_CONFIG}" \
  SNAP_THIRD_CONFIG_SHA_ENV="${THIRD_FAILURE_CONFIG_SHA256}" \
  SNAP_THIRD_TRAIN_LOG_ENV="${THIRD_FAILURE_TRAIN_LOG}" \
  SNAP_THIRD_TRAIN_LOG_SHA_ENV="${THIRD_FAILURE_TRAIN_LOG_SHA256}" \
  SNAP_THIRD_TB_TRAIN_ENV="${THIRD_FAILURE_TENSORBOARD_TRAIN}" \
  SNAP_THIRD_TB_TRAIN_SHA_ENV="${THIRD_FAILURE_TENSORBOARD_TRAIN_SHA256}" \
  SNAP_THIRD_TB_VAL_ENV="${THIRD_FAILURE_TENSORBOARD_VAL}" \
  SNAP_THIRD_TB_VAL_SHA_ENV="${THIRD_FAILURE_TENSORBOARD_VAL_SHA256}" \
  SNAP_THIRD_CODE_MANIFEST_ENV="${THIRD_CODE_MANIFEST}" \
  SNAP_THIRD_CODE_MANIFEST_SHA_ENV="${THIRD_CODE_MANIFEST_SHA256}" \
  SNAP_THIRD_INPUT_MANIFEST_ENV="${THIRD_INPUT_MANIFEST}" \
  SNAP_THIRD_INPUT_MANIFEST_SHA_ENV="${THIRD_INPUT_MANIFEST_SHA256}" \
  SNAP_OWNER_UID_ENV="${SNAPSHOT_OWNER_UID}" \
  SNAP_OWNER_GID_ENV="${SNAPSHOT_OWNER_GID}" \
  "${PYTHON_BIN}" - <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import pathlib
import shutil
import stat


source_root = pathlib.Path(os.environ["SNAP_SOURCE_ROOT_ENV"]).resolve()
code_root = pathlib.Path(os.environ["SNAP_CODE_ROOT_ENV"])
input_root = pathlib.Path(os.environ["SNAP_INPUT_ROOT_ENV"])
owner_uid = int(os.environ["SNAP_OWNER_UID_ENV"])
owner_gid = int(os.environ["SNAP_OWNER_GID_ENV"])


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_regular_bytes(path):
    path = pathlib.Path(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("snapshot source is not regular: {}".format(path))
        digest = hashlib.sha256()
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise SystemExit("snapshot source changed while reading: {}".format(path))
    return b"".join(chunks), digest.hexdigest()


def copy_verified(source, destination, expected_sha, expected_size=None):
    source = pathlib.Path(source)
    destination = pathlib.Path(destination)
    if source.is_symlink():
        raise SystemExit("snapshot source is a symlink: {}".format(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(str(source), source_flags)
    destination_fd = None
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("snapshot source is not regular: {}".format(source))
        destination_fd = os.open(
            str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        digest = hashlib.sha256()
        written = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            offset = 0
            while offset < len(chunk):
                offset += os.write(destination_fd, chunk[offset:])
            written += len(chunk)
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        output_info = os.fstat(destination_fd)
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha:
        raise SystemExit("snapshot source SHA changed: {}".format(source))
    if expected_size is not None and int(expected_size) != written:
        raise SystemExit("snapshot source size changed: {}".format(source))
    if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise SystemExit("snapshot source changed while copying: {}".format(source))
    if output_info.st_size != written:
        raise SystemExit("snapshot output was truncated: {}".format(destination))
    if output_info.st_dev == before.st_dev and output_info.st_ino == before.st_ino:
        raise SystemExit("snapshot output aliases source inode: {}".format(source))
    return {
        "sha256": actual_sha,
        "size": written,
        "source_dev": before.st_dev,
        "source_ino": before.st_ino,
    }


def write_json_exclusive(path, payload):
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chown(str(path), owner_uid, owner_gid)
    os.chmod(str(path), 0o444)
    return hashlib.sha256(raw).hexdigest()


reviewed_raw, reviewed_sha = read_regular_bytes(
    os.environ["SNAP_REVIEWED_MANIFEST_ENV"]
)
if reviewed_sha != os.environ["SNAP_REVIEWED_MANIFEST_SHA_ENV"]:
    raise SystemExit("reviewed runtime-code manifest SHA changed")
reviewed = json.loads(reviewed_raw.decode("utf-8"))
if reviewed.get("schema") != "mcln-fpr-tv-reviewed-runtime-code-v1":
    raise SystemExit("reviewed runtime-code manifest schema changed")
if reviewed.get("source_root") != str(source_root):
    raise SystemExit("reviewed runtime-code source root changed")
reviewed_files = reviewed.get("files")
if not isinstance(reviewed_files, dict) or not reviewed_files:
    raise SystemExit("reviewed runtime-code manifest lacks files")
if reviewed.get("file_count") != len(reviewed_files):
    raise SystemExit("reviewed runtime-code file count changed")


code_temporary = code_root.with_name(code_root.name + ".tmp.{}".format(os.getpid()))
if code_root.exists() or code_temporary.exists():
    raise SystemExit("code snapshot path already exists")
code_temporary.mkdir()
try:
    code_records = {}
    for relative, source_record in sorted(reviewed_files.items()):
        components = relative.split("/") if isinstance(relative, str) else []
        if (
                not components
                or os.path.isabs(relative)
                or any(part in ("", ".", "..") for part in components)):
            raise SystemExit("unsafe reviewed code path: {!r}".format(relative))
        destination = code_temporary.joinpath(*components)
        record = copy_verified(
            source_root.joinpath(*components),
            destination,
            source_record.get("sha256"),
            source_record.get("size"),
        )
        record.update({"mode": "0444", "uid": owner_uid, "gid": owner_gid})
        os.chown(str(destination), owner_uid, owner_gid)
        os.chmod(str(destination), 0o444)
        code_records[relative] = record
    directories = {}
    for current, names, _files in os.walk(str(code_temporary)):
        names.sort()
        current_path = pathlib.Path(current)
        relative = current_path.relative_to(code_temporary).as_posix()
        relative = "." if relative == "." else relative
        directories[relative] = {
            "mode": "0555", "uid": owner_uid, "gid": owner_gid,
        }
    code_manifest_sha = write_json_exclusive(
        code_temporary / "CODE_MANIFEST.json",
        {
            "schema": "mcln-tier-hard-query-code-snapshot-v3",
            "source_root": str(source_root),
            "reviewed_manifest_sha256": reviewed_sha,
            "files": code_records,
            "directories": directories,
            "writable_runtime_directories": [],
        },
    )
    for current, _names, _files in os.walk(str(code_temporary), topdown=False):
        os.chown(current, owner_uid, owner_gid)
        os.chmod(current, 0o555)
    fsync_directory(code_temporary)
    os.rename(str(code_temporary), str(code_root))
    fsync_directory(code_root.parent)
except BaseException:
    if code_temporary.exists():
        for current, _names, _files in os.walk(str(code_temporary)):
            os.chmod(current, 0o755)
        shutil.rmtree(str(code_temporary))
    raise


input_sources = {
    "e57_checkpoint.pth": (
        os.environ["SNAP_E57_ENV"], os.environ["SNAP_E57_SHA_ENV"]),
    "gf_detector_l6o256.pth": (
        os.environ["SNAP_GF_ENV"], os.environ["SNAP_GF_SHA_ENV"]),
    "nr3d_train_input_manifest_v1.json": (
        os.environ["SNAP_DATA_MANIFEST_ENV"],
        os.environ["SNAP_DATA_MANIFEST_SHA_ENV"]),
    "reviewed_runtime_code_manifest.json": (
        os.environ["SNAP_REVIEWED_MANIFEST_ENV"],
        os.environ["SNAP_REVIEWED_MANIFEST_SHA_ENV"]),
    "mcln_landlock_snapshot_exec.py": (
        os.environ["SNAP_LANDLOCK_ENV"],
        os.environ["SNAP_LANDLOCK_SHA_ENV"]),
    "mcln_fpr_audit_static_exec.x86_64": (
        os.environ["SNAP_STATIC_EXEC_ENV"],
        os.environ["SNAP_STATIC_EXEC_SHA_ENV"]),
    "mcln_fpr_audit_static_exec.c": (
        os.environ["SNAP_STATIC_SOURCE_ENV"],
        os.environ["SNAP_STATIC_SOURCE_SHA_ENV"]),
    "original_failed_consumed_provenance.json": (
        os.environ["SNAP_ORIGINAL_PROVENANCE_ENV"],
        os.environ["SNAP_ORIGINAL_PROVENANCE_SHA_ENV"]),
    "original_failed_formal_command.json": (
        os.environ["SNAP_ORIGINAL_COMMAND_ENV"],
        os.environ["SNAP_ORIGINAL_COMMAND_SHA_ENV"]),
    "original_failed_launch.log": (
        os.environ["SNAP_ORIGINAL_LOG_ENV"],
        os.environ["SNAP_ORIGINAL_LOG_SHA_ENV"]),
    "prior_failed_consumed_provenance.json": (
        os.environ["SNAP_PRIOR_PROVENANCE_ENV"],
        os.environ["SNAP_PRIOR_PROVENANCE_SHA_ENV"]),
    "prior_failed_formal_command.json": (
        os.environ["SNAP_PRIOR_COMMAND_ENV"],
        os.environ["SNAP_PRIOR_COMMAND_SHA_ENV"]),
    "prior_failed_launch.log": (
        os.environ["SNAP_PRIOR_LOG_ENV"],
        os.environ["SNAP_PRIOR_LOG_SHA_ENV"]),
    "prior_failed_empty_train.log": (
        os.environ["SNAP_PRIOR_TRAIN_LOG_ENV"],
        os.environ["SNAP_PRIOR_TRAIN_LOG_SHA_ENV"]),
    "second_failed_consumed_provenance.json": (
        os.environ["SNAP_SECOND_PROVENANCE_ENV"],
        os.environ["SNAP_SECOND_PROVENANCE_SHA_ENV"]),
    "second_failed_formal_command.json": (
        os.environ["SNAP_SECOND_COMMAND_ENV"],
        os.environ["SNAP_SECOND_COMMAND_SHA_ENV"]),
    "second_failed_launch.log": (
        os.environ["SNAP_SECOND_LOG_ENV"],
        os.environ["SNAP_SECOND_LOG_SHA_ENV"]),
    "second_failed_config.json": (
        os.environ["SNAP_SECOND_CONFIG_ENV"],
        os.environ["SNAP_SECOND_CONFIG_SHA_ENV"]),
    "second_failed_train.log": (
        os.environ["SNAP_SECOND_TRAIN_LOG_ENV"],
        os.environ["SNAP_SECOND_TRAIN_LOG_SHA_ENV"]),
    "second_failed_tensorboard_train.events": (
        os.environ["SNAP_SECOND_TB_TRAIN_ENV"],
        os.environ["SNAP_SECOND_EMPTY_SHA_ENV"]),
    "second_failed_tensorboard_val.events": (
        os.environ["SNAP_SECOND_TB_VAL_ENV"],
        os.environ["SNAP_SECOND_EMPTY_SHA_ENV"]),
    "second_failed_code_manifest.json": (
        os.environ["SNAP_SECOND_CODE_MANIFEST_ENV"],
        os.environ["SNAP_SECOND_CODE_MANIFEST_SHA_ENV"]),
    "second_failed_input_manifest.json": (
        os.environ["SNAP_SECOND_INPUT_MANIFEST_ENV"],
        os.environ["SNAP_SECOND_INPUT_MANIFEST_SHA_ENV"]),
    "third_failed_collate_evidence.json": (
        os.environ["SNAP_THIRD_EVIDENCE_ENV"],
        os.environ["SNAP_THIRD_EVIDENCE_SHA_ENV"]),
    "third_failed_first_batch_replay.py": (
        os.environ["SNAP_THIRD_REPLAY_SCRIPT_ENV"],
        os.environ["SNAP_THIRD_REPLAY_SCRIPT_SHA_ENV"]),
    "third_failed_first_batch_replay_receipt.json": (
        os.environ["SNAP_THIRD_REPLAY_RECEIPT_ENV"],
        os.environ["SNAP_THIRD_REPLAY_RECEIPT_SHA_ENV"]),
    "third_failed_consumed_provenance.json": (
        os.environ["SNAP_THIRD_PROVENANCE_ENV"],
        os.environ["SNAP_THIRD_PROVENANCE_SHA_ENV"]),
    "third_failed_formal_command.json": (
        os.environ["SNAP_THIRD_COMMAND_ENV"],
        os.environ["SNAP_THIRD_COMMAND_SHA_ENV"]),
    "third_failed_launch.log": (
        os.environ["SNAP_THIRD_LOG_ENV"],
        os.environ["SNAP_THIRD_LOG_SHA_ENV"]),
    "third_failed_config.json": (
        os.environ["SNAP_THIRD_CONFIG_ENV"],
        os.environ["SNAP_THIRD_CONFIG_SHA_ENV"]),
    "third_failed_train.log": (
        os.environ["SNAP_THIRD_TRAIN_LOG_ENV"],
        os.environ["SNAP_THIRD_TRAIN_LOG_SHA_ENV"]),
    "third_failed_tensorboard_train.events": (
        os.environ["SNAP_THIRD_TB_TRAIN_ENV"],
        os.environ["SNAP_THIRD_TB_TRAIN_SHA_ENV"]),
    "third_failed_tensorboard_val.events": (
        os.environ["SNAP_THIRD_TB_VAL_ENV"],
        os.environ["SNAP_THIRD_TB_VAL_SHA_ENV"]),
    "third_failed_code_manifest.json": (
        os.environ["SNAP_THIRD_CODE_MANIFEST_ENV"],
        os.environ["SNAP_THIRD_CODE_MANIFEST_SHA_ENV"]),
    "third_failed_input_manifest.json": (
        os.environ["SNAP_THIRD_INPUT_MANIFEST_ENV"],
        os.environ["SNAP_THIRD_INPUT_MANIFEST_SHA_ENV"]),
}
input_temporary = input_root.with_name(input_root.name + ".tmp.{}".format(os.getpid()))
if input_root.exists() or input_temporary.exists():
    raise SystemExit("input snapshot path already exists")
input_temporary.mkdir()
try:
    input_records = {}
    for relative, (source, expected_sha) in sorted(input_sources.items()):
        destination = input_temporary / relative
        record = copy_verified(source, destination, expected_sha)
        record.update({"mode": "0444", "uid": owner_uid, "gid": owner_gid})
        os.chown(str(destination), owner_uid, owner_gid)
        os.chmod(str(destination), 0o444)
        input_records[relative] = record
    input_manifest_sha = write_json_exclusive(
        input_temporary / "INPUT_MANIFEST.json",
        {
            "schema": "mcln-tier-hard-query-input-snapshot-v3",
            "code_manifest_sha256": code_manifest_sha,
            "files": input_records,
        },
    )
    fsync_directory(input_temporary)
    os.chown(str(input_temporary), owner_uid, owner_gid)
    os.chmod(str(input_temporary), 0o555)
    os.rename(str(input_temporary), str(input_root))
    fsync_directory(input_root.parent)
except BaseException:
    if input_temporary.exists():
        os.chmod(str(input_temporary), 0o755)
        shutil.rmtree(str(input_temporary))
    raise

print("code_manifest_sha256={}".format(code_manifest_sha))
print("input_manifest_sha256={}".format(input_manifest_sha))
PY
)"
printf '%s\n' "${snapshot_build_output}"
readonly CODE_MANIFEST_SHA256="$(
  printf '%s\n' "${snapshot_build_output}" | \
    awk -F= '$1 == "code_manifest_sha256" {print $2}'
)"
readonly INPUT_MANIFEST_SHA256="$(
  printf '%s\n' "${snapshot_build_output}" | \
    awk -F= '$1 == "input_manifest_sha256" {print $2}'
)"
if [[ ! "${CODE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
   [[ ! "${INPUT_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo 'snapshot builder did not emit valid manifest SHAs' >&2
  exit 8
fi

mkdir -p "${RUNTIME_HOME}/hf" "${RUNTIME_HOME}/xdg" "${RUNTIME_HOME}/torch"
chmod 0700 "${RUNTIME_HOME}" \
  "${RUNTIME_HOME}/hf" "${RUNTIME_HOME}/xdg" "${RUNTIME_HOME}/torch"
clean_executor_env=(
  "HOME=${RUNTIME_HOME}"
  'USER=root' 'LOGNAME=root' 'LANG=C.UTF-8' 'LC_ALL=C.UTF-8'
  'PATH=/root/miniconda3/envs/bdetr/bin:/usr/local/cuda/bin:/usr/bin:/bin'
  'PYTHONNOUSERSITE=1' 'PYTHONDONTWRITEBYTECODE=1'
)

verify_no_inheritable_descriptors
/usr/bin/env -i "${clean_executor_env[@]}" \
  "${PYTHON_BIN}" "${INPUT_LANDLOCK_EXECUTOR}" \
  --code-root "${CODE_SNAPSHOT}" \
  --code-manifest-sha256 "${CODE_MANIFEST_SHA256}" \
  --input-root "${INPUT_SNAPSHOT}" \
  --input-manifest-sha256 "${INPUT_MANIFEST_SHA256}" \
  --verify-only
train_args=(
  --num_target 256 --sampling kps
  --num_encoder_layers 3 --num_decoder_layers 6
  --self_position_embedding loc_learned --query_points_obj_topk 4
  --use_color --weight_decay 0.0005
  --data_root "${DATA_ROOT}/"
  --val_freq 1 --batch_size "${BATCH_SIZE}"
  --num_workers 4 --dataloader_prefetch_factor 2 --persistent_train_workers
  --save_freq 1 --print_freq 20 --rng_seed 0
  --lr_backbone 1e-3 --lr 1e-4 --lr_decay_epochs 150 --warmup-epoch -1
  --dataset "${DATASET}" --test_dataset "${DATASET}"
  --joint_det --butd_cls
  --max_train_batches "${AUDIT_BATCHES}"
  --gradient_accumulation_steps 1
  --local_rank "${LOCAL_RANK}"
  --detect_intermediate --use_soft_token_loss --use_contrastive_align
  --log_dir "${RUNTIME_OUTPUT}"
  --pp_checkpoint "${INPUT_GROUPFREE}"
  --self_attend --skip_missing_superpoints
  --checkpoint_path "${INPUT_CHECKPOINT}"
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
  --expected_eval_sample_count 7899
)

formal_env=(
  "HOME=${RUNTIME_HOME}"
  'USER=root' 'LOGNAME=root' 'LANG=C.UTF-8' 'LC_ALL=C.UTF-8'
  'PATH=/root/miniconda3/envs/bdetr/bin:/usr/local/cuda/bin:/usr/bin:/bin'
  'PYTHONNOUSERSITE=1' 'PYTHONDONTWRITEBYTECODE=1'
  "PYTHONPATH=${CODE_SNAPSHOT}:${CODE_SNAPSHOT}/pointnet2"
  'CUDA_VISIBLE_DEVICES=0'
  'OMP_NUM_THREADS=1' 'MKL_NUM_THREADS=1' 'OPENBLAS_NUM_THREADS=1'
  'NUMEXPR_NUM_THREADS=1' 'TOKENIZERS_PARALLELISM=false'
  "HF_HOME=${RUNTIME_HOME}/hf" "TRANSFORMERS_CACHE=${RUNTIME_HOME}/hf"
  "XDG_CACHE_HOME=${RUNTIME_HOME}/xdg" "TORCH_HOME=${RUNTIME_HOME}/torch"
  'RANK=0' 'WORLD_SIZE=1' "LOCAL_RANK=${LOCAL_RANK}" 'LOCAL_WORLD_SIZE=1'
  'LOCAL_SIZE=1' 'MASTER_ADDR=127.0.0.1' "MASTER_PORT=${MASTER_PORT}"
)
formal_command=(
  /usr/bin/env -i "${formal_env[@]}"
  "${PYTHON_BIN}" "${CODE_SNAPSHOT}/train_dist_mod.py" "${train_args[@]}"
)
executor_command=(
  /usr/bin/env -i "${clean_executor_env[@]}"
  "${PYTHON_BIN}" "${INPUT_LANDLOCK_EXECUTOR}"
  --code-root "${CODE_SNAPSHOT}"
  --code-manifest-sha256 "${CODE_MANIFEST_SHA256}"
  --input-root "${INPUT_SNAPSHOT}"
  --input-manifest-sha256 "${INPUT_MANIFEST_SHA256}"
  --allow-write "${RUNTIME_OUTPUT}"
  --allow-write /tmp --allow-write /dev/shm --allow-write /dev
  --allow-write /proc/self/task
  -- "${formal_command[@]}"
)

"${PYTHON_BIN}" - "${COMMAND_MANIFEST}" "${CODE_SNAPSHOT}" \
  "${executor_command[@]}" <<'PY'
from __future__ import print_function

import json
import os
import sys

output = sys.argv[1]
cwd = sys.argv[2]
argv = sys.argv[3:]
payload = (json.dumps({
    "schema": "mcln-fpr-tv-density-audit-command-v3",
    "cwd": cwd,
    "argv": argv,
}, indent=2, sort_keys=True) + "\n").encode("utf-8")
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
try:
    offset = 0
    while offset < len(payload):
        offset += os.write(fd, payload[offset:])
    os.fsync(fd)
finally:
    os.close(fd)
PY
chmod 0444 "${COMMAND_MANIFEST}"
readonly COMMAND_MANIFEST_SHA256="$(sha256sum "${COMMAND_MANIFEST}" | awk '{print $1}')"

"${PYTHON_BIN}" - \
  "${CONSUMED_PROVENANCE}" "${LAUNCHER_PATH}" "${EXPECTED_LAUNCHER_SHA256}" \
  "${CODE_MANIFEST}" "${CODE_MANIFEST_SHA256}" "${CODE_SNAPSHOT}" \
  "${INPUT_MANIFEST}" "${INPUT_MANIFEST_SHA256}" "${INPUT_SNAPSHOT}" \
  "${INPUT_CHECKPOINT}" "${CHECKPOINT_SHA256}" \
  "${INPUT_GROUPFREE}" "${GROUPFREE_SHA256}" \
  "${INPUT_DATA_MANIFEST}" "${DATA_MANIFEST_SHA256}" \
  "${INPUT_REVIEWED_CODE_MANIFEST}" "${REVIEWED_CODE_MANIFEST_SHA256}" \
  "${INPUT_LANDLOCK_EXECUTOR}" "${LANDLOCK_EXECUTOR_SHA256}" \
  "${INPUT_STATIC_EXECUTOR}" "${TRUSTED_STATIC_EXEC_SHA256}" \
  "${INPUT_STATIC_SOURCE}" "${TRUSTED_STATIC_SOURCE_SHA256}" \
  "${INPUT_ORIGINAL_FAILURE_PROVENANCE}" "${ORIGINAL_FAILURE_PROVENANCE_SHA256}" \
  "${INPUT_ORIGINAL_FAILURE_COMMAND}" "${ORIGINAL_FAILURE_COMMAND_SHA256}" \
  "${INPUT_ORIGINAL_FAILURE_LOG}" "${ORIGINAL_FAILURE_LOG_SHA256}" \
  "${INPUT_PRIOR_FAILURE_PROVENANCE}" "${PRIOR_FAILURE_PROVENANCE_SHA256}" \
  "${INPUT_PRIOR_FAILURE_COMMAND}" "${PRIOR_FAILURE_COMMAND_SHA256}" \
  "${INPUT_PRIOR_FAILURE_LOG}" "${PRIOR_FAILURE_LOG_SHA256}" \
  "${INPUT_PRIOR_FAILURE_TRAIN_LOG}" "${PRIOR_FAILURE_TRAIN_LOG_SHA256}" \
  "${INPUT_SECOND_FAILURE_PROVENANCE}" "${SECOND_FAILURE_PROVENANCE_SHA256}" \
  "${INPUT_SECOND_FAILURE_COMMAND}" "${SECOND_FAILURE_COMMAND_SHA256}" \
  "${INPUT_SECOND_FAILURE_LOG}" "${SECOND_FAILURE_LOG_SHA256}" \
  "${INPUT_SECOND_FAILURE_CONFIG}" "${SECOND_FAILURE_CONFIG_SHA256}" \
  "${INPUT_SECOND_FAILURE_TRAIN_LOG}" "${SECOND_FAILURE_TRAIN_LOG_SHA256}" \
  "${INPUT_SECOND_FAILURE_TENSORBOARD_TRAIN}" "${SECOND_FAILURE_EMPTY_SHA256}" \
  "${INPUT_SECOND_FAILURE_TENSORBOARD_VAL}" "${SECOND_FAILURE_EMPTY_SHA256}" \
  "${INPUT_SECOND_CODE_MANIFEST}" "${SECOND_CODE_MANIFEST_SHA256}" \
  "${INPUT_SECOND_INPUT_MANIFEST}" "${SECOND_INPUT_MANIFEST_SHA256}" \
  "${INPUT_THIRD_FAILURE_EVIDENCE_MANIFEST}" "${THIRD_FAILURE_EVIDENCE_SHA256}" \
  "${INPUT_THIRD_FIRST_BATCH_REPLAY_SCRIPT}" "${THIRD_FIRST_BATCH_REPLAY_SCRIPT_SHA256}" \
  "${INPUT_THIRD_FIRST_BATCH_REPLAY_RECEIPT}" "${THIRD_FIRST_BATCH_REPLAY_RECEIPT_SHA256}" \
  "${INPUT_THIRD_FAILURE_PROVENANCE}" "${THIRD_FAILURE_PROVENANCE_SHA256}" \
  "${INPUT_THIRD_FAILURE_COMMAND}" "${THIRD_FAILURE_COMMAND_SHA256}" \
  "${INPUT_THIRD_FAILURE_LOG}" "${THIRD_FAILURE_LOG_SHA256}" \
  "${INPUT_THIRD_FAILURE_CONFIG}" "${THIRD_FAILURE_CONFIG_SHA256}" \
  "${INPUT_THIRD_FAILURE_TRAIN_LOG}" "${THIRD_FAILURE_TRAIN_LOG_SHA256}" \
  "${INPUT_THIRD_FAILURE_TENSORBOARD_TRAIN}" "${THIRD_FAILURE_TENSORBOARD_TRAIN_SHA256}" \
  "${INPUT_THIRD_FAILURE_TENSORBOARD_VAL}" "${THIRD_FAILURE_TENSORBOARD_VAL_SHA256}" \
  "${INPUT_THIRD_CODE_MANIFEST}" "${THIRD_CODE_MANIFEST_SHA256}" \
  "${INPUT_THIRD_INPUT_MANIFEST}" "${THIRD_INPUT_MANIFEST_SHA256}" \
  "${COMMAND_MANIFEST}" "${COMMAND_MANIFEST_SHA256}" <<'PY'
from __future__ import print_function

import hashlib
import json
import os
import sys

(
    output, launcher, launcher_sha,
    code_manifest, code_manifest_sha, code_root,
    input_manifest, input_manifest_sha, input_root,
    checkpoint, checkpoint_sha, groupfree, groupfree_sha,
    data_manifest, data_manifest_sha,
    reviewed_manifest, reviewed_manifest_sha,
    landlock_executor, landlock_executor_sha,
    static_executor, static_executor_sha,
    static_source, static_source_sha,
    original_failure_provenance, original_failure_provenance_sha,
    original_failure_command, original_failure_command_sha,
    original_failure_log, original_failure_log_sha,
    prior_failure_provenance, prior_failure_provenance_sha,
    prior_failure_command, prior_failure_command_sha,
    prior_failure_log, prior_failure_log_sha,
    prior_failure_train_log, prior_failure_train_log_sha,
    second_failure_provenance, second_failure_provenance_sha,
    second_failure_command, second_failure_command_sha,
    second_failure_log, second_failure_log_sha,
    second_failure_config, second_failure_config_sha,
    second_failure_train_log, second_failure_train_log_sha,
    second_failure_tensorboard_train, second_failure_tensorboard_train_sha,
    second_failure_tensorboard_val, second_failure_tensorboard_val_sha,
    second_code_manifest, second_code_manifest_sha,
    second_input_manifest, second_input_manifest_sha,
    third_failure_evidence, third_failure_evidence_sha,
    third_replay_script, third_replay_script_sha,
    third_replay_receipt, third_replay_receipt_sha,
    third_failure_provenance, third_failure_provenance_sha,
    third_failure_command, third_failure_command_sha,
    third_failure_log, third_failure_log_sha,
    third_failure_config, third_failure_config_sha,
    third_failure_train_log, third_failure_train_log_sha,
    third_failure_tensorboard_train, third_failure_tensorboard_train_sha,
    third_failure_tensorboard_val, third_failure_tensorboard_val_sha,
    third_code_manifest, third_code_manifest_sha,
    third_input_manifest, third_input_manifest_sha,
    command, command_sha,
) = sys.argv[1:]

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

items = {
    "launcher": {"path": launcher, "sha256": launcher_sha},
    "checkpoint": {"path": checkpoint, "sha256": checkpoint_sha},
    "groupfree": {"path": groupfree, "sha256": groupfree_sha},
    "data_manifest": {"path": data_manifest, "sha256": data_manifest_sha},
    "reviewed_code_manifest": {
        "path": reviewed_manifest, "sha256": reviewed_manifest_sha,
    },
    "landlock_executor": {
        "path": landlock_executor, "sha256": landlock_executor_sha,
    },
    "static_clean_env_executor": {
        "path": static_executor, "sha256": static_executor_sha,
    },
    "static_clean_env_source": {
        "path": static_source, "sha256": static_source_sha,
    },
    "original_failed_consumed_provenance": {
        "path": original_failure_provenance,
        "sha256": original_failure_provenance_sha,
    },
    "original_failed_formal_command": {
        "path": original_failure_command,
        "sha256": original_failure_command_sha,
    },
    "original_failed_launch_log": {
        "path": original_failure_log, "sha256": original_failure_log_sha,
    },
    "prior_failed_consumed_provenance": {
        "path": prior_failure_provenance,
        "sha256": prior_failure_provenance_sha,
    },
    "prior_failed_formal_command": {
        "path": prior_failure_command,
        "sha256": prior_failure_command_sha,
    },
    "prior_failed_launch_log": {
        "path": prior_failure_log, "sha256": prior_failure_log_sha,
    },
    "prior_failed_empty_train_log": {
        "path": prior_failure_train_log,
        "sha256": prior_failure_train_log_sha,
    },
    "second_failed_consumed_provenance": {
        "path": second_failure_provenance,
        "sha256": second_failure_provenance_sha,
    },
    "second_failed_formal_command": {
        "path": second_failure_command,
        "sha256": second_failure_command_sha,
    },
    "second_failed_launch_log": {
        "path": second_failure_log,
        "sha256": second_failure_log_sha,
    },
    "second_failed_config": {
        "path": second_failure_config,
        "sha256": second_failure_config_sha,
    },
    "second_failed_train_log": {
        "path": second_failure_train_log,
        "sha256": second_failure_train_log_sha,
    },
    "second_failed_tensorboard_train": {
        "path": second_failure_tensorboard_train,
        "sha256": second_failure_tensorboard_train_sha,
    },
    "second_failed_tensorboard_val": {
        "path": second_failure_tensorboard_val,
        "sha256": second_failure_tensorboard_val_sha,
    },
    "second_failed_code_manifest": {
        "path": second_code_manifest,
        "sha256": second_code_manifest_sha,
    },
    "second_failed_input_manifest": {
        "path": second_input_manifest,
        "sha256": second_input_manifest_sha,
    },
    "third_failed_collate_evidence": {
        "path": third_failure_evidence,
        "sha256": third_failure_evidence_sha,
    },
    "third_failed_first_batch_replay_script": {
        "path": third_replay_script,
        "sha256": third_replay_script_sha,
    },
    "third_failed_first_batch_replay_receipt": {
        "path": third_replay_receipt,
        "sha256": third_replay_receipt_sha,
    },
    "third_failed_consumed_provenance": {
        "path": third_failure_provenance,
        "sha256": third_failure_provenance_sha,
    },
    "third_failed_formal_command": {
        "path": third_failure_command,
        "sha256": third_failure_command_sha,
    },
    "third_failed_launch_log": {
        "path": third_failure_log,
        "sha256": third_failure_log_sha,
    },
    "third_failed_config": {
        "path": third_failure_config,
        "sha256": third_failure_config_sha,
    },
    "third_failed_train_log": {
        "path": third_failure_train_log,
        "sha256": third_failure_train_log_sha,
    },
    "third_failed_tensorboard_train": {
        "path": third_failure_tensorboard_train,
        "sha256": third_failure_tensorboard_train_sha,
    },
    "third_failed_tensorboard_val": {
        "path": third_failure_tensorboard_val,
        "sha256": third_failure_tensorboard_val_sha,
    },
    "third_failed_code_manifest": {
        "path": third_code_manifest,
        "sha256": third_code_manifest_sha,
    },
    "third_failed_input_manifest": {
        "path": third_input_manifest,
        "sha256": third_input_manifest_sha,
    },
    "command": {"path": command, "sha256": command_sha},
}
for label, item in items.items():
    if sha256_file(item["path"]) != item["sha256"]:
        raise SystemExit(label + " changed before launch")
payload = (json.dumps({
    "schema": "mcln-fpr-tv-density-audit-consumed-provenance-v7",
    "code": {
        "root": code_root,
        "manifest_path": code_manifest,
        "manifest_sha256": code_manifest_sha,
    },
    "input_snapshot": {
        "root": input_root,
        "manifest_path": input_manifest,
        "manifest_sha256": input_manifest_sha,
    },
    "inputs": items,
}, indent=2, sort_keys=True) + "\n").encode("utf-8")
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
try:
    offset = 0
    while offset < len(payload):
        offset += os.write(fd, payload[offset:])
    os.fsync(fd)
finally:
    os.close(fd)
PY
chmod 0444 "${CONSUMED_PROVENANCE}"
readonly CONSUMED_PROVENANCE_SHA256="$(sha256sum "${CONSUMED_PROVENANCE}" | awk '{print $1}')"

printf 'formal_audit_command='
printf '%q ' "${executor_command[@]}"
printf '\n'
cd "${CODE_SNAPSHOT}"
verify_no_inheritable_descriptors
"${executor_command[@]}"

# No decision is published until every postflight check below has passed.
cd "${ROOT_DIR}"
verify_fixed_inputs
verify_reviewed_code_manifest \
  "${ROOT_DIR}" "${REVIEWED_CODE_MANIFEST}" \
  "${REVIEWED_CODE_MANIFEST_SHA256}"
require_sha256 "${INPUT_CHECKPOINT}" "${CHECKPOINT_SHA256}" 'consumed E57 snapshot'
require_sha256 "${INPUT_GROUPFREE}" "${GROUPFREE_SHA256}" 'consumed GroupFree snapshot'
require_sha256 "${INPUT_DATA_MANIFEST}" "${DATA_MANIFEST_SHA256}" 'consumed data manifest'
require_sha256 "${INPUT_REVIEWED_CODE_MANIFEST}" "${REVIEWED_CODE_MANIFEST_SHA256}" \
  'consumed reviewed-code manifest'
require_sha256 "${INPUT_LANDLOCK_EXECUTOR}" "${LANDLOCK_EXECUTOR_SHA256}" \
  'consumed Landlock executor'
require_sha256 "${INPUT_STATIC_EXECUTOR}" "${TRUSTED_STATIC_EXEC_SHA256}" \
  'consumed static clean-env executor'
require_sha256 "${INPUT_STATIC_SOURCE}" "${TRUSTED_STATIC_SOURCE_SHA256}" \
  'consumed static clean-env source'
require_sha256 "${INPUT_ORIGINAL_FAILURE_PROVENANCE}" \
  "${ORIGINAL_FAILURE_PROVENANCE_SHA256}" 'consumed original-failure provenance'
require_sha256 "${INPUT_ORIGINAL_FAILURE_COMMAND}" \
  "${ORIGINAL_FAILURE_COMMAND_SHA256}" 'consumed original-failure command'
require_sha256 "${INPUT_ORIGINAL_FAILURE_LOG}" \
  "${ORIGINAL_FAILURE_LOG_SHA256}" 'consumed original-failure log'
require_sha256 "${INPUT_PRIOR_FAILURE_PROVENANCE}" \
  "${PRIOR_FAILURE_PROVENANCE_SHA256}" 'consumed prior-failure provenance'
require_sha256 "${INPUT_PRIOR_FAILURE_COMMAND}" \
  "${PRIOR_FAILURE_COMMAND_SHA256}" 'consumed prior-failure command'
require_sha256 "${INPUT_PRIOR_FAILURE_LOG}" \
  "${PRIOR_FAILURE_LOG_SHA256}" 'consumed prior-failure launch log'
require_sha256 "${INPUT_PRIOR_FAILURE_TRAIN_LOG}" \
  "${PRIOR_FAILURE_TRAIN_LOG_SHA256}" 'consumed prior empty train log'
require_sha256 "${INPUT_SECOND_FAILURE_PROVENANCE}" \
  "${SECOND_FAILURE_PROVENANCE_SHA256}" 'consumed second-failure provenance'
require_sha256 "${INPUT_SECOND_FAILURE_COMMAND}" \
  "${SECOND_FAILURE_COMMAND_SHA256}" 'consumed second-failure command'
require_sha256 "${INPUT_SECOND_FAILURE_LOG}" \
  "${SECOND_FAILURE_LOG_SHA256}" 'consumed second-failure launch log'
require_sha256 "${INPUT_SECOND_FAILURE_CONFIG}" \
  "${SECOND_FAILURE_CONFIG_SHA256}" 'consumed second-failure config'
require_sha256 "${INPUT_SECOND_FAILURE_TRAIN_LOG}" \
  "${SECOND_FAILURE_TRAIN_LOG_SHA256}" 'consumed second-failure train log'
require_sha256 "${INPUT_SECOND_FAILURE_TENSORBOARD_TRAIN}" \
  "${SECOND_FAILURE_EMPTY_SHA256}" 'consumed second empty train TensorBoard event'
require_sha256 "${INPUT_SECOND_FAILURE_TENSORBOARD_VAL}" \
  "${SECOND_FAILURE_EMPTY_SHA256}" 'consumed second empty val TensorBoard event'
require_sha256 "${INPUT_SECOND_CODE_MANIFEST}" \
  "${SECOND_CODE_MANIFEST_SHA256}" 'consumed second-failure code manifest'
require_sha256 "${INPUT_SECOND_INPUT_MANIFEST}" \
  "${SECOND_INPUT_MANIFEST_SHA256}" 'consumed second-failure input manifest'
require_sha256 "${INPUT_THIRD_FAILURE_EVIDENCE_MANIFEST}" \
  "${THIRD_FAILURE_EVIDENCE_SHA256}" 'consumed third-failure evidence manifest'
require_sha256 "${INPUT_THIRD_FIRST_BATCH_REPLAY_SCRIPT}" \
  "${THIRD_FIRST_BATCH_REPLAY_SCRIPT_SHA256}" \
  'consumed third-failure first-batch replay script'
require_sha256 "${INPUT_THIRD_FIRST_BATCH_REPLAY_RECEIPT}" \
  "${THIRD_FIRST_BATCH_REPLAY_RECEIPT_SHA256}" \
  'consumed third-failure first-batch replay receipt'
require_sha256 "${INPUT_THIRD_FAILURE_PROVENANCE}" \
  "${THIRD_FAILURE_PROVENANCE_SHA256}" 'consumed third-failure provenance'
require_sha256 "${INPUT_THIRD_FAILURE_COMMAND}" \
  "${THIRD_FAILURE_COMMAND_SHA256}" 'consumed third-failure command'
require_sha256 "${INPUT_THIRD_FAILURE_LOG}" \
  "${THIRD_FAILURE_LOG_SHA256}" 'consumed third-failure launch log'
require_sha256 "${INPUT_THIRD_FAILURE_CONFIG}" \
  "${THIRD_FAILURE_CONFIG_SHA256}" 'consumed third-failure config'
require_sha256 "${INPUT_THIRD_FAILURE_TRAIN_LOG}" \
  "${THIRD_FAILURE_TRAIN_LOG_SHA256}" 'consumed third-failure train log'
require_sha256 "${INPUT_THIRD_FAILURE_TENSORBOARD_TRAIN}" \
  "${THIRD_FAILURE_TENSORBOARD_TRAIN_SHA256}" \
  'consumed third-failure train TensorBoard event'
require_sha256 "${INPUT_THIRD_FAILURE_TENSORBOARD_VAL}" \
  "${THIRD_FAILURE_TENSORBOARD_VAL_SHA256}" \
  'consumed third-failure val TensorBoard event'
require_sha256 "${INPUT_THIRD_CODE_MANIFEST}" \
  "${THIRD_CODE_MANIFEST_SHA256}" 'consumed third-failure code manifest'
require_sha256 "${INPUT_THIRD_INPUT_MANIFEST}" \
  "${THIRD_INPUT_MANIFEST_SHA256}" 'consumed third-failure input manifest'
require_sha256 "${CODE_MANIFEST}" "${CODE_MANIFEST_SHA256}" 'consumed code manifest'
require_sha256 "${INPUT_MANIFEST}" "${INPUT_MANIFEST_SHA256}" 'consumed input manifest'
require_sha256 "${COMMAND_MANIFEST}" "${COMMAND_MANIFEST_SHA256}" 'formal command manifest'
require_sha256 "${CONSUMED_PROVENANCE}" "${CONSUMED_PROVENANCE_SHA256}" 'consumed provenance'
verify_no_inheritable_descriptors
/usr/bin/env -i "${clean_executor_env[@]}" \
  "${PYTHON_BIN}" "${INPUT_LANDLOCK_EXECUTOR}" \
  --code-root "${CODE_SNAPSHOT}" \
  --code-manifest-sha256 "${CODE_MANIFEST_SHA256}" \
  --input-root "${INPUT_SNAPSHOT}" \
  --input-manifest-sha256 "${INPUT_MANIFEST_SHA256}" \
  --verify-only
verify_dataset_manifest "${INPUT_DATA_MANIFEST}" "${DATA_MANIFEST_SHA256}"

mapfile -t receipts < <(
  find "${RUNTIME_OUTPUT}" -path "${RUNTIME_HOME}" -prune -o \
    -type f -name 'train_audit_receipt_epoch_58.json' -print
)
mapfile -t configs < <(
  find "${RUNTIME_OUTPUT}" -path "${RUNTIME_HOME}" -prune -o \
    -type f -name 'config.json' -print
)
mapfile -t weights < <(
  find "${RUNTIME_OUTPUT}" -path "${RUNTIME_HOME}" -prune -o \
    -type f -name '*.pth' -print
)
if ((${#receipts[@]} != 1 || ${#configs[@]} != 1)); then
  echo 'expected exactly one audit receipt and one config' >&2
  exit 8
fi
if ((${#weights[@]} != 0)); then
  echo 'bounded audit unexpectedly created a weight' >&2
  exit 8
fi

readonly RECEIPT="${receipts[0]}"
readonly CONFIG="${configs[0]}"
readonly DECISION="${AUDIT_ROOT}/audit_decision.json"
chmod 0444 "${RECEIPT}" "${CONFIG}" "${CODE_MANIFEST}" \
  "${COMMAND_MANIFEST}" "${CONSUMED_PROVENANCE}"
sync

RECEIPT_ENV="${RECEIPT}" CONFIG_ENV="${CONFIG}" DECISION_ENV="${DECISION}" \
CHECKPOINT_ENV="${INPUT_CHECKPOINT}" CHECKPOINT_SHA_ENV="${CHECKPOINT_SHA256}" \
GROUPFREE_ENV="${INPUT_GROUPFREE}" GROUPFREE_SHA_ENV="${GROUPFREE_SHA256}" \
CODE_MANIFEST_ENV="${CODE_MANIFEST}" CODE_MANIFEST_SHA_ENV="${CODE_MANIFEST_SHA256}" \
INPUT_MANIFEST_ENV="${INPUT_MANIFEST}" INPUT_MANIFEST_SHA_ENV="${INPUT_MANIFEST_SHA256}" \
COMMAND_ENV="${COMMAND_MANIFEST}" COMMAND_SHA_ENV="${COMMAND_MANIFEST_SHA256}" \
PROVENANCE_ENV="${CONSUMED_PROVENANCE}" PROVENANCE_SHA_ENV="${CONSUMED_PROVENANCE_SHA256}" \
DATA_MANIFEST_ENV="${INPUT_DATA_MANIFEST}" DATA_MANIFEST_SHA_ENV="${DATA_MANIFEST_SHA256}" \
REVIEWED_MANIFEST_ENV="${INPUT_REVIEWED_CODE_MANIFEST}" \
REVIEWED_MANIFEST_SHA_ENV="${REVIEWED_CODE_MANIFEST_SHA256}" \
LANDLOCK_ENV="${INPUT_LANDLOCK_EXECUTOR}" LANDLOCK_SHA_ENV="${LANDLOCK_EXECUTOR_SHA256}" \
STATIC_EXEC_ENV="${INPUT_STATIC_EXECUTOR}" STATIC_EXEC_SHA_ENV="${TRUSTED_STATIC_EXEC_SHA256}" \
STATIC_SOURCE_ENV="${INPUT_STATIC_SOURCE}" STATIC_SOURCE_SHA_ENV="${TRUSTED_STATIC_SOURCE_SHA256}" \
ORIGINAL_PROVENANCE_INPUT_ENV="${INPUT_ORIGINAL_FAILURE_PROVENANCE}" \
ORIGINAL_PROVENANCE_SHA_ENV="${ORIGINAL_FAILURE_PROVENANCE_SHA256}" \
ORIGINAL_COMMAND_INPUT_ENV="${INPUT_ORIGINAL_FAILURE_COMMAND}" \
ORIGINAL_COMMAND_SHA_ENV="${ORIGINAL_FAILURE_COMMAND_SHA256}" \
ORIGINAL_LOG_INPUT_ENV="${INPUT_ORIGINAL_FAILURE_LOG}" \
ORIGINAL_LOG_SHA_ENV="${ORIGINAL_FAILURE_LOG_SHA256}" \
PRIOR_PROVENANCE_INPUT_ENV="${INPUT_PRIOR_FAILURE_PROVENANCE}" \
PRIOR_PROVENANCE_SHA_ENV="${PRIOR_FAILURE_PROVENANCE_SHA256}" \
PRIOR_COMMAND_INPUT_ENV="${INPUT_PRIOR_FAILURE_COMMAND}" \
PRIOR_COMMAND_SHA_ENV="${PRIOR_FAILURE_COMMAND_SHA256}" \
PRIOR_LOG_INPUT_ENV="${INPUT_PRIOR_FAILURE_LOG}" \
PRIOR_LOG_SHA_ENV="${PRIOR_FAILURE_LOG_SHA256}" \
PRIOR_TRAIN_LOG_INPUT_ENV="${INPUT_PRIOR_FAILURE_TRAIN_LOG}" \
PRIOR_TRAIN_LOG_SHA_ENV="${PRIOR_FAILURE_TRAIN_LOG_SHA256}" \
SECOND_PROVENANCE_INPUT_ENV="${INPUT_SECOND_FAILURE_PROVENANCE}" \
SECOND_PROVENANCE_SHA_ENV="${SECOND_FAILURE_PROVENANCE_SHA256}" \
SECOND_COMMAND_INPUT_ENV="${INPUT_SECOND_FAILURE_COMMAND}" \
SECOND_COMMAND_SHA_ENV="${SECOND_FAILURE_COMMAND_SHA256}" \
SECOND_LOG_INPUT_ENV="${INPUT_SECOND_FAILURE_LOG}" \
SECOND_LOG_SHA_ENV="${SECOND_FAILURE_LOG_SHA256}" \
SECOND_CONFIG_INPUT_ENV="${INPUT_SECOND_FAILURE_CONFIG}" \
SECOND_CONFIG_SHA_ENV="${SECOND_FAILURE_CONFIG_SHA256}" \
SECOND_TRAIN_LOG_INPUT_ENV="${INPUT_SECOND_FAILURE_TRAIN_LOG}" \
SECOND_TRAIN_LOG_SHA_ENV="${SECOND_FAILURE_TRAIN_LOG_SHA256}" \
SECOND_TB_TRAIN_INPUT_ENV="${INPUT_SECOND_FAILURE_TENSORBOARD_TRAIN}" \
SECOND_TB_VAL_INPUT_ENV="${INPUT_SECOND_FAILURE_TENSORBOARD_VAL}" \
SECOND_EMPTY_SHA_ENV="${SECOND_FAILURE_EMPTY_SHA256}" \
SECOND_CODE_MANIFEST_INPUT_ENV="${INPUT_SECOND_CODE_MANIFEST}" \
SECOND_CODE_MANIFEST_SHA_ENV="${SECOND_CODE_MANIFEST_SHA256}" \
SECOND_INPUT_MANIFEST_INPUT_ENV="${INPUT_SECOND_INPUT_MANIFEST}" \
SECOND_INPUT_MANIFEST_SHA_ENV="${SECOND_INPUT_MANIFEST_SHA256}" \
THIRD_EVIDENCE_INPUT_ENV="${INPUT_THIRD_FAILURE_EVIDENCE_MANIFEST}" \
THIRD_EVIDENCE_SHA_ENV="${THIRD_FAILURE_EVIDENCE_SHA256}" \
THIRD_REPLAY_SCRIPT_INPUT_ENV="${INPUT_THIRD_FIRST_BATCH_REPLAY_SCRIPT}" \
THIRD_REPLAY_SCRIPT_SHA_ENV="${THIRD_FIRST_BATCH_REPLAY_SCRIPT_SHA256}" \
THIRD_REPLAY_RECEIPT_INPUT_ENV="${INPUT_THIRD_FIRST_BATCH_REPLAY_RECEIPT}" \
THIRD_REPLAY_RECEIPT_SHA_ENV="${THIRD_FIRST_BATCH_REPLAY_RECEIPT_SHA256}" \
THIRD_PROVENANCE_INPUT_ENV="${INPUT_THIRD_FAILURE_PROVENANCE}" \
THIRD_PROVENANCE_SHA_ENV="${THIRD_FAILURE_PROVENANCE_SHA256}" \
THIRD_COMMAND_INPUT_ENV="${INPUT_THIRD_FAILURE_COMMAND}" \
THIRD_COMMAND_SHA_ENV="${THIRD_FAILURE_COMMAND_SHA256}" \
THIRD_LOG_INPUT_ENV="${INPUT_THIRD_FAILURE_LOG}" \
THIRD_LOG_SHA_ENV="${THIRD_FAILURE_LOG_SHA256}" \
THIRD_CONFIG_INPUT_ENV="${INPUT_THIRD_FAILURE_CONFIG}" \
THIRD_CONFIG_SHA_ENV="${THIRD_FAILURE_CONFIG_SHA256}" \
THIRD_TRAIN_LOG_INPUT_ENV="${INPUT_THIRD_FAILURE_TRAIN_LOG}" \
THIRD_TRAIN_LOG_SHA_ENV="${THIRD_FAILURE_TRAIN_LOG_SHA256}" \
THIRD_TB_TRAIN_INPUT_ENV="${INPUT_THIRD_FAILURE_TENSORBOARD_TRAIN}" \
THIRD_TB_TRAIN_SHA_ENV="${THIRD_FAILURE_TENSORBOARD_TRAIN_SHA256}" \
THIRD_TB_VAL_INPUT_ENV="${INPUT_THIRD_FAILURE_TENSORBOARD_VAL}" \
THIRD_TB_VAL_SHA_ENV="${THIRD_FAILURE_TENSORBOARD_VAL_SHA256}" \
THIRD_CODE_MANIFEST_INPUT_ENV="${INPUT_THIRD_CODE_MANIFEST}" \
THIRD_CODE_MANIFEST_SHA_ENV="${THIRD_CODE_MANIFEST_SHA256}" \
THIRD_INPUT_MANIFEST_INPUT_ENV="${INPUT_THIRD_INPUT_MANIFEST}" \
THIRD_INPUT_MANIFEST_SHA_ENV="${THIRD_INPUT_MANIFEST_SHA256}" \
LAUNCHER_ENV="${LAUNCHER_PATH}" LAUNCHER_SHA_ENV="${EXPECTED_LAUNCHER_SHA256}" \
CODE_ROOT_ENV="${CODE_SNAPSHOT}" INPUT_ROOT_ENV="${INPUT_SNAPSHOT}" \
RUNTIME_OUTPUT_ENV="${RUNTIME_OUTPUT}" EXP_ENV="${EXP}" \
MIN_DEPLOYABLE_ENV="${MIN_DEPLOYABLE_ROW_RATIO}" \
MIN_DETECTOR_ENV="${MIN_DETECTOR_CANDIDATE_RATIO}" \
MIN_RELIABLE_ENV="${MIN_RELIABLE_ROW_RATIO}" \
MIN_FEASIBLE_ENV="${MIN_FEASIBLE_CANDIDATE_RATIO}" \
MIN_POSITIVE_ROW_ENV="${MIN_POSITIVE_ROW_RATIO}" \
MIN_CANDIDATE_POSITIVE_ENV="${MIN_CANDIDATE_POSITIVE_RATIO}" \
"${PYTHON_BIN}" - <<'PY'
from __future__ import print_function

import hashlib
import json
import math
import os
import pathlib

def load_json_with_sha(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()

def finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )

receipt, receipt_sha = load_json_with_sha(os.environ["RECEIPT_ENV"])
config, config_sha = load_json_with_sha(os.environ["CONFIG_ENV"])
code_manifest, code_manifest_sha = load_json_with_sha(
    os.environ["CODE_MANIFEST_ENV"]
)
input_manifest, input_manifest_sha = load_json_with_sha(
    os.environ["INPUT_MANIFEST_ENV"]
)
command, command_sha = load_json_with_sha(os.environ["COMMAND_ENV"])
provenance, provenance_sha = load_json_with_sha(os.environ["PROVENANCE_ENV"])
data_manifest, data_manifest_sha = load_json_with_sha(
    os.environ["DATA_MANIFEST_ENV"]
)
reviewed_manifest, reviewed_manifest_sha = load_json_with_sha(
    os.environ["REVIEWED_MANIFEST_ENV"]
)
if code_manifest_sha != os.environ["CODE_MANIFEST_SHA_ENV"]:
    raise SystemExit("code manifest SHA changed before decision")
if input_manifest_sha != os.environ["INPUT_MANIFEST_SHA_ENV"]:
    raise SystemExit("input manifest SHA changed before decision")
if command_sha != os.environ["COMMAND_SHA_ENV"]:
    raise SystemExit("command manifest SHA changed before decision")
if provenance_sha != os.environ["PROVENANCE_SHA_ENV"]:
    raise SystemExit("provenance SHA changed before decision")
if data_manifest_sha != os.environ["DATA_MANIFEST_SHA_ENV"]:
    raise SystemExit("data manifest SHA changed before decision")
if reviewed_manifest_sha != os.environ["REVIEWED_MANIFEST_SHA_ENV"]:
    raise SystemExit("reviewed-code manifest SHA changed before decision")
if code_manifest.get("schema") != "mcln-tier-hard-query-code-snapshot-v3":
    raise SystemExit("code manifest schema changed")
if input_manifest.get("schema") != "mcln-tier-hard-query-input-snapshot-v3":
    raise SystemExit("input manifest schema changed")
if input_manifest.get("code_manifest_sha256") != code_manifest_sha:
    raise SystemExit("input snapshot is bound to another code snapshot")
if command.get("schema") != "mcln-fpr-tv-density-audit-command-v3":
    raise SystemExit("command schema changed")
if provenance.get("schema") != "mcln-fpr-tv-density-audit-consumed-provenance-v7":
    raise SystemExit("provenance schema changed")
if data_manifest.get("schema") != "mcln-nr3d-fpr-tv-audit-data-manifest-v1":
    raise SystemExit("data manifest schema changed")
if reviewed_manifest.get("schema") != "mcln-fpr-tv-reviewed-runtime-code-v1":
    raise SystemExit("reviewed-code manifest schema changed")
expected_code_provenance = {
    "root": os.environ["CODE_ROOT_ENV"],
    "manifest_path": os.environ["CODE_MANIFEST_ENV"],
    "manifest_sha256": code_manifest_sha,
}
expected_input_provenance = {
    "root": os.environ["INPUT_ROOT_ENV"],
    "manifest_path": os.environ["INPUT_MANIFEST_ENV"],
    "manifest_sha256": input_manifest_sha,
}
if provenance.get("code") != expected_code_provenance:
    raise SystemExit("consumed code provenance changed")
if provenance.get("input_snapshot") != expected_input_provenance:
    raise SystemExit("consumed input provenance changed")
expected_provenance_inputs = {
    "launcher": {
        "path": os.environ["LAUNCHER_ENV"],
        "sha256": os.environ["LAUNCHER_SHA_ENV"],
    },
    "checkpoint": {
        "path": os.environ["CHECKPOINT_ENV"],
        "sha256": os.environ["CHECKPOINT_SHA_ENV"],
    },
    "groupfree": {
        "path": os.environ["GROUPFREE_ENV"],
        "sha256": os.environ["GROUPFREE_SHA_ENV"],
    },
    "data_manifest": {
        "path": os.environ["DATA_MANIFEST_ENV"],
        "sha256": data_manifest_sha,
    },
    "reviewed_code_manifest": {
        "path": os.environ["REVIEWED_MANIFEST_ENV"],
        "sha256": reviewed_manifest_sha,
    },
    "landlock_executor": {
        "path": os.environ["LANDLOCK_ENV"],
        "sha256": os.environ["LANDLOCK_SHA_ENV"],
    },
    "static_clean_env_executor": {
        "path": os.environ["STATIC_EXEC_ENV"],
        "sha256": os.environ["STATIC_EXEC_SHA_ENV"],
    },
    "static_clean_env_source": {
        "path": os.environ["STATIC_SOURCE_ENV"],
        "sha256": os.environ["STATIC_SOURCE_SHA_ENV"],
    },
    "original_failed_consumed_provenance": {
        "path": os.environ["ORIGINAL_PROVENANCE_INPUT_ENV"],
        "sha256": os.environ["ORIGINAL_PROVENANCE_SHA_ENV"],
    },
    "original_failed_formal_command": {
        "path": os.environ["ORIGINAL_COMMAND_INPUT_ENV"],
        "sha256": os.environ["ORIGINAL_COMMAND_SHA_ENV"],
    },
    "original_failed_launch_log": {
        "path": os.environ["ORIGINAL_LOG_INPUT_ENV"],
        "sha256": os.environ["ORIGINAL_LOG_SHA_ENV"],
    },
    "prior_failed_consumed_provenance": {
        "path": os.environ["PRIOR_PROVENANCE_INPUT_ENV"],
        "sha256": os.environ["PRIOR_PROVENANCE_SHA_ENV"],
    },
    "prior_failed_formal_command": {
        "path": os.environ["PRIOR_COMMAND_INPUT_ENV"],
        "sha256": os.environ["PRIOR_COMMAND_SHA_ENV"],
    },
    "prior_failed_launch_log": {
        "path": os.environ["PRIOR_LOG_INPUT_ENV"],
        "sha256": os.environ["PRIOR_LOG_SHA_ENV"],
    },
    "prior_failed_empty_train_log": {
        "path": os.environ["PRIOR_TRAIN_LOG_INPUT_ENV"],
        "sha256": os.environ["PRIOR_TRAIN_LOG_SHA_ENV"],
    },
    "second_failed_consumed_provenance": {
        "path": os.environ["SECOND_PROVENANCE_INPUT_ENV"],
        "sha256": os.environ["SECOND_PROVENANCE_SHA_ENV"],
    },
    "second_failed_formal_command": {
        "path": os.environ["SECOND_COMMAND_INPUT_ENV"],
        "sha256": os.environ["SECOND_COMMAND_SHA_ENV"],
    },
    "second_failed_launch_log": {
        "path": os.environ["SECOND_LOG_INPUT_ENV"],
        "sha256": os.environ["SECOND_LOG_SHA_ENV"],
    },
    "second_failed_config": {
        "path": os.environ["SECOND_CONFIG_INPUT_ENV"],
        "sha256": os.environ["SECOND_CONFIG_SHA_ENV"],
    },
    "second_failed_train_log": {
        "path": os.environ["SECOND_TRAIN_LOG_INPUT_ENV"],
        "sha256": os.environ["SECOND_TRAIN_LOG_SHA_ENV"],
    },
    "second_failed_tensorboard_train": {
        "path": os.environ["SECOND_TB_TRAIN_INPUT_ENV"],
        "sha256": os.environ["SECOND_EMPTY_SHA_ENV"],
    },
    "second_failed_tensorboard_val": {
        "path": os.environ["SECOND_TB_VAL_INPUT_ENV"],
        "sha256": os.environ["SECOND_EMPTY_SHA_ENV"],
    },
    "second_failed_code_manifest": {
        "path": os.environ["SECOND_CODE_MANIFEST_INPUT_ENV"],
        "sha256": os.environ["SECOND_CODE_MANIFEST_SHA_ENV"],
    },
    "second_failed_input_manifest": {
        "path": os.environ["SECOND_INPUT_MANIFEST_INPUT_ENV"],
        "sha256": os.environ["SECOND_INPUT_MANIFEST_SHA_ENV"],
    },
    "third_failed_collate_evidence": {
        "path": os.environ["THIRD_EVIDENCE_INPUT_ENV"],
        "sha256": os.environ["THIRD_EVIDENCE_SHA_ENV"],
    },
    "third_failed_first_batch_replay_script": {
        "path": os.environ["THIRD_REPLAY_SCRIPT_INPUT_ENV"],
        "sha256": os.environ["THIRD_REPLAY_SCRIPT_SHA_ENV"],
    },
    "third_failed_first_batch_replay_receipt": {
        "path": os.environ["THIRD_REPLAY_RECEIPT_INPUT_ENV"],
        "sha256": os.environ["THIRD_REPLAY_RECEIPT_SHA_ENV"],
    },
    "third_failed_consumed_provenance": {
        "path": os.environ["THIRD_PROVENANCE_INPUT_ENV"],
        "sha256": os.environ["THIRD_PROVENANCE_SHA_ENV"],
    },
    "third_failed_formal_command": {
        "path": os.environ["THIRD_COMMAND_INPUT_ENV"],
        "sha256": os.environ["THIRD_COMMAND_SHA_ENV"],
    },
    "third_failed_launch_log": {
        "path": os.environ["THIRD_LOG_INPUT_ENV"],
        "sha256": os.environ["THIRD_LOG_SHA_ENV"],
    },
    "third_failed_config": {
        "path": os.environ["THIRD_CONFIG_INPUT_ENV"],
        "sha256": os.environ["THIRD_CONFIG_SHA_ENV"],
    },
    "third_failed_train_log": {
        "path": os.environ["THIRD_TRAIN_LOG_INPUT_ENV"],
        "sha256": os.environ["THIRD_TRAIN_LOG_SHA_ENV"],
    },
    "third_failed_tensorboard_train": {
        "path": os.environ["THIRD_TB_TRAIN_INPUT_ENV"],
        "sha256": os.environ["THIRD_TB_TRAIN_SHA_ENV"],
    },
    "third_failed_tensorboard_val": {
        "path": os.environ["THIRD_TB_VAL_INPUT_ENV"],
        "sha256": os.environ["THIRD_TB_VAL_SHA_ENV"],
    },
    "third_failed_code_manifest": {
        "path": os.environ["THIRD_CODE_MANIFEST_INPUT_ENV"],
        "sha256": os.environ["THIRD_CODE_MANIFEST_SHA_ENV"],
    },
    "third_failed_input_manifest": {
        "path": os.environ["THIRD_INPUT_MANIFEST_INPUT_ENV"],
        "sha256": os.environ["THIRD_INPUT_MANIFEST_SHA_ENV"],
    },
    "command": {
        "path": os.environ["COMMAND_ENV"],
        "sha256": command_sha,
    },
}
if provenance.get("inputs") != expected_provenance_inputs:
    raise SystemExit("consumed input provenance changed")
if command.get("cwd") != os.environ["CODE_ROOT_ENV"]:
    raise SystemExit("formal command cwd changed")
if receipt.get("schema") != "mcln-train-loss-epoch-v1":
    raise SystemExit("audit receipt schema changed")
if receipt.get("epoch") != 58 or receipt.get("batch_count") != 100:
    raise SystemExit("audit did not process exact E58/100 batches")
if receipt.get("max_train_batches") != 100:
    raise SystemExit("bounded audit flag changed")
if receipt.get("checkpoint_path") != os.environ["CHECKPOINT_ENV"]:
    raise SystemExit("audit consumed a different checkpoint snapshot")

expected_config = {
    "batch_size": 16,
    "dataset": ["nr3d"],
    "test_dataset": "nr3d",
    "joint_det": True,
    "butd_cls": True,
    "rng_seed": 0,
    "eval": False,
    "model": "MCLN",
    "start_epoch": 58,
    "max_epoch": 58,
    "max_train_batches": 100,
    "gradient_accumulation_steps": 1,
    "local_rank": 0,
    "use_source_choice_selector": True,
    "eval_use_selector_choice_scores": True,
    "source_choice_selector_sources": (
        "default,default_rank_blend_contrastive010"
    ),
    "source_choice_selector_default_source": "default",
    "source_choice_selector_hidden_dim": 288,
    "source_choice_selector_lr": 1.25e-4,
    "source_choice_selector_loss_weight": 0.5,
    "source_choice_selector_choice_target": (
        "precision_gain_default_sourcewise_focal_bce"
    ),
    "source_choice_selector_min_iou_gap": 0.03,
    "use_parent_relative_text_verifier": True,
    "parent_relative_text_verifier_train_only": True,
    "parent_relative_text_verifier_top_k": 5,
    "parent_relative_text_verifier_max_candidates": 10,
    "parent_relative_text_verifier_hidden_dim": 256,
    "parent_relative_text_verifier_heads": 4,
    "parent_relative_text_verifier_dropout": 0.1,
    "parent_relative_text_verifier_max_parent_score_gap": 0.25,
    "parent_relative_text_verifier_promotion_margin": 0.0001,
    "parent_relative_text_verifier_min_parse_confidence": 0.5,
    "parent_relative_text_verifier_min_anchor_mass": 0.5,
    "parent_relative_text_verifier_promotion_epsilon": 0.0001,
    "parent_relative_text_verifier_lr": 0.0003,
    "parent_relative_text_verifier_loss_weight": 1.0,
    "parent_relative_text_verifier_positive_margin": 0.25,
    "parent_relative_text_verifier_neutral_margin": 0.25,
    "expected_eval_sample_count": 7899,
}
for key, value in expected_config.items():
    if type(config.get(key)) is not type(value) or config.get(key) != value:
        raise SystemExit("formal config changed: " + key)
if config.get("checkpoint_path") != os.environ["CHECKPOINT_ENV"]:
    raise SystemExit("config checkpoint snapshot changed")
if config.get("pp_checkpoint") != os.environ["GROUPFREE_ENV"]:
    raise SystemExit("config GroupFree snapshot changed")
if config.get("exp") != os.environ["EXP_ENV"]:
    raise SystemExit("config experiment changed")
runtime_output = pathlib.Path(os.environ["RUNTIME_OUTPUT_ENV"])
expected_exp_root = runtime_output / "nr3d" / os.environ["EXP_ENV"]
config_path = pathlib.Path(os.environ["CONFIG_ENV"])
receipt_path = pathlib.Path(os.environ["RECEIPT_ENV"])
run_dir = config_path.parent
if runtime_output.resolve() != runtime_output:
    raise SystemExit("runtime output path is not canonical")
if run_dir.parent != expected_exp_root:
    raise SystemExit("normalized run directory escaped dataset/experiment root")
if not run_dir.name.isdigit() or not run_dir.name:
    raise SystemExit("normalized run directory lacks a numeric timestamp")
if config_path != run_dir / "config.json":
    raise SystemExit("config path escaped normalized run directory")
if receipt_path != run_dir / "train_audit_receipt_epoch_58.json":
    raise SystemExit("receipt path escaped normalized run directory")
for component in (
        runtime_output, runtime_output / "nr3d", expected_exp_root, run_dir):
    if component.is_symlink() or component.resolve() != component:
        raise SystemExit("normalized run directory contains a symlink or escape")
if config.get("log_dir") != str(run_dir):
    raise SystemExit("config did not record its normalized run directory")
for forbidden in (
    "use_source_moe", "use_sacr_source", "use_sacr_score_refiner",
    "source_moe_train_only", "sacr_score_refiner_train_only",
    "tier_hard_query_aux_loss_weight",
    "relation_counterfactual_aux_loss_weight",
):
    value = config.get(forbidden, False)
    if value not in (False, 0, 0.0, None):
        raise SystemExit("formal config enabled excluded branch: " + forbidden)

losses = receipt.get("loss_means")
stats = receipt.get("stat_means")
if not isinstance(losses, dict) or not isinstance(stats, dict):
    raise SystemExit("audit receipt lacks loss/stat means")
for section in (losses, stats):
    if any(not finite_number(value) for value in section.values()):
        raise SystemExit("audit contains non-finite statistics")
for name in ("total_loss", "parent_relative_text_verifier_loss"):
    if name not in losses or float(losses[name]) <= 0.0:
        raise SystemExit("missing or non-positive audit loss: " + name)

required_minima = {
    "grad_norm": 0.0,
    "parent_relative_text_verifier_deployable_row_ratio": float(
        os.environ["MIN_DEPLOYABLE_ENV"]
    ),
    "parent_relative_text_verifier_detector_candidate_ratio": float(
        os.environ["MIN_DETECTOR_ENV"]
    ),
    "parent_relative_text_verifier_reliable_row_ratio": float(
        os.environ["MIN_RELIABLE_ENV"]
    ),
    "parent_relative_text_verifier_feasible_candidate_ratio": float(
        os.environ["MIN_FEASIBLE_ENV"]
    ),
    "parent_relative_text_verifier_positive_row_ratio": float(
        os.environ["MIN_POSITIVE_ROW_ENV"]
    ),
    "parent_relative_text_verifier_candidate_positive_ratio": float(
        os.environ["MIN_CANDIDATE_POSITIVE_ENV"]
    ),
}
required_losses = [
    "parent_relative_text_verifier_action_loss",
    "parent_relative_text_verifier_repair_loss",
    "parent_relative_text_verifier_break_loss",
    "parent_relative_text_verifier_iou_loss",
    "parent_relative_text_verifier_reliability_loss",
    "parent_relative_text_verifier_positive_action_loss",
    "parent_relative_text_verifier_preserve_loss",
]
required_ratios = [
    "parent_relative_text_verifier_switch_ratio",
    "parent_relative_text_verifier_fallback_ratio",
    "parent_relative_text_verifier_learned_reliable_candidate_ratio",
    "parent_relative_text_verifier_eligible_candidate_ratio",
    "parent_relative_text_verifier_fix025_ratio",
    "parent_relative_text_verifier_break025_ratio",
    "parent_relative_text_verifier_fix050_ratio",
    "parent_relative_text_verifier_break050_ratio",
    "parent_relative_text_verifier_parent_acc025",
    "parent_relative_text_verifier_parent_acc050",
    "parent_relative_text_verifier_selected_acc025",
    "parent_relative_text_verifier_selected_acc050",
]
missing = sorted(
    (set(required_minima) | set(required_losses) | set(required_ratios))
    - set(stats)
)
if missing:
    raise SystemExit("missing audit stats: " + ",".join(missing))
for name in required_losses:
    if float(stats[name]) < 0.0:
        raise SystemExit("negative verifier loss statistic: " + name)
for name in required_ratios:
    value = float(stats[name])
    if value < 0.0 or value > 1.0:
        raise SystemExit("ratio/accuracy outside [0,1]: " + name)
for name, minimum in required_minima.items():
    value = float(stats[name])
    if name != "grad_norm":
        if minimum < 0.0 or minimum > 1.0:
            raise SystemExit("density threshold outside [0,1]: " + name)
        if value < 0.0 or value > 1.0:
            raise SystemExit("density ratio outside [0,1]: " + name)
switch = float(stats["parent_relative_text_verifier_switch_ratio"])
fallback = float(stats["parent_relative_text_verifier_fallback_ratio"])
if abs((switch + fallback) - 1.0) > 1e-6:
    raise SystemExit("switch/fallback probabilities are inconsistent")

gate_failures = []
for name, minimum in required_minima.items():
    value = float(stats[name])
    if name == "grad_norm":
        if value <= minimum:
            gate_failures.append(name + "<=0")
    elif value < minimum:
        gate_failures.append("{}<{}".format(name, minimum))

decision = {
    "schema": "mcln-fpr-tv-density-audit-decision-v7",
    "audit_only": True,
    "long_training_authorized": False,
    "next_stage": (
        "scene_disjoint_audit_only" if not gate_failures else "stop"
    ),
    "density_gate_pass": not gate_failures,
    "gate_failures": gate_failures,
    "receipt": {
        "path": os.environ["RECEIPT_ENV"],
        "sha256": receipt_sha,
    },
    "config": {
        "path": os.environ["CONFIG_ENV"],
        "sha256": config_sha,
    },
    "launcher": {
        "path": os.environ["LAUNCHER_ENV"],
        "sha256": os.environ["LAUNCHER_SHA_ENV"],
    },
    "code_manifest": {
        "path": os.environ["CODE_MANIFEST_ENV"],
        "sha256": code_manifest_sha,
    },
    "input_manifest": {
        "path": os.environ["INPUT_MANIFEST_ENV"],
        "sha256": input_manifest_sha,
    },
    "reviewed_code_manifest": {
        "path": os.environ["REVIEWED_MANIFEST_ENV"],
        "sha256": reviewed_manifest_sha,
        "file_count": reviewed_manifest["file_count"],
        "total_size": reviewed_manifest["total_size"],
    },
    "landlock_executor": {
        "path": os.environ["LANDLOCK_ENV"],
        "sha256": os.environ["LANDLOCK_SHA_ENV"],
    },
    "static_clean_env_executor": {
        "path": os.environ["STATIC_EXEC_ENV"],
        "sha256": os.environ["STATIC_EXEC_SHA_ENV"],
    },
    "static_clean_env_source": {
        "path": os.environ["STATIC_SOURCE_ENV"],
        "sha256": os.environ["STATIC_SOURCE_SHA_ENV"],
    },
    "recovery_from_failed_startups": {
        "initial_local_rank_failure": {
            "consumed_provenance": {
                "path": os.environ["ORIGINAL_PROVENANCE_INPUT_ENV"],
                "sha256": os.environ["ORIGINAL_PROVENANCE_SHA_ENV"],
            },
            "formal_command": {
                "path": os.environ["ORIGINAL_COMMAND_INPUT_ENV"],
                "sha256": os.environ["ORIGINAL_COMMAND_SHA_ENV"],
            },
            "launch_log": {
                "path": os.environ["ORIGINAL_LOG_INPUT_ENV"],
                "sha256": os.environ["ORIGINAL_LOG_SHA_ENV"],
            },
            "cause": "argparse_local_rank_default_1_on_single_gpu",
            "batches": 0,
            "receipts": 0,
            "weights": 0,
        },
        "tensorboard_output_failure": {
            "consumed_provenance": {
                "path": os.environ["PRIOR_PROVENANCE_INPUT_ENV"],
                "sha256": os.environ["PRIOR_PROVENANCE_SHA_ENV"],
            },
            "formal_command": {
                "path": os.environ["PRIOR_COMMAND_INPUT_ENV"],
                "sha256": os.environ["PRIOR_COMMAND_SHA_ENV"],
            },
            "launch_log": {
                "path": os.environ["PRIOR_LOG_INPUT_ENV"],
                "sha256": os.environ["PRIOR_LOG_SHA_ENV"],
            },
            "empty_train_log": {
                "path": os.environ["PRIOR_TRAIN_LOG_INPUT_ENV"],
                "sha256": os.environ["PRIOR_TRAIN_LOG_SHA_ENV"],
            },
            "cause": "tensorboard_writer_ignored_absolute_log_dir",
            "batches": 0,
            "receipts": 0,
            "weights": 0,
        },
        "normalized_log_dir_postflight_failure": {
            "consumed_provenance": {
                "path": os.environ["SECOND_PROVENANCE_INPUT_ENV"],
                "sha256": os.environ["SECOND_PROVENANCE_SHA_ENV"],
            },
            "formal_command": {
                "path": os.environ["SECOND_COMMAND_INPUT_ENV"],
                "sha256": os.environ["SECOND_COMMAND_SHA_ENV"],
            },
            "launch_log": {
                "path": os.environ["SECOND_LOG_INPUT_ENV"],
                "sha256": os.environ["SECOND_LOG_SHA_ENV"],
            },
            "config": {
                "path": os.environ["SECOND_CONFIG_INPUT_ENV"],
                "sha256": os.environ["SECOND_CONFIG_SHA_ENV"],
            },
            "train_log": {
                "path": os.environ["SECOND_TRAIN_LOG_INPUT_ENV"],
                "sha256": os.environ["SECOND_TRAIN_LOG_SHA_ENV"],
            },
            "tensorboard_train": {
                "path": os.environ["SECOND_TB_TRAIN_INPUT_ENV"],
                "sha256": os.environ["SECOND_EMPTY_SHA_ENV"],
            },
            "tensorboard_val": {
                "path": os.environ["SECOND_TB_VAL_INPUT_ENV"],
                "sha256": os.environ["SECOND_EMPTY_SHA_ENV"],
            },
            "code_manifest": {
                "path": os.environ["SECOND_CODE_MANIFEST_INPUT_ENV"],
                "sha256": os.environ["SECOND_CODE_MANIFEST_SHA_ENV"],
            },
            "input_manifest": {
                "path": os.environ["SECOND_INPUT_MANIFEST_INPUT_ENV"],
                "sha256": os.environ["SECOND_INPUT_MANIFEST_SHA_ENV"],
            },
            "cause": "postflight_expected_runtime_root_instead_of_normalized_run_dir",
            "preemptively_stopped": True,
            "pre_loader_stop_verified": True,
            "batches": 0,
            "receipts": 0,
            "weights": 0,
        },
        "structured_collate_routing_failure": {
            "evidence_manifest": {
                "path": os.environ["THIRD_EVIDENCE_INPUT_ENV"],
                "sha256": os.environ["THIRD_EVIDENCE_SHA_ENV"],
            },
            "first_batch_replay_script": {
                "path": os.environ["THIRD_REPLAY_SCRIPT_INPUT_ENV"],
                "sha256": os.environ["THIRD_REPLAY_SCRIPT_SHA_ENV"],
            },
            "first_batch_replay_receipt": {
                "path": os.environ["THIRD_REPLAY_RECEIPT_INPUT_ENV"],
                "sha256": os.environ["THIRD_REPLAY_RECEIPT_SHA_ENV"],
            },
            "consumed_provenance": {
                "path": os.environ["THIRD_PROVENANCE_INPUT_ENV"],
                "sha256": os.environ["THIRD_PROVENANCE_SHA_ENV"],
            },
            "formal_command": {
                "path": os.environ["THIRD_COMMAND_INPUT_ENV"],
                "sha256": os.environ["THIRD_COMMAND_SHA_ENV"],
            },
            "launch_log": {
                "path": os.environ["THIRD_LOG_INPUT_ENV"],
                "sha256": os.environ["THIRD_LOG_SHA_ENV"],
            },
            "config": {
                "path": os.environ["THIRD_CONFIG_INPUT_ENV"],
                "sha256": os.environ["THIRD_CONFIG_SHA_ENV"],
            },
            "train_log": {
                "path": os.environ["THIRD_TRAIN_LOG_INPUT_ENV"],
                "sha256": os.environ["THIRD_TRAIN_LOG_SHA_ENV"],
            },
            "tensorboard_train": {
                "path": os.environ["THIRD_TB_TRAIN_INPUT_ENV"],
                "sha256": os.environ["THIRD_TB_TRAIN_SHA_ENV"],
            },
            "tensorboard_val": {
                "path": os.environ["THIRD_TB_VAL_INPUT_ENV"],
                "sha256": os.environ["THIRD_TB_VAL_SHA_ENV"],
            },
            "code_manifest": {
                "path": os.environ["THIRD_CODE_MANIFEST_INPUT_ENV"],
                "sha256": os.environ["THIRD_CODE_MANIFEST_SHA_ENV"],
            },
            "input_manifest": {
                "path": os.environ["THIRD_INPUT_MANIFEST_INPUT_ENV"],
                "sha256": os.environ["THIRD_INPUT_MANIFEST_SHA_ENV"],
            },
            "cause": "missing_fpr_flag_in_structured_collate_routing",
            "failure_stage": "dataloader_collate_before_first_batch_yield",
            "optimizer_steps": 0,
            "batches": 0,
            "receipts": 0,
            "weights": 0,
        },
    },
    "command": {
        "path": os.environ["COMMAND_ENV"],
        "sha256": command_sha,
    },
    "consumed_provenance": {
        "path": os.environ["PROVENANCE_ENV"],
        "sha256": provenance_sha,
    },
    "data_manifest": {
        "path": os.environ["DATA_MANIFEST_ENV"],
        "sha256": data_manifest_sha,
        "file_count": data_manifest["file_count"],
        "total_size": data_manifest["total_size"],
    },
    "thresholds": required_minima,
    "observed": {name: float(stats[name]) for name in required_minima},
    "diagnostics": {
        name: float(value)
        for name, value in stats.items()
        if name.startswith("parent_relative_text_verifier_")
    },
    "losses": {
        name: float(losses[name])
        for name in ("total_loss", "parent_relative_text_verifier_loss")
    },
}
payload = (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode("utf-8")
path = os.environ["DECISION_ENV"]
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
try:
    offset = 0
    while offset < len(payload):
        offset += os.write(fd, payload[offset:])
    os.fsync(fd)
finally:
    os.close(fd)
directory_fd = os.open(os.path.dirname(path), os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
print(json.dumps(decision, indent=2, sort_keys=True))
PY

chmod 0444 "${DECISION}" "${LAUNCH_LOG}"
sync
echo "audit_complete=true decision=${DECISION} long_training_authorized=false"
