#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly SOURCE_CHECKPOINT="${DATA_ROOT%/}/gf_detector_l6o256.pth"
readonly SOURCE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly DATASET="sr3d"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v99_baseline_gt/sr3d"
readonly EXP="sr3d_mcln_joint_butdcls_fromscratch_v99_e1_e140_b14x1_w4p2"
readonly BATCH_SIZE=14
readonly MAX_EPOCH=140
readonly EXPECTED_EVAL_SAMPLE_COUNT=17726
readonly MASTER_PORT=5399
readonly MIN_FREE_GB=7
readonly BACKBONE_JOINT_TRAINING=1
readonly INFERENCE_USES_GROUND_TRUTH=1
readonly USE_BACKBONE_INITIALIZATION=1
readonly TASK_CHECKPOINT_TRANSFER=0
readonly BACKBONE_AUGMENT_DET=0
CHECKPOINT_RETENTION_METRICS=(rec_acc025)
DATASET_LR_ARGS=(
  --lr_backbone 1e-3 --lr 1e-4
  --lr_decay_epochs 30 40
)
BACKBONE_EXTRA_ARGS=(--joint_det --butd_cls)

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
