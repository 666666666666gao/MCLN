#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly SOURCE_CHECKPOINT="${DATA_ROOT%/}/output/single_stage_best_postprocess/scanrefer/mcln_epoch71_parent_geometry_single_stage_e1_e100_b18x4/1785907694/ckpt_best_rec_acc025.pth"
readonly SOURCE_SHA256="8804109f0db25113cc6683314dcc7ab1ca2f7a93c1307a1fcf76420c6dc43eec"
readonly DATASET="scanrefer"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v99_dataset_only/scanrefer_single_stage"
readonly EXP="scanrefer_single_stage_v99_backbone_e1_e12_b18x1"
readonly BATCH_SIZE=18
readonly MAX_EPOCH=12
readonly EXPECTED_EVAL_SAMPLE_COUNT=9508
readonly MASTER_PORT=5199
readonly MIN_FREE_GB=7
DATASET_LR_ARGS=(
  --lr_backbone 5e-5 --lr 5e-6 --text_encoder_lr 7.5e-7
  --lr_decay_epochs 8 11
)

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
