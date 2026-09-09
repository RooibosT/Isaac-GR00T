#!/usr/bin/env bash
# A already trained to 35000; scan it on both GPUs, then train B on both and let
# run_finetune_ikea.sh scan that one itself. Sequential: each step wants both cards.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
LOG=$ROOT/datasets/finish_sd04.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

A=g1_dex1_ikea_relarm_3view_aug_b64_stage1_absarm_armvel_sd04
B=g1_dex1_ikea_relarm_3view_aug_b64_stage1_absarm_sd04

say "scanning A (already trained to 35000)"
VAL=/root/02_hub/datasets/IKEA_pick_leg_stage1_val STRIDE=7 \
  bash "$EX/scan_when_done.sh" "$A" "$EX/g1_dex1_ikea_absarm_armvel_config.py" 0 1

say "training B on both GPUs"
CONFIG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py \
DATASET_ROOT=/root/02_hub/datasets/IKEA_pick_leg_stage1 \
EXP_SUFFIX=_stage1_absarm_sd04 \
STATE_DROPOUT=0.4 SCAN_STRIDE=7 \
MAX_STEPS=35000 SAVE_STEPS=2000 USE_WANDB=0 MASTER_PORT=29680 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1

say "done: A scanned, B trained and scanned"
