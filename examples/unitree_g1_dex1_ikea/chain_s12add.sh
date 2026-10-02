#!/usr/bin/env bash
# stage1_2_add on its own: the re-collection as a specialist, 46-dim ABS, 16-dim action.
#
# `RooibosT/stage1_2_add_subtask` converted and split by split_stage1_2_add.py — 69 / 8
# episodes, 92,704 H40 train windows, five instructions with the two orient strings rewritten
# to the ones every other model here takes. `action.base_cmd_vel` is constant zero in this
# recording (measured), so this run uses the plain 16-dim action config rather than the
# 19-dim one the merges need.
#
# 18,750 steps = 12.9 epochs, the band the project's best checkpoints land in, and the same
# epoch count the four-set merge runs use. Checkpoints every 1,250 because 15 of them over a
# set this small is what makes the selection fine enough; scan stride 3 over the 9,584-window
# val gives 3,195 scored windows.
#
# Two GPUs, not four: the four-set scan is still on GPUs 0 and 1.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/stage1_2_add
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_s12add.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "launching 46-dim ABS on stage1_2_add alone (5 instructions, 18.75k steps = 12.9 epochs)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
GPUS="${GPUS:-2,3}"
CUDA_VISIBLE_DEVICES="$GPUS" \
NUM_GPUS=2 \
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_s12add_abs \
MAX_STEPS=18750 SAVE_STEPS=1250 EVAL_STEPS=1250 \
SCAN_STRIDE=3 SCAN_GPUS="${GPUS//,/ }" \
USE_WANDB=1 MASTER_PORT=29700 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
