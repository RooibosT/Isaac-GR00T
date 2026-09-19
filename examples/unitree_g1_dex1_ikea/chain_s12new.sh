#!/usr/bin/env bash
# stage1_2_new: the re-acquired stage 1 + 2 recording, 46-dim ABS, seven instructions.
#
# What is new in the data is the camera framing: the insertion was recorded so the
# LEFT WRIST view sees the leg meeting the base. Every earlier export had that
# camera pointed elsewhere, and the 3-view config already reads it
# (cam_left_high, cam_left_wrist, cam_right_wrist), so no config change is needed
# to pick it up.
#
# The seven labels are kept unmerged on purpose. On the frozen backbone with image
# and state held fixed, `stage1.1` vs `stage2.2` is 0.00001 and `stage1.3` vs
# `stage2.4` 0.00001, against 0.00003 for merely adding an article and 0.00019 for
# two genuinely different tasks -- the model already treats the repeated wordings
# as one instruction, which is what the data says it should: the two assemblies
# are the same motion (right wrist medians 5 mm and 2.4 deg apart; arm joints
# predict the stage at AUC 0.618, near chance). Seven strings cost nothing in
# training and leave the deploy able to name where it is in the sequence.
#
# 90,000 steps, which is 13.4 epochs of this split's 430,689 windows.
#
# The step count is chosen from where earlier runs actually peaked, and that turns
# out to track EPOCHS rather than steps. Best `mae_arm_first8` by run:
#
#   stage1v2_absarm     215,772 windows   40,000 steps   11.9 epochs
#   s1v2sub_abs         211,268           37,500         11.4
#   stage1_absarm       153,100           35,000         14.6
#   fliptable_absarm     22,211            4,000         11.5
#
# A 19x range of dataset size moves the best step by 10x and leaves the epoch
# count inside 11-15. At 45,000 this split would get 6.7 epochs and at 60,000
# 8.9 -- both below every one of those. It has to be right at launch, because
# there is no extending later: `--save-only-model` keeps no optimizer or
# scheduler state and the finished cosine sits at ~1e-12, so a resume jumps the
# LR and loses more than it gains (measured, 10.5% at 22k).
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/stage1_2_new
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
REL_CFG=$EX/g1_dex1_ikea_armvel_config.py
LOG=$ROOT/datasets/chain_s12new.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ] || [ ! -f "$D/meta/relative_stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$REL_CFG" \
            >> "$ROOT/datasets/stats_s12new.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

say "launching 46-dim ABS on stage1_2_new (7 instructions, 90k steps = 13.4 epochs)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_s12new_abs \
MAX_STEPS=90000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29689 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
