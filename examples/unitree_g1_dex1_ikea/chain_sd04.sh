#!/usr/bin/env bash
# Run the two state-dropout 0.4 ablations back to back, scanning each.
#
# Both GPUs are used by whichever step is running, so this is a queue rather
# than a fan-out: 60-dim finishes and is scanned, then 46-dim trains and is
# scanned. Each run's baseline is the same config at state_dropout 0.2, so the
# pair isolates the dropout change on each state width.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
LOG=$ROOT/datasets/chain_sd04.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_for() {   # wait until no launch_finetune names this output dir
    while pgrep -f "output_dir $ROOT/outputs/$1 " >/dev/null 2>&1; do sleep 60; done
    sleep 45
}
scan() {       # $1 experiment name, $2 config
    say "scanning $1"
    VAL=/root/02_hub/datasets/IKEA_pick_leg_stage1_val STRIDE=7 \
      bash "$EX/scan_when_done.sh" "$1" "$2" 0 1
}

A=g1_dex1_ikea_relarm_3view_aug_b64_stage1_absarm_armvel_sd04
B=g1_dex1_ikea_relarm_3view_aug_b64_stage1_absarm_sd04

say "waiting for $A to finish training"
wait_for "$A"
scan "$A" "$EX/g1_dex1_ikea_absarm_armvel_config.py"

# Run A was launched before run_finetune_ikea.sh grew AUTO_SCAN, so its scan is
# the explicit one above. B goes through the launcher, which now scans itself on
# exit -- SCAN_STRIDE 7 because these numbers sit beside the stage1 table.
say "launching $B (46-dim ABS, state_dropout 0.4)"
CONFIG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py \
DATASET_ROOT=/root/02_hub/datasets/IKEA_pick_leg_stage1 \
EXP_SUFFIX=_stage1_absarm_sd04 \
STATE_DROPOUT=0.4 SCAN_STRIDE=7 \
MAX_STEPS=35000 SAVE_STEPS=2000 USE_WANDB=0 MASTER_PORT=29670 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1

say "chain complete"
