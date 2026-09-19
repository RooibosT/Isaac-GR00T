#!/usr/bin/env bash
# stage1_v2 demonstrations + the human-intervention (DAgger) expert clips, 46-dim ABS.
#
# The pair to the stage1_v2 ABS baseline (`..._stage1v2_absarm`, ckpt-40000:
# arm8 2.099 deg / EE8 14.94 mm / grip 0.1088): same config, same schedule
# (45,000 / save 2,500 / scan stride 7), same untouched val. The only change is
# 61 extra episodes -- the teleop takeovers recorded just before the deployed
# policy would have failed, i.e. the states a demonstration never visits.
#
# The expert is 5.5% of frames and 4.6% of horizon-40 windows here, its natural
# proportion. Oversampling is a SEPARATE run built with
# `merge_stage1v2_dagger.py --repeat N`, deliberately not folded into this one:
# at --repeat 1 the scan answers "does adding it help", and only against that
# number does a weighted run mean anything.
#
# The scan cannot settle this. Val is 10 demonstration episodes and holds no
# intervention states at all, so it measures whether the extra data BROKE
# anything, not whether it fixed the failure. The verdict is on the robot.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_pick_leg_stage1_v2_dagger
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_dagger.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_dagger.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

say "launching 46-dim ABS on stage1_v2 + intervention expert"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_stage1v2_dagger \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29687 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
