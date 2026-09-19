#!/usr/bin/env bash
# stage1_v2, 46-dim ABS, plus the goal-pose auxiliary block on the action.
#
# The pair to `chain_stage1v2_absarm.sh`: same export, same split, same schedule
# (45,000 / save 2,500 / scan stride 7), same 46-dim state and absolute joint
# actions. The only difference is the nine extra action columns
# `make_goalpose_variant.py` writes -- the right wrist pose at the next gripper
# transition -- so the run measures what that auxiliary loss is worth and nothing
# else.
#
# Read the scan on `mae_arm_first8`, `ee_mm_first8` and `mae_grip`, which are
# block-specific and therefore comparable to the baseline. `mse` is NOT: it
# averages over the action vector, which is nine columns wider here.
#
# The question this run exists for is not in the scan at all. It is whether the
# policy aims at the hole more precisely, measured by
# `probe_hole_localisation.py` -- the baseline sits at 11.6-14.1 mm from 45 to 75
# frames before the release, against 55.8 mm for a policy that always aims at the
# average hole.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_pick_leg_stage1_v2_goal
CFG=$EX/g1_dex1_ikea_absarm_goal_w01_config.py
LOG=$ROOT/datasets/chain_goalpose_w01.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_goalpose_w01.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

say "launching 46-dim ABS + goal-pose auxiliary on stage1_v2"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_stage1v2_absarm_goal_w01 \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29686 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
