#!/usr/bin/env bash
# target_thread_INSERT_subtask + stage1_v2_subtask from the base grasp on, 46-dim ABS.
#
# The follow-up to `chain_thread_s1v2sub.sh`, whose model tested well on the robot. The
# only change is the target_thread half: `RooibosT/target_thread_insert_subtask` keeps
# the same start (the last threading turn) but runs on to the end of the insertion --
# the frame the right gripper finishes opening over the hole -- instead of stopping at
# the base grasp. Same 57 episodes in the same order, same task-1 starts, 51,029 ->
# 60,228 frames (+170 per episode at the median).
#
# So the two halves now OVERLAP across the insertion rather than butting together, which
# is the point: the stage1_v2 half still carries insert -> rotate -> align from the base
# grasp, and these frames are extra insertion demonstrations on top of it.
#
# The stage1_v2 half is untouched -- `IKEA_pick_leg_stage1_v2_subtask_fromgrasp`, the
# same directory the previous run trained on, cut at the frame the left gripper starts
# closing on the rail.
#
# Val holds the same five target_thread recordings the previous run held ([9, 28, 32, 38,
# 49] reproduce from the same seed) and the same eight stage1_v2 ones, so the two runs are
# scored on the same recordings -- but NOT on the same windows, because the target_thread
# episodes are longer here. Read a difference against the previous run as "same clips,
# more insertion in them", and rescore rather than assume.
#
# Recipe unchanged from the deploy SOTA: 46-dim ABS, 45,000 / save 2,500 / scan stride 7.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/thread_ins_s1v2sub
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
REL_CFG=$EX/g1_dex1_ikea_armvel_config.py
LOG=$ROOT/datasets/chain_thread_ins_s1v2sub.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -d "$D" ]; then
        say "$D is missing -- run merge_thread_s1v2sub.py --tt $HUB/target_thread_insert_subtask --out $DS"
        exit 1
    fi
    if [ ! -f "$D/meta/stats.json" ] || [ ! -f "$D/meta/relative_stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$REL_CFG" \
            >> "$ROOT/datasets/stats_thread_ins_s1v2sub.log" 2>&1 \
            || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

say "launching 46-dim ABS on target_thread_insert + stage1_v2_subtask from the base grasp"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_thread_ins_s1v2sub_abs \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29690 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
