#!/usr/bin/env bash
# The thread_ins_s1v2sub recipe again, 60,000 steps instead of 45,000.
#
# Two reasons to extend, one from each side:
#
#   * The 45,000-step run never flattened. arm8, EE8 and grip were all still at their
#     minimum on the LAST checkpoint (2.208 deg / 15.01 mm / 0.1175 at 45,000), unlike
#     `_thread_s1v2sub_abs`, which bottomed at 42,500 and sat flat from 35,000.
#   * On the robot, the predecessor generalised poorly in `stage0.5` — closing the right
#     gripper and pushing it into the leg hole. `stage0.5` is the thinnest instruction
#     here, 25,274 of 204,325 train frames (12.4%).
#
# 60,000 steps is 19.5 epochs of this split's 196,486 H40 windows, against 14.7 at
# 45,000. Save grid stays 2,500, so the two runs' checkpoints line up and 45,000 can be
# read off this run's scan as well — 24 checkpoints, ~290 GB.
#
# ⚠️ What this cannot do: epochs do not add orient data. If the robot failure is that
# `stage0.5` is under-represented rather than under-trained, more passes over the same
# 25,274 frames overfit them instead, and the open-loop scan cannot tell the two apart —
# every val frame is a demonstration, and the failure is closed-loop. The scan here is
# for checkpoint selection; the verdict is the robot. If this run does not fix it, the
# lever is more `stage0.5` material (or weighting it), not more steps.
#
# Dataset unchanged from `chain_thread_ins_s1v2sub.sh`, stats already generated.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/thread_ins_s1v2sub
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
REL_CFG=$EX/g1_dex1_ikea_armvel_config.py
LOG=$ROOT/datasets/chain_thread_ins_s1v2sub_60k.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ] || [ ! -f "$D/meta/relative_stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$REL_CFG" \
            >> "$ROOT/datasets/stats_thread_ins_s1v2sub.log" 2>&1 \
            || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

say "launching 46-dim ABS, 60,000 steps, on target_thread_insert + stage1_v2_subtask"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_thread_ins_s1v2sub_abs_60k \
MAX_STEPS=60000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29691 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
