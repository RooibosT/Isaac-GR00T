#!/usr/bin/env bash
# fliptable again, 46-dim ABSOLUTE joint actions, no arm velocity.
#
# The pair to `chain_fliptable.sh`: same dataset, same split, same schedule --
# 16,000 steps, save 1,000, scan stride 3 -- so the only thing that moves is
# whether left_arm_vel and right_arm_vel are in the state. Matching the schedule
# is what makes the two scans a single-variable comparison, which is why this
# does not shorten to the 3,000 steps where the armvel run bottomed.
#
# stats.json is shared with that run: `gr00t.data.stats` writes one file per
# dataset covering the keys its config names, and the 46-dim set is a subset of
# the 60-dim one, so the existing file already covers it. That is the same
# sharing the two stage1_v2 ABS chains rely on.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_fliptable
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_fliptable_absarm.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_fliptable.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

say "launching 46-dim ABS (no arm velocity) on fliptable"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_fliptable_absarm \
MAX_STEPS=16000 SAVE_STEPS=1000 EVAL_STEPS=1000 \
SCAN_STRIDE=3 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29684 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
