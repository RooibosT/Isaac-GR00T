#!/usr/bin/env bash
# fliptable v2, 60-dim ABSOLUTE joint actions *with* arm joint velocities.
#
# The pair to `chain_fliptablev2_absarm.sh`: same export, same split, same
# schedule -- 20,000 steps, save 1,000, eval 1,000, scan stride 7 -- so the only
# thing that moves between the two scans is whether left_arm_vel and
# right_arm_vel are in the state. Matching the schedule is what makes it a
# single-variable comparison, the same way the v1 fliptable pair was run.
#
# 20,000 steps is ~15.4 epochs of this split's 83,125 H40 windows
# (86,947 frames - 98 episodes x 39).
#
# The split and the stats are already on disk from the absarm run and are reused:
# `gr00t.data.stats` writes one file per dataset covering every column of
# observation.state (117 dims here), not just the keys one config names, so the
# 60-dim state this config asks for is already covered. modality.json carries
# left_arm_vel at 46:53 and right_arm_vel at 53:60. Both steps stay guarded so
# the script is restartable if either is ever missing.
#
# MASTER_PORT differs from the absarm chain's 29685 because the absarm run for
# this dataset is training on another host against the same CephFS `/root`;
# the ports cannot actually collide across hosts, but keeping them distinct
# means the two can also be run side by side here.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_fliptable_v2
CFG=$EX/g1_dex1_ikea_absarm_armvel_config.py
LOG=$ROOT/datasets/chain_fliptablev2_armvel.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -d "$DS" ] || { say "no converted dataset at $DS"; exit 1; }

if [ ! -d "${DS}_train" ]; then
    say "splitting $DS by source recording"
    "$ROOT/.venv/bin/python" "$HUB/split_fliptable_v2.py" 2>&1 | tee -a "$LOG" || exit 1
fi

# One stats pass per split before training. Concurrent writers corrupt
# stats.json (DATASETS.md), so this never runs while another chain might.
for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_fliptablev2.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_fliptablev2_armvel
if [ -f "$ROOT/outputs/$EXP/$EXP/scan.json" ]; then
    say "=== already trained and scanned, nothing to do ==="
    exit 0
fi

say "launching 60-dim ABS + arm velocity on fliptable v2"
cd "$ROOT"
# finetune.sh ends in `exec torchrun`, which is only on PATH inside the venv.
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_fliptablev2_armvel \
MAX_STEPS=20000 SAVE_STEPS=1000 EVAL_STEPS=1000 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29686 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
