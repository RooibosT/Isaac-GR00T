#!/usr/bin/env bash
# RooibosT/thread_snapshot, 60-dim ABSOLUTE joint actions + arm velocity.
#
# 85 short episodes (median 198 frames) under **four** instructions -- `thread
# snapshot at 0 / 90 / 180 / 270 degrees`, 20-22 episodes each. `split_thread_
# snapshot.py` holds out 3 per label so every orientation is scored: 73 eps /
# 14,240 frames train, 12 / 2,375 val.
#
# 10,000 steps, not the 16,000 `chain_legori.sh` and `chain_fliptable.sh` use.
# Those sets are ~24k windows; this one is ~13k, so 16,000 would be 79 epochs.
# The legori scan is the guide for a set this small: its ABS run bottomed at
# checkpoint-4000 (10.7 epochs) and was worse by 16,000, while the REL run
# bottomed at 11,000. 10,000 here is 49 epochs, well past both, and SAVE_STEPS
# 500 puts 20 checkpoints on the grid so the scan can see the curve rather than
# four points on it.
#
# Scan stride 2: val is 2,375 frames in 12 short episodes, so stride 2 leaves
# ~950 windows. Read the checkpoint ranking from it, not small differences.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/thread_snapshot
CFG=$EX/g1_dex1_ikea_absarm_armvel_config.py
LOG=$ROOT/datasets/chain_thread_snapshot.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_thread_snapshot.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

say "launching 60-dim ABS+armvel on thread_snapshot"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_thread_snapshot_armvel \
MAX_STEPS=10000 SAVE_STEPS=500 EVAL_STEPS=500 \
SCAN_STRIDE=2 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29682 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
