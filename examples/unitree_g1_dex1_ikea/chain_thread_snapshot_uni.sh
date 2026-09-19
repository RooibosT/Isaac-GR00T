#!/usr/bin/env bash
# thread_snapshot with its four instructions collapsed to one, 60-dim ABS+armvel.
#
# The pair to `chain_thread_snapshot.sh`. `make_thread_snapshot_uni.py` derives
# this set from the already-split one, so the episodes, the split and the arrays
# are identical and only the instruction differs: `thread snapshot` for every
# frame instead of `thread snapshot at 0 / 90 / 180 / 270 degrees`.
#
# Everything else is held at the four-label run's values -- 10,000 steps, save
# 500, scan stride 2, same config, same batch -- because the comparison is only
# worth reading if the schedule does not move. The four-label run bottomed at
# checkpoint-2000 (9.8 epochs) on arm8, EE8 and MSE alike.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/thread_snapshot_uni
CFG=$EX/g1_dex1_ikea_absarm_armvel_config.py
LOG=$ROOT/datasets/chain_thread_snapshot_uni.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_thread_snapshot_uni.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

say "launching 60-dim ABS+armvel on thread_snapshot, one instruction"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_thread_snapshot_uni_armvel \
MAX_STEPS=10000 SAVE_STEPS=500 EVAL_STEPS=500 \
SCAN_STRIDE=2 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29683 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
