#!/usr/bin/env bash
# URL-RFM/IKEA_fliptable on its own, 60-dim ABSOLUTE joint actions + arm velocity.
#
# The source is one session, 43 episodes of `flip table` under a single
# instruction, cut to v2.1 by `convert_urlrfm_v3_to_v2.py stage1` and split
# episode-level by `split_fliptable.py` (5 held out, seed 20260912): 38 eps /
# 23,693 frames train, 5 / 2,712 val.
#
# 16,000 steps for the same reason `chain_legori.sh` uses them on a set of this
# size (23,983 windows there, 23,693 frames here): an epoch match would be about
# 5,500 steps and 0.35M samples, well short of the ~1.02M where sections 6 and 14
# put arm/EE saturation. 16,000 reaches that sample count and SAVE_STEPS 1,000
# lets the scan pick where it stops improving, overfitting included.
#
# Scan stride 3 rather than 7: val is 2,712 frames, so stride 7 leaves ~360
# windows. Stride 3 gives ~840, which is still thinner than the stage-1 scans
# (1,593) -- read the checkpoint ranking from it, not small differences.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_fliptable
CFG=$EX/g1_dex1_ikea_absarm_armvel_config.py
LOG=$ROOT/datasets/chain_fliptable.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Generated here, once, rather than left to the launcher: concurrent runs writing
# the same stats.json corrupt it (DATASETS.md).
for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_fliptable.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

say "launching 60-dim ABS+armvel on fliptable"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_fliptable_armvel \
MAX_STEPS=16000 SAVE_STEPS=1000 EVAL_STEPS=1000 \
SCAN_STRIDE=3 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29681 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
