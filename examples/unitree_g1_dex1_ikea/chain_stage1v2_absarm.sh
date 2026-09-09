#!/usr/bin/env bash
# stage1_v2 (172 eps) + 46-dim ABSOLUTE joint actions, no arm velocity.
#
# `RooibosT/IKEA-pick-leg-stage1_v2` is the stage-1 export re-issued with 45 more
# episodes prepended; its last 127 are bit-identical to v1's, so
# `split_stage1_v2.py` carries the v1 val set over unchanged (+45 offset) and
# only train grows: 117 -> 162 eps, 157,663 -> 222,090 frames (+41%).
# Scans of this run therefore sit directly beside the EXPERIMENTS.md 27-28
# tables, where the same config on v1 read arm8 2.261 / EE8 15.48 at 35,000.
#
# 45,000 steps rather than v1's 35,000 because the epoch count is what the old
# schedule was tuned on: 35,000 was 14.6 epochs of 153,100 windows, and 45,000 is
# 13.3 of the 215,772 here. The v1 ABS run flattened at 28,000 = 11.7 epochs,
# which lands at step ~40,000 on this set, so the schedule runs past the expected
# plateau rather than up to it. SAVE_STEPS 2,500 keeps checkpoint-35000 on the
# grid for a like-for-like row against v1.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_pick_leg_stage1_v2
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_stage1v2_absarm.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if [ ! -d "${DS}_train" ]; then
    say "splitting $DS"
    "$ROOT/.venv/bin/python" "$HUB/split_stage1_v2.py" 2>&1 | tee -a "$LOG" || exit 1
fi

# Generated here, once, rather than left to the launcher: concurrent runs writing
# the same stats.json corrupt it (DATASETS.md).
for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_stage1v2.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

say "launching 46-dim ABS on stage1_v2"
cd "$ROOT"
# finetune.sh ends in `exec torchrun`, which is only on PATH inside the venv;
# nohup-ing this script from a bare shell does not inherit an activated one.
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_stage1v2_absarm \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29671 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
