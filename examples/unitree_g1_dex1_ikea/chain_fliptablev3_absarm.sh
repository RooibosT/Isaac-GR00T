#!/usr/bin/env bash
# fliptable v3, 46-dim ABSOLUTE joint actions, no arm velocity.
#
# `RooibosT/fliptablev3` is `fliptablev2` plus a second recording session: 153 source
# recordings over sessions 260916 (101, v2's material) and 260917 (52), converted to
# 164 LeRobot episodes / 138,245 frames against v2's 109 / 96,869. Same single
# instruction, `flip the table` -- v2's string exactly, so a model from this run drops
# into the same deploy slot.
#
# Everything about the recipe is `chain_fliptablev2_absarm.sh`'s: same config, same
# scan stride 7, both GPUs. **Only the schedule moves, and it moves by epochs rather
# than by copying v2's number**, which is the rule that chain set and
# EXPERIMENTS/memory back: v2 ran 20,000 steps over 83,125 train windows = 15.4 epochs,
# and 28,000 over this split's 119,033 windows is 15.05 -- the same place on the curve.
# save 1,500 puts 18 checkpoints on the grid (v2: 20 at save 1,000).
#
# The split is drawn over *recordings*, stratified by session, 10% of each -- see
# split_fliptable_v3.py. Its grouping key is not v2's: v3 writes `source_episode` as
# `episode_<index>__<session>_e<number>`, so the two halves of a gap-cut recording differ
# in the first field and grouping on the whole string would have split all 11 pairs.
#
# ⚠️ Val is NOT v2's val -- different recordings, and v3's val holds session 260917
# material v2 never had. This run's scan numbers therefore do not sit beside
# `_fliptablev2_absarm`'s; rescore that model on `IKEA_fliptable_v3_val` before comparing.
#
# Split, then stats, then train -- each step skipped if its output is already there, so
# this is restartable.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_fliptable_v3
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_fliptablev3_absarm.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -d "$DS" ] || { say "no converted dataset at $DS -- run convert_stage1_v3_to_v2.py first"; exit 1; }

if [ ! -d "${DS}_train" ]; then
    say "splitting $DS by source recording"
    "$ROOT/.venv/bin/python" "$HUB/split_fliptable_v3.py" 2>&1 | tee -a "$LOG" || exit 1
fi

# One stats pass per split before any training starts; concurrent writers corrupt
# stats.json (DATASETS.md).
for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_fliptablev3.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_fliptablev3_absarm
if [ -f "$ROOT/outputs/$EXP/$EXP/scan.json" ]; then
    say "=== already trained and scanned, nothing to do ==="
    exit 0
fi

say "launching 46-dim ABS (no arm velocity) on fliptable v3"
cd "$ROOT"
# finetune.sh ends in `exec torchrun`, which is only on PATH inside the venv.
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_fliptablev3_absarm \
MAX_STEPS=28000 SAVE_STEPS=1500 EVAL_STEPS=1500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29692 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
