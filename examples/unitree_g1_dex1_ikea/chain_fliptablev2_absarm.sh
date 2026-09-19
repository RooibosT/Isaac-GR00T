#!/usr/bin/env bash
# fliptable v2, 46-dim ABSOLUTE joint actions, no arm velocity.
#
# `RooibosT/fliptablev2` is the table-flip motion re-recorded with a different
# way of performing it, and it is 3.7x the data of the v1 set this repo already
# trained on: 101 successful teleop recordings / 109 episodes / 96,869 frames
# against v1's 43 / 26,405. Single instruction, `flip the table`.
#
# That string is **not** v1's, which is `flip table`, and `formalize_language`
# only lowercases and strips punctuation, so the two never converge. It costs
# nothing in training -- one instruction either way -- but a model from this run
# dropped into the v1 model's deploy slot would be fed a string it never saw.
# Left as the dataset has it, deliberately; the deploy side sends the right one.
#
# Split, then stats, then train -- each step skipped if its output is already
# there, so this is restartable.
#
# **The split is drawn over the 101 source recordings, not the 109 episodes.**
# Eight recordings were cut in two by the converter's frame-gap filter, and a
# per-episode draw would put one half in train and the other in val. See
# split_fliptable_v2.py; that is the leak stage1_v2_subtask shipped with.
#
# Schedule: 20,000 steps, save 1,000, scan stride 7. v1 ran 16,000/1,000 on a
# quarter of the data and its scan bottomed at checkpoint-4000 -- 11.5 epochs --
# with the curve flat to worse after, so steps are set by epochs rather than by
# copying v1's count: 20,000 here is ~15 epochs of this split's windows, which
# brackets that bottom instead of running 46 epochs past it. Stride 7 because
# val is ~9k frames here; v1's stride 3 was compensating for a 2,712-frame val.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_fliptable_v2
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_fliptablev2_absarm.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -d "$DS" ] || { say "no converted dataset at $DS -- run convert_stage1_v3_to_v2.py first"; exit 1; }

if [ ! -d "${DS}_train" ]; then
    say "splitting $DS by source recording"
    "$ROOT/.venv/bin/python" "$HUB/split_fliptable_v2.py" 2>&1 | tee -a "$LOG" || exit 1
fi

# One stats pass per split before any training starts; concurrent writers
# corrupt stats.json (DATASETS.md).
for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_fliptablev2.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_fliptablev2_absarm
if [ -f "$ROOT/outputs/$EXP/$EXP/scan.json" ]; then
    say "=== already trained and scanned, nothing to do ==="
    exit 0
fi

say "launching 46-dim ABS (no arm velocity) on fliptable v2"
cd "$ROOT"
# finetune.sh ends in `exec torchrun`, which is only on PATH inside the venv.
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_fliptablev2_absarm \
MAX_STEPS=20000 SAVE_STEPS=1000 EVAL_STEPS=1000 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29685 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
