#!/usr/bin/env bash
# nature_pouch, 46-dim ABSOLUTE joint actions, no arm velocity -- a new task, trained alone.
#
# `RooibosT/nature_pouch` is `pick and place the pouch`, recorded 2026-10-03 in the same room and
# on the same day as `nature_fliptable`. 100 successful recordings (18 failures excluded), 100
# episodes, 26,262 frames; same 117-dim state / 33-dim action layout and the same recorder.
# No recording was cut at a frame gap, so here one recording is one episode. base_cmd_vel is
# constant zero, and the 46-dim absarm config does not use it. Checked before conversion: a
# scan of every 10th frame of all 7 source mp4s finds no still run, and after it every
# episode-camera pair moves (median |dpixel| 36-38, min 3.3; frozen clips read 0.2-1.0).
#
# Recipe is `chain_nature_fliptable_absarm.sh`'s, schedule set by epochs. Episodes are short
# (median 261 frames, min 59), so this split has only 20,140 train windows. 5,000 steps is
# 15.9 epochs. Small sets bottom out at 5.6-11.5 epochs and the tail then climbs 13-27%, so
# save 250 puts 20 checkpoints on the grid, one every 0.8 epochs. The val has 2,222 windows,
# and stride 2 scans ~1,100 of them.
#
# The split is split_fliptable_v3.py's: by recording, 10% held out (10 of 100, seed 20260917).
# There is no earlier pouch model, so there is no baseline scan.
#
# Split, motion check, stats, train -- each skipped if its output is already there.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_nature_pouch
RAWDS=$HUB/RooibosT/nature_pouch
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_nature_pouch_absarm.log
STRIDE=2
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -d "$DS" ] || { say "no converted dataset at $DS -- run convert_stage1_v3_to_v2.py first"; exit 1; }

if [ ! -d "${DS}_train" ]; then
    say "splitting $DS by source recording"
    "$ROOT/.venv/bin/python" "$HUB/split_fliptable_v3.py" --src "$DS" --raw "$RAWDS" 2>&1 | tee -a "$LOG" || exit 1
fi

for D in "${DS}_train" "${DS}_val"; do
    say "video motion check on $D"
    "$ROOT/.venv/bin/python" "$HUB/check_video_motion.py" --dataset "$D" --cams all --frames 4 \
        2>&1 | tee -a "$LOG" || { say "FROZEN VIDEO in $D -- refusing to train"; exit 1; }
done

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_nature_pouch.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_naturepouch_absarm
if [ -f "$ROOT/outputs/$EXP/$EXP/scan.json" ]; then
    say "already trained and scanned"
else
    say "launching 46-dim ABS (no arm velocity) on nature_pouch"
    cd "$ROOT"
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    CONFIG=$CFG \
    DATASET_ROOT=$DS \
    EXP_SUFFIX=_naturepouch_absarm \
    MAX_STEPS=5000 SAVE_STEPS=250 EVAL_STEPS=250 \
    SCAN_STRIDE=$STRIDE SCAN_GPUS="0 1" \
    USE_WANDB=1 MASTER_PORT=29702 \
    bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
    say "training + scan exited ($?)"
fi
say "chain complete"
