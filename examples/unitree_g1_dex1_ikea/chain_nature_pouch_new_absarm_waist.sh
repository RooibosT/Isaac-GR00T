#!/usr/bin/env bash
# nature_pouch_new, 46-dim state, ABSOLUTE joint actions plus the waist yaw command -- 17-dim action.
#
# `RooibosT/nature_pouch_new` is `pick and place the pouch` again, recorded 2026-10-05 with the
# recorder's [w] waist turn: every one of the 99 kept episodes turns the waist out to -pi/4 and
# back once. 99 recordings, 99 episodes, 43,506 frames, median 438 frames (1.7x nature_pouch,
# the turn adds the difference). Same 117-dim state as every IKEA set; the action is 34-dim, the
# first 33 as before and `waist_yaw` appended at dim 33. Checked before training: all 396 clips
# have the expected frame count and match the source, and every episode-camera pair moves
# (median |dpixel| 76-77, min 38).
#
# The config is `g1_dex1_ikea_absarm_waist_config.py`: the 46-dim absarm state with
# `waist_yaw` added as a fifth action key, appended so dims 0-15 keep the 16-dim order. Without
# it the model has no way to produce the turn. scan_ikea.py reports `mae_waist` alongside the
# arm metrics for it.
#
# Schedule by epochs, as nature_pouch: 9,000 steps over this split's 35,531 train windows is
# 16.2 epochs (nature_pouch: 5,000 over 20,140 = 15.9). save 500 puts 18 checkpoints on the
# grid, one every 0.9 epochs. The val has 4,114 windows, and stride 4 scans ~1,030 of them.
#
# The split is split_fliptable_v3.py's: by recording, 10% held out (10 of 99, seed 20260917).
# This export's segments.json names recordings by `original_episode` rather than by a
# `__<session>_e<n>` suffix; the split script reads both. There is no earlier waist model, so
# there is no baseline scan.
#
# Split, motion check, stats, train -- each skipped if its output is already there.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_nature_pouch_new
RAWDS=$HUB/RooibosT/nature_pouch_new
CFG=$EX/g1_dex1_ikea_absarm_waist_config.py
LOG=$ROOT/datasets/chain_nature_pouch_new_absarm_waist.log
STRIDE=4
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
            >> "$ROOT/datasets/stats_nature_pouch_new.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_naturepouchnew_absarm_waist
if [ -f "$ROOT/outputs/$EXP/$EXP/scan.json" ]; then
    say "already trained and scanned"
else
    say "launching 46-dim ABS + waist_yaw action (17-dim) on nature_pouch_new"
    cd "$ROOT"
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    CONFIG=$CFG \
    DATASET_ROOT=$DS \
    EXP_SUFFIX=_naturepouchnew_absarm_waist \
    MAX_STEPS=9000 SAVE_STEPS=500 EVAL_STEPS=500 \
    SCAN_STRIDE=$STRIDE SCAN_GPUS="0 1" \
    USE_WANDB=1 MASTER_PORT=29703 \
    bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
    say "training + scan exited ($?)"
fi
say "chain complete"
