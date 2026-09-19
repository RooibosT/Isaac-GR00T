#!/usr/bin/env bash
# fliptable v3.1, 46-dim ABSOLUTE joint actions, no arm velocity.
#
# `RooibosT/fliptablev3.1` replaces `fliptablev3`, whose added session (260917) shipped with
# **frozen video on all four cameras** while proprioception moved -- 55 episodes, 37.7% of
# that set's training windows, and the reason its model deployed worse than v2's. v3.1 keeps
# v2's session 260916 (101 recordings, episode lengths identical to the v2 export) and adds
# 260919 instead: 57 recordings, 67 episodes, verified moving on every episode and every
# camera (median |dpixel| 60-67 against the frozen session's 0.2-1.0). 158 recordings / 176
# episodes / 137,841 frames.
#
# Recipe is `chain_fliptablev2_absarm.sh`'s, schedule set by epochs as that chain established:
# 28,000 steps over this split's 116,944 train windows is 15.3 epochs, where v2's 20,000 over
# 83,125 was 15.4. save 1,500 puts 18 checkpoints on the grid; stride 7 leaves ~2,000 of the
# val's 14,033 windows.
#
# The split is drawn over *recordings*, stratified by session (10 of 101 and 6 of 57), because
# the frame-gap filter cut 18 recordings in two here and a per-episode draw would straddle
# them. See split_fliptable_v3.py, which also asserts no recording appears in both splits.
#
# ⚠️ Val is neither v2's nor v3's. Rescore any model you want to compare against on
# `IKEA_fliptable_v31_val` first -- the numbers do not line up otherwise.
#
# Split, motion check, stats, train -- each skipped if its output is already there.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_fliptable_v31
RAWDS=$HUB/RooibosT/fliptablev3.1
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_fliptablev31_absarm.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -d "$DS" ] || { say "no converted dataset at $DS -- run convert_stage1_v3_to_v2.py first"; exit 1; }

if [ ! -d "${DS}_train" ]; then
    say "splitting $DS by source recording, stratified by session"
    "$ROOT/.venv/bin/python" "$HUB/split_fliptable_v3.py" --src "$DS" --raw "$RAWDS" 2>&1 | tee -a "$LOG" || exit 1
fi

# The check v3 did not have. A stalled camera passes every other check in this pipeline:
# clip frame counts match meta, cut clips match the source, and the open-loop scan still
# scores well because the arm state alone predicts the trajectory.
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
            >> "$ROOT/datasets/stats_fliptablev31.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_fliptablev31_absarm
if [ -f "$ROOT/outputs/$EXP/$EXP/scan.json" ]; then
    say "=== already trained and scanned, nothing to do ==="
    exit 0
fi

say "launching 46-dim ABS (no arm velocity) on fliptable v3.1"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_fliptablev31_absarm \
MAX_STEPS=28000 SAVE_STEPS=1500 EVAL_STEPS=1500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29694 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
