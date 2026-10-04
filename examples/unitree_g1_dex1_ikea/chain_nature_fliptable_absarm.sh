#!/usr/bin/env bash
# nature_fliptable, 46-dim ABSOLUTE joint actions, no arm velocity -- the new room alone.
#
# `RooibosT/nature_fliptable` is the fliptable task re-recorded in a different room (2026-10-03):
# planked dark table top instead of the black cloth, grey partitions and a tiled floor behind
# it, and the table sits nearer the robot (left EE x 0.246 m against v3.1's 0.296 / 0.325).
# 105 successful recordings, 115 episodes, 77,299 frames; same 117-dim state / 33-dim action
# layout as v3.1 and the same task string, `flip the table`. Checked before conversion: every
# episode-camera pair moves (median |dpixel| 55-68, min 8.7), a scan of every 10th frame of all
# 21 source mp4s finds no stall, and every file's frame count matches meta.
#
# Recipe is `chain_fliptablev31_absarm.sh`'s, schedule set by epochs: 16,000 steps over this
# split's 65,851 train windows is 15.6 epochs (v3.1: 28,000 over 116,944 = 15.3). save 1,000
# puts 16 checkpoints on the grid. The val is half v3.1's size (6,963 windows), so stride 4
# rather than 7 keeps ~1,700 scanned windows.
#
# The split is split_fliptable_v3.py's: by recording, 10% held out (10 of 105, seed 20260917),
# because the frame-gap filter cut 6 recordings into 13 episodes.
#
# After the scan, v3.1's uploaded checkpoint-28000 is scored on the same val. Neither model
# trained on these recordings, so that is v3.1's transfer to the new room, with no leak.
#
# Split, motion check, stats, train, baseline -- each skipped if its output is already there.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_nature_fliptable
RAWDS=$HUB/RooibosT/nature_fliptable
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_nature_fliptable_absarm.log
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
            >> "$ROOT/datasets/stats_nature_fliptable.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_naturefliptable_absarm
if [ -f "$ROOT/outputs/$EXP/$EXP/scan.json" ]; then
    say "already trained and scanned"
else
    say "launching 46-dim ABS (no arm velocity) on nature_fliptable"
    cd "$ROOT"
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    CONFIG=$CFG \
    DATASET_ROOT=$DS \
    EXP_SUFFIX=_naturefliptable_absarm \
    MAX_STEPS=16000 SAVE_STEPS=1000 EVAL_STEPS=1000 \
    SCAN_STRIDE=$STRIDE SCAN_GPUS="0 1" \
    USE_WANDB=1 MASTER_PORT=29701 \
    bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
    say "training + scan exited ($?)"
fi

# Written one level above the run's checkpoint dir: scan_when_done.sh merges every
# scan_*.json it finds there into scan.json, and this one is not this run's.
V31=$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_fliptablev31_absarm/g1_dex1_ikea_relarm_3view_aug_b64_fliptablev31_absarm
BASE=$ROOT/outputs/$EXP/v31_ckpt28000_on_nature_val.json
if [ ! -f "$BASE" ] && [ -d "$V31/checkpoint-28000" ]; then
    say "scoring v3.1 checkpoint-28000 on ${DS}_val (stride $STRIDE)"
    cd "$ROOT"
    if [ -d "$HOME/micromamba/envs/ffmpeg7/lib" ]; then
        export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES=0 "$ROOT/.venv/bin/python" "$EX/scan_ikea.py" \
        --checkpoints-dir "$V31" --dataset-path "${DS}_val" --config "$CFG" \
        --embodiment-tag new_embodiment --stride "$STRIDE" --steps 28000 \
        --output "$BASE" >> "$ROOT/datasets/scan_v31_on_nature_val.log" 2>&1 \
        || say "baseline scan FAILED (see datasets/scan_v31_on_nature_val.log)"
fi
say "chain complete"
