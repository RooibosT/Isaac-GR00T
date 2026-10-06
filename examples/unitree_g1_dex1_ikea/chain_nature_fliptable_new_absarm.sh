#!/usr/bin/env bash
# nature_fliptable_new, 46-dim ABSOLUTE joint actions, no arm velocity -- the second nature session alone.
#
# `RooibosT/nature_fliptable_new` is `flip the table` recorded 2026-10-05 (session `table2`), on
# the same table and the same day as nature_pouch_new. The head camera sees a wooden desk behind
# the table instead of nature_fliptable's grey cabinets. 101 kept recordings (27 excluded), 109
# episodes, 84,685 frames, median recording 839 frames. Same 117-dim state as every IKEA set; the
# action is 34-dim like nature_pouch_new's, with `waist_yaw` at dim 33, but it is constant zero
# here ([w] was never pressed), as is base_cmd_vel. The 16-dim absarm action uses neither.
# Checked before training: all 436 clips have the expected frame count and match the source.
# Posture agrees with nature_fliptable: start EE within 2 cm on both hands, and the mean left EE
# sits 5 cm further left (y 0.238 vs 0.184).
#
# Recipe is `chain_nature_fliptable_absarm.sh`'s, schedule set by epochs: 18,000 steps over this
# split's 72,356 train windows is 15.9 epochs (nature_fliptable: 16,000 over 65,851 = 15.6, best
# at 12.6). save 1,000 puts 18 checkpoints on the grid. The val has 8,078 windows, and stride 4
# scans ~2,000 of them.
#
# The split is split_fliptable_v3.py's: by recording, 10% held out (10 of 101, seed 20260917),
# because the frame-gap filter cut 7 recordings into 15 episodes.
#
# Before training, the two earlier nature models are scored on this val. Neither trained on
# session table2, so these are leak-free transfer numbers to the new background:
#   - nature_fliptable checkpoint-13000 (the uploaded specialist, same 16-dim config)
#   - nature flip + pouch(waist) checkpoint-20000 (uploaded, 17-dim config; this set's
#     waist_yaw column is zero, which is what that model was trained to output on flip)
#
# Split, motion check, stats, baselines, train -- each skipped if its output is already there.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_nature_fliptable_new
RAWDS=$HUB/RooibosT/nature_fliptable_new
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_nature_fliptable_new_absarm.log
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
            >> "$ROOT/datasets/stats_nature_fliptable_new.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_naturefliptablenew_absarm

# Baselines, one per GPU, written one level above the run's checkpoint dir (scan_when_done.sh
# merges every scan_*.json it finds there into scan.json, and these are not this run's).
if [ -d "$HOME/micromamba/envs/ffmpeg7/lib" ]; then
    export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
baseline() {  # gpu  run  step  config  output-name
    local dir=$ROOT/outputs/$2/$2 out=$ROOT/outputs/$EXP/$5
    [ -f "$out" ] && return 0
    [ -d "$dir/checkpoint-$3" ] || { say "baseline $2 checkpoint-$3 missing -- skipped"; return 0; }
    say "scoring $2 checkpoint-$3 on ${DS}_val (stride $STRIDE, GPU $1)"
    cd "$ROOT" && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES=$1 "$ROOT/.venv/bin/python" "$EX/scan_ikea.py" \
        --checkpoints-dir "$dir" --dataset-path "${DS}_val" --config "$4" \
        --embodiment-tag new_embodiment --stride "$STRIDE" --steps "$3" \
        --output "$out" >> "$ROOT/datasets/scan_baselines_on_naturefliptablenew_val.log" 2>&1 \
        || say "baseline $2 FAILED (see datasets/scan_baselines_on_naturefliptablenew_val.log)"
}
baseline 0 g1_dex1_ikea_relarm_3view_aug_b64_naturefliptable_absarm 13000 "$CFG" \
    naturefliptable_ckpt13000_on_new_val.json &
baseline 1 g1_dex1_ikea_relarm_3view_aug_b64_natureflippouchw_absarm_waist 20000 \
    "$EX/g1_dex1_ikea_absarm_waist_config.py" natureflippouchw_ckpt20000_on_new_val.json &
wait
say "baselines done"

if [ -f "$ROOT/outputs/$EXP/$EXP/scan.json" ]; then
    say "already trained and scanned"
else
    say "launching 46-dim ABS (no arm velocity) on nature_fliptable_new"
    cd "$ROOT"
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    CONFIG=$CFG \
    DATASET_ROOT=$DS \
    EXP_SUFFIX=_naturefliptablenew_absarm \
    MAX_STEPS=18000 SAVE_STEPS=1000 EVAL_STEPS=1000 \
    SCAN_STRIDE=$STRIDE SCAN_GPUS="0 1" \
    USE_WANDB=1 MASTER_PORT=29705 \
    bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
    say "training + scan exited ($?)"
fi
say "chain complete"
