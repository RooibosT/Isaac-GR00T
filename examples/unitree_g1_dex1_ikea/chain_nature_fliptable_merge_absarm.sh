#!/usr/bin/env bash
# nature_fliptable + nature_fliptable_new, 46-dim ABSOLUTE joint actions -- both nature flip sessions in one model.
#
# The two sessions flip the same table under the same instruction, `flip the table`, but the head
# camera sees grey cabinets behind it in `nature_fliptable` (2026-10-03) and a wooden desk in
# `nature_fliptable_new` (2026-10-05, session table2). Each specialist scores ~2.7x worse on the
# other session's val (nature_fliptable ckpt-13000 on the new val: arm8 7.029 deg against 2.570 for
# the new specialist), so the question is whether one model covers both backgrounds at no cost.
#
# Data: `merge_ikea_tasksets.py --sets IKEA_nature_fliptable_w0 IKEA_nature_fliptable_new`. The w0
# copy of nature_fliptable carries the 34-dim action (zero waist_yaw) the merge needs to match
# the new set's shape. The 16-dim absarm action uses neither waist_yaw nor base_cmd_vel. Each set
# keeps its own split, so val = both specialists' vals and neither model's val leaks into the
# other's train. Train has 138,207 H40 windows (nature 47.6%, new 52.4%), val 15,041.
#
# Schedule by epochs: 34,000 steps is 15.7 epochs (the specialists ran 15.6 and 15.9; their best
# checkpoints sat at 12.6 and 15.9). save 2,000 puts 17 checkpoints on the grid. The merged val at
# stride 4 is ~3,760 windows.
#
# Both sets share one instruction, so the scan cannot split the merged val by session. After the
# main scan, the arm8 pick is scored on each specialist's own val at stride 4 -- the identical
# windows those specialists were scanned on (1,745 nature / 2,022 new) -- so the comparison is
# direct: per_session_<step>_{nature,new}.json beside the run.
#
# Stats, train, scan, per-session scans -- each skipped if its output is already there.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_nature_fliptable_merge
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_nature_fliptable_merge_absarm.log
STRIDE=4
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    [ -d "$D" ] || { say "no merged split at $D -- run merge_ikea_tasksets.py first"; exit 1; }
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_nature_fliptable_merge.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_naturefliptablemerge_absarm
RUN=$ROOT/outputs/$EXP/$EXP
if [ -f "$RUN/scan.json" ]; then
    say "already trained and scanned"
else
    say "launching 46-dim ABS (no arm velocity) on nature_fliptable + nature_fliptable_new"
    cd "$ROOT"
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    CONFIG=$CFG \
    DATASET_ROOT=$DS \
    EXP_SUFFIX=_naturefliptablemerge_absarm \
    MAX_STEPS=34000 SAVE_STEPS=2000 EVAL_STEPS=2000 \
    SCAN_STRIDE=$STRIDE SCAN_GPUS="0 1" \
    USE_WANDB=1 MASTER_PORT=29706 \
    bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
    say "training + scan exited ($?)"
fi

# Per-session scores of the arm8 pick, written one level above the run's checkpoint dir
# (scan_when_done.sh merges every scan_*.json it finds there into scan.json).
[ -f "$RUN/scan.json" ] || { say "NO scan.json -- skipping the per-session scans"; say "chain complete"; exit 1; }
STEP=$("$ROOT/.venv/bin/python" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(min(d, key=lambda k: d[k]['__all__']['mae_arm_first8']).split('-')[1])" "$RUN/scan.json")
if [ -d "$HOME/micromamba/envs/ffmpeg7/lib" ]; then
    export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
session_scan() {  # gpu  val-dataset  name
    local out=$ROOT/outputs/$EXP/per_session_${STEP}_$3.json
    [ -f "$out" ] && return 0
    say "scoring checkpoint-$STEP on $2 (stride $STRIDE, GPU $1)"
    cd "$ROOT" && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES=$1 "$ROOT/.venv/bin/python" "$EX/scan_ikea.py" \
        --checkpoints-dir "$RUN" --dataset-path "$2" --config "$CFG" \
        --embodiment-tag new_embodiment --stride "$STRIDE" --steps "$STEP" \
        --output "$out" >> "$ROOT/datasets/scan_naturefliptablemerge_per_session.log" 2>&1 \
        || say "per-session scan on $3 FAILED (see datasets/scan_naturefliptablemerge_per_session.log)"
}
session_scan 0 "$HUB/IKEA_nature_fliptable_val" nature &
session_scan 1 "$HUB/IKEA_nature_fliptable_new_val" new &
wait
say "chain complete"
