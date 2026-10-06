#!/usr/bin/env bash
# One model for both nature-room tasks: `flip the table` + `pick and place the pouch` (with the
# waist turn). 46-dim state, 17-dim ABSOLUTE action -- arms 7+7, grippers 1+1, waist_yaw 1.
#
# Waits for the nature_pouch_new upload chain to finish first (the user's order: upload, then
# this), which also means that run's training and scan are done and both GPUs are free.
#
# Data (`IKEA_nature_flip_pouchw_{train,val}`, built by merge_ikea_tasksets.py):
#   * `IKEA_nature_fliptable_w0` -- nature_fliptable's own split, unchanged, with a zero
#     `waist_yaw` appended to the action (make_nature_fliptable_w0.py). The fliptable recording
#     held the waist at its startup pose: waist yaw state -0.36..+0.21 deg, so 0 is the
#     recorded fact. The model learns to keep the waist forward while flipping.
#   * `IKEA_nature_pouch_new` -- its own split, unchanged.
#   Each val is that set's own val, so neither specialist's val leaks into this train and the
#   per-task rows sit on the specialists' exact scan windows (same episodes, stride 4).
#   train 194 eps / 108,948 frames / 101,382 windows (flip 65%, pouch 35%); val 20 eps /
#   11,077 windows. Instructions: 0 `flip the table`, 1 `pick and place the pouch`.
#   Merged stats keep `waist_yaw`'s full band (q01 / q99 = -0.7854 / 0).
#
# Schedule by epochs: 22,000 steps over 101,382 windows is 13.9 epochs. The specialists bottomed
# at 12.6 (fliptable) and 12.7 (pouch) epochs, and sets of this size have flat tails. save
# 1,000 puts 22 checkpoints on the grid. Scan stride 4 is the specialists' stride, so the
# per-task arm8 / EE8 compare directly with `..._naturefliptable_absarm` checkpoint-13000 and the
# pouch_new pick; `mae_waist` on the flip rows says whether the waist stays put there.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_nature_flip_pouchw
CFG=$EX/g1_dex1_ikea_absarm_waist_config.py
LOG=$ROOT/datasets/chain_nature_flip_pouchw_absarm_waist.log
UPLOAD_LOG=$ROOT/datasets/upload_naturepouchnew.log
STRIDE=4
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the nature_pouch_new upload chain to finish"
until grep -qE "upload done|UPLOAD FAILED|NO scan.json|pick step FAILED" "$UPLOAD_LOG" 2>/dev/null; do
    sleep 60
done
say "upload chain finished: $(grep -E "upload done|UPLOAD FAILED|NO scan.json|pick step FAILED" "$UPLOAD_LOG" | tail -1)"

for D in "${DS}_train" "${DS}_val"; do
    [ -f "$D/meta/episodes.jsonl" ] || { say "no merged dataset at $D -- run merge_ikea_tasksets.py first"; exit 1; }
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_nature_flip_pouchw.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_natureflippouchw_absarm_waist
if [ -f "$ROOT/outputs/$EXP/$EXP/scan.json" ]; then
    say "already trained and scanned"
else
    say "launching flip + pouch (waist) multi-task, 46-dim ABS + waist_yaw action (17-dim)"
    cd "$ROOT"
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    CONFIG=$CFG \
    DATASET_ROOT=$DS \
    EXP_SUFFIX=_natureflippouchw_absarm_waist \
    MAX_STEPS=22000 SAVE_STEPS=1000 EVAL_STEPS=1000 \
    SCAN_STRIDE=$STRIDE SCAN_GPUS="0 1" \
    USE_WANDB=1 MASTER_PORT=29704 \
    bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
    say "training + scan exited ($?)"
fi
say "chain complete"
