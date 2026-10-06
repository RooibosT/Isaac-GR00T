#!/usr/bin/env bash
# nature_pouch_new, 46-dim state, RELATIVE arm actions plus the waist yaw command -- 17-dim action.
#
# The REL half of chain_nature_pouch_new_absarm_waist.sh. Same data, split, schedule, state,
# views and 17-dim action layout. The one change is the config:
# `g1_dex1_ikea_relarm_waist_config.py` makes the two arm blocks RELATIVE (an offset from the
# last observed arm state). Grippers and `waist_yaw` stay ABSOLUTE in both runs, so the pair
# differs only in the arm representation. The scan converts the chunk back to absolute targets
# before scoring, so its numbers are directly comparable with the ABS run's scan.json.
#
# RELATIVE also needs `meta/relative_stats.json` entries for `left_arm` / `right_arm`. The ABS
# chain wrote an empty file there, so the stats pass below runs with this config even though
# stats.json already exists. It skips the stats.json entries by their fingerprints and appends
# the two arm entries.
#
# Schedule: 9,000 steps (16.2 epochs over 35,531 train windows), save 500, scan stride 4, as
# the ABS run. MASTER_PORT 29704 (the ABS run holds 29703).
#
# Split, motion check, stats, train -- each skipped if its output is already there.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_nature_pouch_new
RAWDS=$HUB/RooibosT/nature_pouch_new
CFG=$EX/g1_dex1_ikea_relarm_waist_config.py
LOG=$ROOT/datasets/chain_nature_pouch_new_relarm_waist.log
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
    if [ ! -f "$D/meta/stats.json" ] || ! grep -q '"left_arm"' "$D/meta/relative_stats.json" 2>/dev/null; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_nature_pouch_new.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_naturepouchnew_relarm_waist
if [ -f "$ROOT/outputs/$EXP/$EXP/scan.json" ]; then
    say "already trained and scanned"
else
    say "launching 46-dim REL arms + ABS grippers + ABS waist_yaw (17-dim) on nature_pouch_new"
    cd "$ROOT"
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    CONFIG=$CFG \
    DATASET_ROOT=$DS \
    EXP_SUFFIX=_naturepouchnew_relarm_waist \
    MAX_STEPS=9000 SAVE_STEPS=500 EVAL_STEPS=500 \
    SCAN_STRIDE=$STRIDE SCAN_GPUS="0 1" \
    USE_WANDB=1 MASTER_PORT=29704 \
    bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
    say "training + scan exited ($?)"
fi
say "chain complete"
