#!/usr/bin/env bash
# stage3_rc: 46-dim ABSOLUTE arms, and the base velocity command in the action.
#
# `RooibosT/stage3_rc` is the first IKEA recording here where the robot drives: one
# instruction, `move to the stage 3 location, and rotate the table base`, 100 source
# recordings / 121 episodes / 41,027 frames (the frame-gap filter cut 21 in two), one
# session. Episodes are short -- median 389 frames, 13 s.
#
# **Action is 19 dims, not 16.** Arms 7+7 and grippers 1+1 as usual, plus `base_cmd_vel`
# 3, via `g1_dex1_ikea_absarm_basevel_config.py`. Every other IKEA set here was stationary
# with that block exactly zero, which is why every other config drops it. Measured over
# all 41,027 frames: dim 17 is the real one (fires on 8.8% of frames across 101 of the 121
# episodes, correlates +0.75 with measured `base_lin_vel` y, and the base does move when it
# fires -- |lin| 0.165 against 0.011 idle); dims 16 and 18 are dead (0.2% and 0.0% nonzero,
# q01 == q99 == 0) and ride along only because a modality key cannot be split. They
# normalize to a constant and the model learns to emit zero there.
#
# State stays the 46-dim set, so this differs from `_absarm` runs in the action alone.
# `base_lin_vel` / `base_ang_vel` are in the dataset but not in the state config; putting
# them in is a separate experiment and would move the deploy-side ObservationSpec too.
#
# Schedule by epochs, as every run here does: 8,000 steps over this split's 32,783 train
# windows is 15.6 epochs. save 500 puts 16 checkpoints on the grid. Scan stride 3, not 7:
# val is 3,993 frames, so stride 3 leaves ~1,175 windows -- v1 fliptable used stride 3 for
# a val of that size.
#
# The split is drawn over recordings (10 of 100 held out); 21 recordings were cut in two,
# and split_fliptable_v3.py groups their halves and asserts none straddles.
#
# ⚠️ Deployment must unpack 19 action dims. A client built for the 16-dim IKEA models will
# mis-slice this model's output; the extra three are base_cmd_vel x / y / yaw in dataset
# order, of which only y was ever commanded.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_stage3_rc
RAWDS=$HUB/RooibosT/stage3_rc
CFG=$EX/g1_dex1_ikea_absarm_basevel_config.py
LOG=$ROOT/datasets/chain_stage3_rc.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -d "$DS" ] || { say "no converted dataset at $DS -- run convert_stage1_v3_to_v2.py first"; exit 1; }

if [ ! -d "${DS}_train" ]; then
    say "splitting $DS by source recording"
    "$ROOT/.venv/bin/python" "$HUB/split_fliptable_v3.py" --src "$DS" --raw "$RAWDS" 2>&1 | tee -a "$LOG" || exit 1
fi

# A stalled camera passes every other check in this pipeline (see fliptablev3).
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
            >> "$ROOT/datasets/stats_stage3_rc.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_stage3rc_absarm_basevel
if [ -f "$ROOT/outputs/$EXP/$EXP/scan.json" ]; then
    say "=== already trained and scanned, nothing to do ==="
    exit 0
fi

say "launching 46-dim state / 19-dim action (arms+grippers+base_cmd_vel) on stage3_rc"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_stage3rc_absarm_basevel \
MAX_STEPS=8000 SAVE_STEPS=500 EVAL_STEPS=500 \
SCAN_STRIDE=3 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29695 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
