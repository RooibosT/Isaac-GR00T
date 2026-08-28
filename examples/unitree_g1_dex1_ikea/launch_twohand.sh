#!/usr/bin/env bash
# Three 2-GPU runs on the re-shot rotate data.
#
#   RV   split labels, 60-dim arm velocity   GPU 0,1   the candidate
#   R    split labels, 46-dim                GPU 2,3   if arm velocity is not allowed
#   R1V  one rotate label, 60-dim            GPU 4,5   prices the split
#
# What changed in the data. `URL-RFM/IKEA_rotatetable1_v2` re-shoots the set that
# fails on hardware with a different technique, and the difference is not subtle.
# The older three rotate sets are all one-handed: the right arm's joints move
# 0.05-0.23 rad and its wrist never leaves an 8-17 cm box. The re-shoot moves the
# right arm 0.35-1.24 rad per joint and its wrist 9-11 cm on every axis, at a
# higher median torque (1.08 -> 1.62 N.m), and never closes the right gripper --
# it braces the table with an open hand while the left turns it. It also finishes
# 13% sooner (median 449 -> 390 frames).
#
# So the re-shoot replaces `IKEA_rotatetable1` rather than joining it: two answers
# to one picture average into a motion that is neither. `IKEA_rotatetable3` goes
# with it -- same direction, still one-handed, and sets 1 and 3 are the pair that
# fails on hardware while set 2 works.
#
# ⚠️ What this costs: the three-leg configuration now has no rotate data at all.
# The bet is that the two-handed technique generalises to it, which is untested.
#
# The labels. `turn the tabletop square` for the re-shoot, `spin the crossbars
# around` for set 2. They sit 0.1574 apart in the frozen text stack -- the widest
# of the pairs measured, and wider than the 0.1447 between `insert` and `rotate leg
# to tighten`, the pair already known to interfere. Neither comes closer than
# 0.1136 to a task that stays fixed. The strings say nothing about direction on
# purpose: direction adverbs measured 0.014-0.045 in this encoder, i.e. invisible.
#
# Why price the split again when section 18 found it unnecessary. Section 18 split
# two sets performing the *same* motion, and the split cost more than it bought.
# Here the two sets genuinely differ in technique, so the earlier result does not
# carry over -- R1V is the control that says whether it does.
#
# SAVE_TOTAL_LIMIT 6 keeps 20k-30k. Every checkpoint selected so far has come from
# 24k up, and three runs at 8 would want 324 GB of the 472 free.
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
cd "$ROOT"
export PATH="$HOME/micromamba/envs/ffmpeg7/bin:$PATH"
source "$ROOT/.venv/bin/activate"

STEPS="${MAX_STEPS:-30000}"
KEEP="${SAVE_TOTAL_LIMIT:-6}"
WORKERS="${DATALOADER_NUM_WORKERS:-12}"
PORT_BASE="${MASTER_PORT_BASE:-29550}"
port_offset=0

D="$ROOT/datasets/carroll511"
E="$ROOT/examples/unitree_g1_dex1_ikea"

launch() {   # name gpus dataset config
    local name="$1" gpus="$2" ds="$3" cfg="$4"
    if [ ! -f "${ds}_train/meta/stats.json" ]; then
        echo "refusing to launch $name: ${ds}_train has no stats.json" >&2
        return 1
    fi
    echo "[$(date '+%F %T')] launching $name on GPU $gpus port $((PORT_BASE + port_offset))" \
         "<- $(basename "$ds")  [$(basename "$cfg")]"
    CUDA_VISIBLE_DEVICES="$gpus" \
    MASTER_PORT="$((PORT_BASE + port_offset))" \
    CONFIG="$cfg" \
    DATASET_ROOT="$ds" \
    EXP_SUFFIX="_2h_$name" \
    MAX_STEPS="$STEPS" \
    SAVE_TOTAL_LIMIT="$KEEP" \
    NUM_GPUS=2 \
    DATALOADER_NUM_WORKERS="$WORKERS" \
    nohup bash examples/unitree_g1_dex1_ikea/run_finetune_ikea.sh \
        --use-ddp --ddp-comm-bf16 \
        > "$ROOT/datasets/train_2h_$name.log" 2>&1 &
    port_offset=$((port_offset + 1))
    sleep 20
}

NEED=$(( 3 * (KEEP + 1) * 12 ))
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "disk: ${FREE} GB free, three runs need ~${NEED} GB"
if [ "$FREE" -lt "$NEED" ]; then
    echo "ERROR: not enough space for three concurrent runs." >&2
    exit 1
fi

launch rv  0,1 "$D/G1_Dex1_IKEA_all_30hz_twohand"  "$E/g1_dex1_ikea_armvel_config.py"
launch r   2,3 "$D/G1_Dex1_IKEA_all_30hz_twohand"  "$E/g1_dex1_ikea_relarm_3view_aug_config.py"
launch r1v 4,5 "$D/G1_Dex1_IKEA_all_30hz_twohand1" "$E/g1_dex1_ikea_armvel_config.py"

echo "[$(date '+%F %T')] launched; logs in datasets/train_2h_*.log"
