#!/usr/bin/env bash
# One 2-GPU run: the pick-and-place merge with arm velocity AND arm joint torque.
#
#   PVT  74-dim state   GPU 4,5   vs the pnp_pv run on GPU 2,3
#
# The only difference from pnp_pv is `left_arm_torque` and `right_arm_torque`.
# The dataset is built from URL-RFM/IKEA_table_assembly_torque, a re-export of the
# same 178 episodes: state[:86] and action[:19] are bit-identical to the original
# and the mp4s are the same files, so the merge groups came out identical (129
# output episodes, the same 14 refused boundaries) and the clips are hardlinked
# rather than re-encoded. The rotate-table and flip sources already carried the
# 117-dim state, so the merged set needed no zero-fill anywhere.
#
# Torque is worth the dims: only R2 0.65/0.68 of arm torque is recoverable from
# the 60-dim arm-velocity state, against the 99.85% that got torso_gravity
# dropped. It also opens a shortcut -- see the config docstring for the decay
# table -- so this run is judged on the late chunk steps and the gripper.
#
# SAVE_TOTAL_LIMIT 5 rather than 8, purely for disk: pnp_p and pnp_pv are already
# holding 216 GB of the 347 GB that was free. 22k-30k still covers the band every
# selected checkpoint has come from.
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
cd "$ROOT"
export PATH="$HOME/micromamba/envs/ffmpeg7/bin:$PATH"
source "$ROOT/.venv/bin/activate"

STEPS="${MAX_STEPS:-30000}"
KEEP="${SAVE_TOTAL_LIMIT:-5}"
WORKERS="${DATALOADER_NUM_WORKERS:-12}"
GPUS="${GPUS:-4,5}"
PORT="${MASTER_PORT:-29542}"

DS="$ROOT/datasets/carroll511/G1_Dex1_IKEA_all_30hz_pnptq"
CFG="$ROOT/examples/unitree_g1_dex1_ikea/g1_dex1_ikea_armvel_torque_config.py"

if [ ! -f "${DS}_train/meta/stats.json" ]; then
    echo "refusing to launch: ${DS}_train has no stats.json" >&2
    exit 1
fi

NEED=$(( (KEEP + 1) * 12 ))
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "disk: ${FREE} GB free now, this run needs ~${NEED} GB on top of whatever else is running"
if [ "$FREE" -lt $(( NEED + 150 )) ]; then
    echo "ERROR: too tight next to the two pnp runs still writing checkpoints." >&2
    exit 1
fi

echo "[$(date '+%F %T')] launching pvt on GPU $GPUS port $PORT <- $(basename "$DS")  [$(basename "$CFG")]"
CUDA_VISIBLE_DEVICES="$GPUS" \
MASTER_PORT="$PORT" \
CONFIG="$CFG" \
DATASET_ROOT="$DS" \
EXP_SUFFIX="_pnp_pvt" \
MAX_STEPS="$STEPS" \
SAVE_TOTAL_LIMIT="$KEEP" \
NUM_GPUS=2 \
DATALOADER_NUM_WORKERS="$WORKERS" \
nohup bash examples/unitree_g1_dex1_ikea/run_finetune_ikea.sh \
    --use-ddp --ddp-comm-bf16 \
    > "$ROOT/datasets/train_pnp_pvt.log" 2>&1 &

echo "[$(date '+%F %T')] launched; log in datasets/train_pnp_pvt.log"
