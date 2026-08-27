#!/usr/bin/env bash
# One 2-GPU run on the enlarged three-task set: does more `insert` data fix `insert`?
#
#   LEG  60-dim arm velocity   GPU 6,7   vs g1_dex1_ikea_relarm_3view_aug_b64_v2_armvel
#
# The data. URL-RFM/IKEA_pickuptheleg is the original recording with a second
# block appended -- 178 episodes become 322, 79,205 frames become 149,437 -- shot
# with deliberate z-axis motion during the insert. `rotate table base` and
# `flip table` are deliberately absent: this run answers whether `insert`
# improves, and the two new tasks are the confound §18 measured at -12.06% on it.
#
# The val split is byte-for-byte the old one. Episodes 0-177 of the new export are
# the original 178 in the same order and with the same lengths, so `--val-sessions
# 3,6,19` still selects the same 15 episodes and the same 7,490 frames that every
# earlier number was measured on. All 144 new episodes go to train. That buys a
# direct comparison against the existing armvel control and costs the ability to
# see whether the new z-motion style itself is learned -- val cannot show that.
#
# ⚠️ The export's `file_index` is wrong and the converter repairs it. Both high
# cameras label episodes 0-177 as file_index 1 when the first 91 are physically in
# file-000; each pair of files was collapsed onto the pair's last index. The wrist
# cameras, with two files instead of four, came out right, so the damage is silent
# and partial. `resolve_file_index` derives the mapping from where the timestamp
# resets to 0 instead, which agrees with the recorded value on both older sources
# and reconstructs each physical file's frame count exactly (42,943 / 36,262 /
# 45,946 / 24,286 for cam_left_high, summing to its episodes' lengths). Verify
# then decoded 240 frames through the repaired mapping at worst 1.618/255.
#
# 40k steps, not 30k. The window count went 65,358 -> 129,974 (1.99x), so at
# effective batch 64:
#     20k = 9.8 epochs      30k = 14.8 epochs      40k = 19.7 epochs
# and the control got 19.6. 40k restores the control's epoch count exactly, which
# makes this "does more data help per epoch" rather than "per step". Keeping 10
# checkpoints spans 22k-40k, so the equal-step reading at 30k is available too.
#   11 x 12 GB = 132 GB.
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
cd "$ROOT"
export PATH="$HOME/micromamba/envs/ffmpeg7/bin:$PATH"
source "$ROOT/.venv/bin/activate"

STEPS="${MAX_STEPS:-40000}"
KEEP="${SAVE_TOTAL_LIMIT:-10}"
WORKERS="${DATALOADER_NUM_WORKERS:-12}"
GPUS="${GPUS:-6,7}"
PORT="${MASTER_PORT:-29543}"

DS="$ROOT/datasets/carroll511/G1_Dex1_IKEA_leg_30hz"
CFG="$ROOT/examples/unitree_g1_dex1_ikea/g1_dex1_ikea_armvel_config.py"

if [ ! -f "${DS}_train/meta/stats.json" ]; then
    echo "refusing to launch: ${DS}_train has no stats.json" >&2
    exit 1
fi

NEED=$(( (KEEP + 1) * 12 ))
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "disk: ${FREE} GB free now, this run needs ~${NEED} GB on top of whatever else is running"
if [ "$FREE" -lt $(( NEED + 150 )) ]; then
    echo "ERROR: too tight next to the pnp runs still writing checkpoints." >&2
    exit 1
fi

echo "[$(date '+%F %T')] launching leg on GPU $GPUS port $PORT <- $(basename "$DS")  [$(basename "$CFG")]"
CUDA_VISIBLE_DEVICES="$GPUS" \
MASTER_PORT="$PORT" \
CONFIG="$CFG" \
DATASET_ROOT="$DS" \
EXP_SUFFIX="_leg_armvel" \
MAX_STEPS="$STEPS" \
SAVE_TOTAL_LIMIT="$KEEP" \
NUM_GPUS=2 \
DATALOADER_NUM_WORKERS="$WORKERS" \
nohup bash examples/unitree_g1_dex1_ikea/run_finetune_ikea.sh \
    --use-ddp --ddp-comm-bf16 \
    > "$ROOT/datasets/train_leg_armvel.log" 2>&1 &

echo "[$(date '+%F %T')] launched; log in datasets/train_leg_armvel.log"
