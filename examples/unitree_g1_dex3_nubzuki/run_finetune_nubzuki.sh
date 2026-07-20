#!/usr/bin/env bash
# G1 Dex3 "nubzuki" pick-and-place finetune — real sonic-teleop data, single GPU.
#
# Data: RooibosT/g1-nubzuki-pickandplace-260715, 74 episodes / 73,733 frames
# @50fps, single head cam (ego_view, AV1 360x640), 43-dim state/action
# (12 legs + 3 waist + 7+7 arms + 7+7 dex3 hands). Single pick-and-place task.
# Lower body nearly static (sonic balance output) -> legs state-only.
# Dex3 hands teleoped binary open/close -> ABSOLUTE hand actions.
#
# Recipe (adapted from the validated BCT/redblock large-batch runs):
#   effective batch 256 (16/GPU x accum 16 on 1 GPU), lr 1e-4, warmup 0.05,
#   state dropout 0.2, horizon 40 (native; 0.8s @50fps — same as the built-in
#   unitree_g1_sonic posttrain config), 16 dataloader workers + ffmpeg_threads=1
#   (the oversubscription fix), val split (every 15th episode -> 5 eps) held out
#   for checkpoint selection via eval MSE.
#   Train split: 69 eps ~= 68.8k frames -> ~269 steps/epoch at effective 256.
#   MAX_STEPS 5000 ~= 18.6 epochs. This is a CEILING, not a target: the
#   fps-scaled validated anchor (36 epochs @10fps == ~7.2 epochs @50fps) is
#   ~1,950 steps, so pick best checkpoint from the val curve, not the last one.
#
# 8-hour budget on 1x 90GB GPU:
#   B200-class  ~4.8-5.5 s/it at effective 256 -> 5000 steps ~= 6.7-7.6h  (fits)
#   H100-class  ~9-10 s/it                     -> set MAX_STEPS~=2800     (fits)
#   Probe rule: read sustained s/it around step 100-300 from the log, then
#   MAX_STEPS ~= 27000 / s_it. Restarting with a lower MAX_STEPS is fine —
#   checkpoints and the val curve are the product, not the final step.
#
# Usage:
#   bash examples/unitree_g1_dex3_nubzuki/run_finetune_nubzuki.sh
#
# Speed probe alternative (same effective 256, bigger microbatch):
#   GLOBAL_BATCH_SIZE=32 GRAD_ACCUM=8 bash examples/unitree_g1_dex3_nubzuki/run_finetune_nubzuki.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/NHNHOME/WORKSPACE/chan/.cache/huggingface}"
# Base model + Cosmos backbone/tokenizer resolve from local cache; offline mode
# keeps launches immune to HF 429s. Override with HF_HUB_OFFLINE=0 on cache miss.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

EXAMPLE_DIR=examples/unitree_g1_dex3_nubzuki
CONFIG=$EXAMPLE_DIR/g1_dex3_nubzuki_config.py
MODALITY_JSON=$EXAMPLE_DIR/modality.json
LEROBOT_ROOT="${LEROBOT_ROOT:-/NHNHOME/WORKSPACE/chan/.cache/lerobot}"
DATASET_ROOT="${DATASET_ROOT:-$LEROBOT_ROOT/RooibosT/g1-nubzuki-pickandplace-260715}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/NHNHOME/WORKSPACE/chan/models/GR00T-N1.7-3B}"
TRAIN_DATASET="${DATASET_ROOT}_train"
VAL_DATASET="${DATASET_ROOT}_val"
EXP_NAME="g1_dex3_nubzuki_b256${EXP_SUFFIX:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/NHNHOME/WORKSPACE/chan/outputs/${EXP_NAME}}"
WANDB_PROJECT_NAME=gr00t-n1.7-g1-dex3-nubzuki

# Effective batch 256 = 16/GPU x accum 16 — single-GPU version of the validated
# microbatch shape (2xB200 sweep: microbatch 16 beat 128 on overlap).
# ffmpeg_threads=1 prevents decode-thread oversubscription (RETRAIN_NOTES.md §6-9).
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_STEPS="${MAX_STEPS:-5000}"                   # ceiling ~18.6 epochs; see header
SAVE_STEPS="${SAVE_STEPS:-500}"
NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
STATE_DROPOUT="${STATE_DROPOUT:-0.2}"
EP_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.2}"
SHARD="${SHARD_SIZE:-1024}"
FFMPEG_THREADS="${DATALOADER_FFMPEG_THREADS:-1}"

# The GR00T loader needs LeRobot v2.1 layout (episodes.jsonl / per-episode files).
# The hub dataset is v3.0 — convert once with scripts/lerobot_conversion.
if [ ! -f "$DATASET_ROOT/meta/episodes.jsonl" ]; then
    echo "ERROR: $DATASET_ROOT is not in LeRobot v2.1 layout (meta/episodes.jsonl missing)." >&2
    echo "Convert it first (uses the dedicated conversion venv; downloads from the Hub):" >&2
    echo "  cd scripts/lerobot_conversion && .venv/bin/python convert_v3_to_v2.py \\" >&2
    echo "    --repo-id RooibosT/g1-nubzuki-pickandplace-260715 --root $LEROBOT_ROOT" >&2
    exit 1
fi

# GR00T-side modality map (state/action slices, camera naming, task annotation).
# Copied into the dataset so the split + loader can find it at meta/modality.json.
cp "$MODALITY_JSON" "$DATASET_ROOT/meta/modality.json"

# Train/val split: every 15th episode held out (5 val eps of 74) for
# checkpoint selection — critical here, 74 demos overfit well before 5k steps.
if [ ! -d "$TRAIN_DATASET" ]; then
    echo "Train split not found — creating ${TRAIN_DATASET} (+_val) ..."
    python examples/unitree_g1_dex1/split_train_val.py --src "$DATASET_ROOT" --val-every 15
else
    cp "$MODALITY_JSON" "$TRAIN_DATASET/meta/modality.json"
    cp "$MODALITY_JSON" "$VAL_DATASET/meta/modality.json"
fi

# Absolute stats for both splits (relative stats regenerate at training start).
for DS in "$TRAIN_DATASET" "$VAL_DATASET"; do
    if [ ! -f "$DS/meta/stats.json" ]; then
        echo "Generating statistics for ${DS} ..."
        python gr00t/data/stats.py \
            --dataset-path "$DS" \
            --embodiment-tag NEW_EMBODIMENT \
            --modality-config-path "$CONFIG"
    fi
done

mkdir -p "$OUTPUT_DIR"
echo "=== g1_dex3 nubzuki finetune ==="
echo "  config          : $CONFIG"
echo "  dataset         : $TRAIN_DATASET"
echo "  effective batch : $((GLOBAL_BATCH_SIZE * GRAD_ACCUM)) (global $GLOBAL_BATCH_SIZE x accum $GRAD_ACCUM)"
echo "  steps           : $MAX_STEPS (save every $SAVE_STEPS)"
echo "  output          : $OUTPUT_DIR"

env \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    NUM_GPUS=1 USE_WANDB=1 GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    DATALOADER_NUM_WORKERS="$NUM_WORKERS" EPISODE_SAMPLING_RATE="$EP_SAMPLING_RATE" \
    SHARD_SIZE="$SHARD" MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
    DATALOADER_FFMPEG_THREADS="$FFMPEG_THREADS" \
    bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$TRAIN_DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$CONFIG" \
    --state-dropout-prob "$STATE_DROPOUT" \
    --wandb-project "$WANDB_PROJECT_NAME" \
    --experiment-name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --save-only-model \
    -- --save-total-limit 20 --gradient-accumulation-steps "$GRAD_ACCUM" \
    --val-dataset-path "$VAL_DATASET" --eval-steps "${EVAL_STEPS:-500}" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
