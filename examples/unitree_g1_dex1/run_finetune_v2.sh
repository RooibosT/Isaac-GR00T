#!/usr/bin/env bash
# G1 Dex1 pick-redblock retrain — validated large-batch recipe on 2xB200.
#
# Recipe (mirrors the cs1032 batch-ablation-validated setup, adapted to 30fps data):
#   effective batch 256 (per-device 128 x 2 GPUs, grad accum 1), lr 1e-4,
#   state dropout 0.1, horizon-40 config (1.33s lookahead @30fps),
#   train split only (val held out for checkpoint selection),
#   ~13k steps (~36 epochs @ effective 256), save every 1000, keep all checkpoints.
#
# Usage:
#   bash examples/unitree_g1_dex1/run_finetune_v2.sh [a40|b40|a|b]   # default a40
#
# OOM fallback (numerically identical effective batch):
#   GLOBAL_BATCH_SIZE=128 GRAD_ACCUM=2 bash examples/unitree_g1_dex1/run_finetune_v2.sh a40

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export HF_HOME=/NHNHOME/WORKSPACE/chan/.cache/huggingface
# Everything (base model local, Cosmos backbone + tokenizer cached) resolves offline;
# staying offline makes launches immune to HF 429 rate limits. Override with
# HF_HUB_OFFLINE=0 if a cache miss ever needs a fresh download.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

VARIANT="${1:-a40}"
case "$VARIANT" in
    a40) CONFIG=examples/unitree_g1_dex1/variants/g1_dex1_config_a40.py ;;
    b40) CONFIG=examples/unitree_g1_dex1/variants/g1_dex1_config_b40.py ;;
    a)   CONFIG=examples/unitree_g1_dex1/g1_dex1_config.py ;;
    b)   CONFIG=examples/unitree_g1_dex1/variants/g1_dex1_config_b.py ;;
    *)   echo "variant must be a40|b40|a|b" >&2; exit 1 ;;
esac

BASE_MODEL_PATH=/NHNHOME/WORKSPACE/chan/models/GR00T-N1.7-3B
DATASET_ROOT=demo_data/RooibosT/g1_pick_redblock_dex1_sim_merged_107demo
TRAIN_DATASET="${DATASET_ROOT}_train"
VAL_DATASET="${DATASET_ROOT}_val"
# EXP_SUFFIX lets speed probes / variants write to a distinct dir without clobbering
# a completed run (e.g. EXP_SUFFIX=_shard256 MAX_STEPS=300 for a throughput probe).
EXP_NAME="g1_dex1_v2_${VARIANT}_b256${EXP_SUFFIX:-}"
OUTPUT_DIR="/NHNHOME/WORKSPACE/chan/outputs/${EXP_NAME}"
WANDB_PROJECT_NAME=gr00t-n1.7-g1-dex1

# Effective batch 256. Default shape = micro-batch 16/GPU x accum 8. Measured facts
# (production threads, OMP=2): dataloader raw capacity ~190 samples/s/rank >> observed
# ~46/rank, GPU util ~40% -> neither decode/processor nor compute saturates. The real
# ceiling is the shard-caching BURST STALL: a worker decodes a whole 1024-sample shard
# (~86s, 76% decode) while serving the current one in ~20s -> ~66s stall, round-robin
# blocks. Levers: SHARD_SIZE down (shorter/smoother bursts, free), pre-decode RAM cache
# (removes 76% of the burst -> big win). Once the stall is gone the run becomes
# compute-bound -> then GLOBAL_BATCH_SIZE=128 GRAD_ACCUM=1 is faster (fewer accum steps).
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"    # per-device = this / NUM_GPUS
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_STEPS="${MAX_STEPS:-13000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
EP_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.2}"
SHARD="${SHARD_SIZE:-1024}"                      # smaller = shorter caching bursts

# Refuse to start while another training run is still using the GPUs.
if [ "${FORCE:-0}" != "1" ]; then
    BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "$BUSY" -gt 0 ]; then
        echo "GPUs are busy ($BUSY compute process(es) running)." >&2
        echo "Wait for the current run to finish, or re-run with FORCE=1." >&2
        exit 1
    fi
fi

# Train/val split must exist (created by split_train_val.py, val held out for eval).
if [ ! -d "$TRAIN_DATASET" ]; then
    echo "Train split not found — creating ${TRAIN_DATASET} (+_val) ..."
    python examples/unitree_g1_dex1/split_train_val.py --src "$DATASET_ROOT"
fi

# Absolute stats for the train split (relative stats are regenerated at training start).
if [ ! -f "$TRAIN_DATASET/meta/stats.json" ]; then
    echo "Generating statistics for ${TRAIN_DATASET} ..."
    python gr00t/data/stats.py \
        --dataset-path "$TRAIN_DATASET" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "$CONFIG"
fi

# Stats for the val split (its episode loader requires them; normalization of the
# eval samples still uses the merged train stats via the training processor).
if [ ! -f "$VAL_DATASET/meta/stats.json" ]; then
    echo "Generating statistics for ${VAL_DATASET} ..."
    python gr00t/data/stats.py \
        --dataset-path "$VAL_DATASET" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "$CONFIG"
fi

mkdir -p "$OUTPUT_DIR"
echo "=== g1_dex1 v2 retrain [$VARIANT] ==="
echo "  config          : $CONFIG"
echo "  effective batch : $((GLOBAL_BATCH_SIZE * GRAD_ACCUM)) (global $GLOBAL_BATCH_SIZE x accum $GRAD_ACCUM)"
echo "  steps           : $MAX_STEPS (save every $SAVE_STEPS)"
echo "  output          : $OUTPUT_DIR"

# No numactl node binding: 16 workers x 2 ranks of video decode need both NUMA nodes.
env \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    NUM_GPUS=2 USE_WANDB=1 GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    DATALOADER_NUM_WORKERS="$NUM_WORKERS" EPISODE_SAMPLING_RATE="$EP_SAMPLING_RATE" \
    SHARD_SIZE="$SHARD" MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
    bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$TRAIN_DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$CONFIG" \
    --state-dropout-prob 0.1 \
    --wandb-project "$WANDB_PROJECT_NAME" \
    --experiment-name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --save-only-model \
    -- --save-total-limit 20 --gradient-accumulation-steps "$GRAD_ACCUM" \
    --val-dataset-path "$VAL_DATASET" --eval-steps "${EVAL_STEPS:-1000}" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
