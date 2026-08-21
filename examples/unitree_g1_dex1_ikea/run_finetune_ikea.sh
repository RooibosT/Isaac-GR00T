#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on the IKEA table-assembly dataset (G1 + Dex1, stationary).
#
# Recipe carried over from the BCT winner (examples/unitree_g1_dex1_bct/EXPERIMENTS.md):
# RELATIVE arms + ABSOLUTE grippers, 3 views, 46-dim augmented state, horizon 40,
# state_dropout 0.2, lr 1e-4, warmup 0.05. What changes is the *scale*.
#
# This dataset is 59,589 train frames — 8.4x smaller than BCT's 500k and 3.5x
# larger than the brainco set CK-Sung deployed from (17k frames, 43 epochs,
# effective batch 64). Effective batch is therefore 64, not BCT's 192: at 64 one
# epoch is 931 optimizer steps, so the schedule still gets enough steps to
# converge instead of the ~310 an epoch would buy at 192.
#
# Checkpoints every 2,000 steps (~2.1 epochs) exist to be *scanned*. eval_loss
# rose after ~7.5k steps in all five BCT runs while open-loop accuracy kept
# improving, so it is not a selection signal — pick with eval_val_mse.py.
#
# Usage:
#   bash examples/unitree_g1_dex1_ikea/run_finetune_ikea.sh [extra launch_finetune args]
# Common overrides:
#   NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 MAX_STEPS=20000 bash ... --use-ddp
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLE_DIR="$REPO_ROOT/examples/unitree_g1_dex1_ikea"
EXTRA_ARGS=("$@")

# torchcodec needs the ffmpeg 7 shared libs; the venv ships none.
if [ -d "$HOME/micromamba/envs/ffmpeg7/lib" ]; then
    export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Ablations are config-only: the dataset carries all 86 state dims and
# meta/stats.json covers every one of them, so adding or dropping a state block
# needs no data rebuild and no stats regeneration. Override CONFIG + EXP_SUFFIX.
CONFIG="${CONFIG:-$EXAMPLE_DIR/g1_dex1_ikea_relarm_3view_aug_config.py}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/carroll511/G1_Dex1_IKEA_table_30hz}"
# Point this at a finished checkpoint to warm-start from it. Note that a real
# HF --resume-from-checkpoint is NOT possible for these runs: they save with
# --save-only-model, so optimizer.pt / scheduler.pt / rng_state.pth are absent.
# Resuming would also rebuild the cosine schedule for the new max_steps and jump
# the LR from ~0 (a completed 20k cosine ends near 1e-12) back up to ~2.7e-5,
# with Adam moments reset to zero. Warm-starting as a fresh short run with an
# explicitly chosen low LR is the controllable version of that.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$REPO_ROOT/models/GR00T-N1.7-3B}"
TRAIN_DATASET="${DATASET_ROOT}_train"
VAL_DATASET="${DATASET_ROOT}_val"

# effective batch 64 = global 16 (8/GPU on 2 GPUs) x accum 4.
#
# 2 GPUs, not 4. At a *fixed* effective batch more GPUs do not add work per
# optimizer step, they only split the same 64 samples further, while the
# 1.62B-param gradient all-reduce stays once per step and gets more expensive
# with more peers — this host has no NVLink, so it rides PCIe/SYS. Measured
# here at effective batch 64, 30 steps each (s/step, steady state):
#
#   1 GPU  micro 8  x accum 8              2.46
#   2 GPU  global 16 x accum 4             1.55   + --ddp-comm-bf16 -> 1.46  <-- best
#   4 GPU  global 32 x accum 2             3.98   + --ddp-comm-bf16 -> 2.24
#
# bf16 gradient compression halves all-reduce traffic and is worth 6% at 2 GPUs
# but 44% at 4 — confirming communication, not compute, is the limit. Cutting
# dataloader workers 16 -> 8 changed the 4-GPU number by 0.03 s, so it is not
# CPU contention. Always pass `--use-ddp --ddp-comm-bf16`.
#
# The spare GPUs are better spent on parallel ablations (3 concurrent 2-GPU runs
# on 0-1 / 2-3 / 4-5) than on widening one run; drop DATALOADER_NUM_WORKERS to
# ~10 in that case, 96 cores are shared.
#
# Per-GPU micro-batch stays at 8, the shape the BCT runs fit in 80 GB with three
# 480x640 video streams.
NUM_GPUS="${NUM_GPUS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_STEPS="${MAX_STEPS:-20000}"          # 931 steps/epoch -> ~21.5 epochs
SAVE_STEPS="${SAVE_STEPS:-2000}"
NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
STATE_DROPOUT="${STATE_DROPOUT:-0.2}"
# The repo default, never validated on a precision task. Measured on this
# dataset against the frozen Cosmos-Reason2 encoder: this setting alone moves
# the image features 64% as far as a different scene does, and crop+jitter
# together 78%. Since tune_visual=False the encoder learns no invariance from
# it — the jitter is feature-space noise regularizing the action expert, so
# "stronger" is not automatically better. Halving it is ablation P2.
COLOR_JITTER_PARAMS="${COLOR_JITTER_PARAMS:-brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08}"
EP_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.2}"
SHARD="${SHARD_SIZE:-1024}"
FFMPEG_THREADS="${DATALOADER_FFMPEG_THREADS:-1}"

EXP_NAME="g1_dex1_ikea_relarm_3view_aug_b$((GLOBAL_BATCH_SIZE * GRAD_ACCUM))${EXP_SUFFIX:-}"
OUTPUT_DIR="$REPO_ROOT/outputs/${EXP_NAME}"
WANDB_PROJECT_NAME=gr00t-n1.7-g1-dex1-ikea

for DS in "$TRAIN_DATASET" "$VAL_DATASET"; do
    [ -f "$DS/meta/episodes.jsonl" ] || {
        echo "ERROR: $DS missing (LeRobot v2.1 layout). Convert first:" >&2
        echo "  python $EXAMPLE_DIR/convert_ikea_v3_to_v2.py \\" >&2
        echo "    --src $REPO_ROOT/datasets/carroll511/IKEA_table_assembly \\" >&2
        echo "    --out $DATASET_ROOT --sessions-file $EXAMPLE_DIR/sessions.json \\" >&2
        echo "    --val-sessions 3,12,19,23" >&2
        exit 1; }
    if [ ! -f "$DS/meta/stats.json" ]; then
        echo "Generating statistics for ${DS} ..."
        python -m gr00t.data.stats --dataset-path "$DS" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CONFIG"
    fi
done

mkdir -p "$OUTPUT_DIR"

# Preflight on free space. A checkpoint here is ~12 GB and --save-only-model
# still writes the full 3.14B-param model each time, so a 20k/2k run needs
# ~120 GB on its own; several concurrent runs fill a disk quietly and the first
# symptom is a SafetensorError mid-save that kills the run outright, hours in.
CKPT_GB="${CKPT_GB:-12}"
NEED_GB=$(( (MAX_STEPS / SAVE_STEPS + 1) * CKPT_GB ))
FREE_GB=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
    echo "ERROR: need ~${NEED_GB} GB for $((MAX_STEPS / SAVE_STEPS + 1)) checkpoints, only ${FREE_GB} GB free." >&2
    echo "       Free space or raise SAVE_STEPS, then retry. Set SKIP_DISK_CHECK=1 to override." >&2
    [ "${SKIP_DISK_CHECK:-0}" = "1" ] || exit 1
fi
echo "disk: ${FREE_GB} GB free, this run needs ~${NEED_GB} GB"

echo "=== g1_dex1 IKEA finetune ==="
echo "  config          : $CONFIG"
echo "  dataset         : $TRAIN_DATASET"
echo "  gpus            : $NUM_GPUS (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all})"
echo "  effective batch : $((GLOBAL_BATCH_SIZE * GRAD_ACCUM)) (global $GLOBAL_BATCH_SIZE x accum $GRAD_ACCUM)"
echo "  steps           : $MAX_STEPS (save every $SAVE_STEPS)"
echo "  output          : $OUTPUT_DIR"

env \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    NUM_GPUS="$NUM_GPUS" USE_WANDB="${USE_WANDB:-1}" GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    DATALOADER_NUM_WORKERS="$NUM_WORKERS" EPISODE_SAMPLING_RATE="$EP_SAMPLING_RATE" \
    SHARD_SIZE="$SHARD" MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
    DATALOADER_FFMPEG_THREADS="$FFMPEG_THREADS" \
    COLOR_JITTER_PARAMS="$COLOR_JITTER_PARAMS" \
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
    -- --save-total-limit "${SAVE_TOTAL_LIMIT:-20}" --gradient-accumulation-steps "$GRAD_ACCUM" \
    --val-dataset-path "$VAL_DATASET" --eval-steps "${EVAL_STEPS:-2000}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    2>&1 | tee "$OUTPUT_DIR/train.log"
