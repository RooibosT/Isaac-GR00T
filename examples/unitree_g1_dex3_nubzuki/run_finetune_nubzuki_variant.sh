#!/usr/bin/env bash
# G1 Dex3 "nubzuki" finetune — ablation variants (representation x horizon).
#
# The baseline (ABS arms, horizon 40) runs via run_finetune_nubzuki.sh. This
# launcher runs the other three cells of the 2x2 sweep on spare GPUs:
#
#   rel_h40   RELATIVE arms (waist ABS, hands ABS), horizon 40   -> deploy apc=40
#   abs_h16   ABSOLUTE arms,                        horizon 16   -> deploy apc=16
#   rel_h16   RELATIVE arms (waist ABS, hands ABS), horizon 16   -> deploy apc=16
#
# Same recipe as the baseline: effective batch 256 (microbatch 16 x accum 16 on
# ONE GPU, ~39 GB), lr 1e-4, warmup 0.05, state dropout 0.2, 16 workers +
# ffmpeg_threads=1, MAX_STEPS 5000 ceiling (pick best checkpoint from the val
# curve, not the last step). ~2.2 s/it observed -> ~3 h/variant.
#
# It reuses the baseline's train/val split (created by run_finetune_nubzuki.sh)
# WITHOUT copying the 584 MB of video: each variant gets its own dataset dir with
# data/ + videos/ symlinked to the baseline split and a PRIVATE meta/ copy. The
# private meta is the whole point — a RELATIVE arm's relative_stats.json is
# horizon-specific but keyed by plain joint name (left_arm/right_arm), so two REL
# variants sharing a meta/ would silently clobber each other's normalization.
#
# Usage (run the baseline abs_h40 first so the split exists):
#   CUDA_VISIBLE_DEVICES=0 bash examples/unitree_g1_dex3_nubzuki/run_finetune_nubzuki_variant.sh rel_h40
#   CUDA_VISIBLE_DEVICES=1 bash examples/unitree_g1_dex3_nubzuki/run_finetune_nubzuki_variant.sh abs_h16
#   CUDA_VISIBLE_DEVICES=2 bash examples/unitree_g1_dex3_nubzuki/run_finetune_nubzuki_variant.sh rel_h16
#
# 90 GB-server paths (shared FS mounts at a different point there): prefix the
# HF_HOME / LEROBOT_ROOT / BASE_MODEL_PATH / OUTPUT_DIR env vars, same as the
# baseline runner — no need to edit this file.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# venv is machine-specific (its activate hardcodes an absolute VIRTUAL_ENV that
# only resolves on the box it was built on). Default .venv suits the B200 box;
# on the ws/external GPU boxes pass VENV=.venv-45gb_1 (or _2 / a 90 GB env).
source "${VENV:-.venv}/bin/activate"
export HF_HOME="${HF_HOME:-/NHNHOME/WORKSPACE/chan/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

EXAMPLE_DIR=examples/unitree_g1_dex3_nubzuki
MODALITY_JSON=$EXAMPLE_DIR/modality.json   # dim slices are identical for every variant

VARIANT="${1:-}"
case "$VARIANT" in
    rel_h40) CONFIG=$EXAMPLE_DIR/g1_dex3_nubzuki_rel_config.py;     ACTIONS_PER_CHUNK=40 ;;
    abs_h16) CONFIG=$EXAMPLE_DIR/g1_dex3_nubzuki_h16_config.py;     ACTIONS_PER_CHUNK=16 ;;
    rel_h16) CONFIG=$EXAMPLE_DIR/g1_dex3_nubzuki_rel_h16_config.py; ACTIONS_PER_CHUNK=16 ;;
    *) echo "usage: $0 <rel_h40|abs_h16|rel_h16>" >&2; exit 1 ;;
esac

LEROBOT_ROOT="${LEROBOT_ROOT:-/NHNHOME/WORKSPACE/chan/.cache/lerobot}"
DATASET_ROOT="${DATASET_ROOT:-$LEROBOT_ROOT/RooibosT/g1-nubzuki-pickandplace-260715}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/NHNHOME/WORKSPACE/chan/models/GR00T-N1.7-3B}"

BASE_TRAIN="${DATASET_ROOT}_train"
BASE_VAL="${DATASET_ROOT}_val"
if [ ! -d "$BASE_TRAIN" ] || [ ! -d "$BASE_VAL" ]; then
    echo "ERROR: baseline split not found ($BASE_TRAIN / $BASE_VAL)." >&2
    echo "Run the ABS h40 baseline first — it creates the split this reuses:" >&2
    echo "  bash examples/unitree_g1_dex3_nubzuki/run_finetune_nubzuki.sh" >&2
    exit 1
fi

# Per-variant dataset dir: heavy data/videos symlinked to the baseline split,
# private meta/ so each variant's stats (esp. relative_stats.json) stay isolated.
# The baseline meta is never written to, so the running baseline job is untouched.
#
# We deliberately do NOT copy stats.json: the baseline runner rewrites it via an
# atomic tmp-file (mode 0600), and on the shared FS the baseline job runs as root,
# so it is unreadable to a normal user -> cp would abort under `set -e`. Instead we
# let stats.py recompute absolute stats from the (symlinked) parquet — cheap
# (~seconds; data/ is ~100 MB) and identical, since it is the same split.
setup_variant_split () {  # $1 = baseline split dir, $2 = variant split dir
    local base="$1" var="$2"
    mkdir -p "$var/meta"
    ln -sfn "$base/data" "$var/data"
    ln -sfn "$base/videos" "$var/videos"
    for f in info.json episodes.jsonl tasks.jsonl; do
        cp -f "$base/meta/$f" "$var/meta/$f"
    done
    cp -f "$MODALITY_JSON" "$var/meta/modality.json"
    # Start clean so this variant's horizon/rep recomputes its own stats.
    rm -f "$var/meta/stats.json" "$var/meta/relative_stats.json"
}

TRAIN_DATASET="${BASE_TRAIN}__${VARIANT}"
VAL_DATASET="${BASE_VAL}__${VARIANT}"
setup_variant_split "$BASE_TRAIN" "$TRAIN_DATASET"
setup_variant_split "$BASE_VAL" "$VAL_DATASET"

EXP_NAME="g1_dex3_nubzuki_b256_${VARIANT}${EXP_SUFFIX:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/NHNHOME/WORKSPACE/chan/outputs/${EXP_NAME}}"
WANDB_PROJECT_NAME=gr00t-n1.7-g1-dex3-nubzuki

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"   # ~39 GB at horizon 40; fits a 45 GB GPU
GRAD_ACCUM="${GRAD_ACCUM:-16}"                 # effective batch 256
MAX_STEPS="${MAX_STEPS:-5000}"                 # ceiling ~18.6 epochs; best ckpt by val curve
SAVE_STEPS="${SAVE_STEPS:-500}"
NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
STATE_DROPOUT="${STATE_DROPOUT:-0.2}"
EP_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.2}"
SHARD="${SHARD_SIZE:-1024}"
FFMPEG_THREADS="${DATALOADER_FFMPEG_THREADS:-1}"

# Absolute stats are copied from the baseline (fingerprint-fresh -> no recompute);
# relative stats (REL variants only) are computed here into the private meta.
for DS in "$TRAIN_DATASET" "$VAL_DATASET"; do
    echo "Generating statistics for ${DS} ..."
    python gr00t/data/stats.py \
        --dataset-path "$DS" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "$CONFIG"
done

mkdir -p "$OUTPUT_DIR"
echo "=== g1_dex3 nubzuki variant [$VARIANT] ==="
echo "  config          : $CONFIG"
echo "  dataset         : $TRAIN_DATASET"
echo "  deploy apc      : $ACTIONS_PER_CHUNK (client --actions_per_chunk)"
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
