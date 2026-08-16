#!/usr/bin/env bash
# G1 Dex1 "Building Children Table" (BCT) subtask finetune — real teleop data.
#
# Data: BitRobot/G1_WBT_Dex1_Building-Children-Table, segmented into 5 manipulation
# subtasks (flip table / rotate table base / insert leg / pick leg / rotate to tighten),
# 1338 episodes, 270,668 frames @15fps, frame-balanced ~54k frames (20%) per subtask.
# Upper body teleoped via xr_teleoperate, lower body via HOMIE (legs mostly static;
# knees adjust height in ~24% of episodes, ankles carry balance chatter).
#
# Recipe (adapted from the validated redblock large-batch run, run_finetune_v2.sh):
#   effective batch 256 (global 32 x accum 8 on 2 GPUs), lr 1e-4, warmup 0.05,
#   state dropout 0.2 (repo default; whole-state zeroing — favors visual grounding,
#   sensible for real deployment + legs-in-state; redblock used 0.1 but that was a
#   cs1032 replication choice, not an ablation winner), horizon 40 (2.67s @15fps),
#   16 dataloader workers/GPU + ffmpeg_threads=1 (the oversubscription fix),
#   val split (every 20th episode) held out for checkpoint selection via eval MSE.
#   817 steps/epoch on the train split (209k effective horizon-40 samples)
#   -> default 25,000 steps ~= 31 epochs (prior validated recipe was 36).
#
# Usage:
#   bash examples/unitree_g1_dex1_bct/run_finetune_bct.sh [joint|joint_h16|joint_wbc|ee]   # default joint
#
# OOM fallback (numerically identical effective batch):
#   GLOBAL_BATCH_SIZE=16 GRAD_ACCUM=16 bash examples/unitree_g1_dex1_bct/run_finetune_bct.sh joint

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
# torchcodec needs FFmpeg<8 shared libs; this host has no system FFmpeg, so they
# come from a user-space conda-forge env (no sudo needed).
if [ -d "$HOME/micromamba/envs/ffmpeg7/lib" ]; then
    export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Offline mode keeps launches immune to HF 429s once the Cosmos backbone/tokenizer
# are cached; first run on a fresh machine must be online (HF_HUB_OFFLINE=0).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

VARIANT="${1:-joint}"
shift $(( $# > 0 ? 1 : 0 ))
EXTRA_ARGS=("$@")   # forwarded verbatim to launch_finetune (e.g. --use-ddp)
EXAMPLE_DIR=examples/unitree_g1_dex1_bct
case "$VARIANT" in
    joint)
        CONFIG=$EXAMPLE_DIR/g1_dex1_bct_joint_config.py
        MODALITY_JSON=$EXAMPLE_DIR/modality_joint.json
        DATASET_ROOT=$REPO_ROOT/datasets/RooibosT/G1_Dex1_BCT_subtask_joint
        ;;
    joint_h16)
        CONFIG=$EXAMPLE_DIR/g1_dex1_bct_joint_h16_config.py
        MODALITY_JSON=$EXAMPLE_DIR/modality_joint.json
        DATASET_ROOT=$REPO_ROOT/datasets/RooibosT/G1_Dex1_BCT_subtask_joint
        ;;
    joint_wbc)
        CONFIG=$EXAMPLE_DIR/g1_dex1_bct_joint_wbc_config.py
        MODALITY_JSON=$EXAMPLE_DIR/modality_joint.json
        DATASET_ROOT=$REPO_ROOT/datasets/RooibosT/G1_Dex1_BCT_subtask_joint
        ;;
    joint_30hz)
        # 30 fps re-export (fps_divisor=1): same 31-dim layout and modality map as
        # `joint`, so the config is shared. Horizon 40 now spans 1.33 s (was 2.67 s).
        # ~465k effective horizon-40 train samples -> ~1815 steps/epoch at eff. 256;
        # default 50k steps ~= 27 epochs (the 15 fps recipe's 25k ~= 31 epochs).
        CONFIG=$EXAMPLE_DIR/g1_dex1_bct_joint_config.py
        MODALITY_JSON=$EXAMPLE_DIR/modality_joint.json
        DATASET_ROOT=$REPO_ROOT/datasets/RooibosT/G1_Dex1_BCT_subtask_joint_30hz
        DEFAULT_MAX_STEPS=50000
        ;;
    ee)
        CONFIG=$EXAMPLE_DIR/g1_dex1_bct_ee_config.py
        MODALITY_JSON=$EXAMPLE_DIR/modality_ee.json
        DATASET_ROOT=$REPO_ROOT/datasets/RooibosT/G1_Dex1_BCT_subtask_ee
        ;;
    joint_30hz_relarm_4view)
        # RELATIVE arms + 4 views (adds cam_head_right). Long schedule: select by
        # open-loop scan, not eval_loss (EXPERIMENTS.md).
        CONFIG=$EXAMPLE_DIR/g1_dex1_bct_joint_relarm_4view_config.py
        MODALITY_JSON=$EXAMPLE_DIR/modality_joint.json
        DATASET_ROOT=$REPO_ROOT/datasets/RooibosT/G1_Dex1_BCT_subtask_joint_30hz
        DEFAULT_MAX_STEPS=30000
        ;;
    joint_30hz_relarm)
        # RELATIVE arms + ABSOLUTE waist/grippers on the 30 fps dataset — the
        # field-validated brainco split (CK-Sung: REL arms deployed well on real G1).
        # Same dataset/schedule as joint_30hz for a clean action-rep A/B.
        CONFIG=$EXAMPLE_DIR/g1_dex1_bct_joint_relarm_config.py
        MODALITY_JSON=$EXAMPLE_DIR/modality_joint.json
        DATASET_ROOT=$REPO_ROOT/datasets/RooibosT/G1_Dex1_BCT_subtask_joint_30hz
        DEFAULT_MAX_STEPS=50000
        ;;
    *)  echo "variant must be joint|joint_h16|joint_wbc|joint_30hz|joint_30hz_relarm|joint_30hz_relarm_4view|ee" >&2; exit 1 ;;
esac
DEFAULT_MAX_STEPS="${DEFAULT_MAX_STEPS:-25000}"

BASE_MODEL_PATH=$REPO_ROOT/models/GR00T-N1.7-3B
TRAIN_DATASET="${DATASET_ROOT}_train"
VAL_DATASET="${DATASET_ROOT}_val"
EXP_NAME="g1_dex1_bct_${VARIANT}_b256${EXP_SUFFIX:-}"
OUTPUT_DIR="$REPO_ROOT/outputs/${EXP_NAME}"
WANDB_PROJECT_NAME=gr00t-n1.7-g1-dex1-bct

# Effective batch 256 = global 32 (16/GPU) x accum 8 — the validated 2xB200 shape.
# ffmpeg_threads=1 prevents decode-thread oversubscription (RETRAIN_NOTES.md §6-9);
# the BCT AV1 videos use GOP=2, so single-thread random access stays cheap.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_STEPS="${MAX_STEPS:-$DEFAULT_MAX_STEPS}"     # 15fps: 25k ~= 31 epochs; 30hz: 50k ~= 27 epochs
SAVE_STEPS="${SAVE_STEPS:-2500}"
NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
STATE_DROPOUT="${STATE_DROPOUT:-0.2}"
EP_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.2}"
SHARD="${SHARD_SIZE:-1024}"
FFMPEG_THREADS="${DATALOADER_FFMPEG_THREADS:-1}"

# The GR00T loader needs LeRobot v2.1 layout (episodes.jsonl / per-episode files).
# The subtask datasets ship as v3.0 — convert once with scripts/lerobot_conversion.
if [ ! -f "$DATASET_ROOT/meta/episodes.jsonl" ]; then
    echo "ERROR: $DATASET_ROOT is not in LeRobot v2.1 layout (meta/episodes.jsonl missing)." >&2
    echo "Convert it first (uses the dedicated conversion venv):" >&2
    echo "  cd scripts/lerobot_conversion && .venv/bin/python convert_v3_to_v2.py \\" >&2
    echo "    --repo-id RooibosT/$(basename "$DATASET_ROOT") --root $REPO_ROOT/datasets" >&2
    exit 1
fi

# GR00T-side modality map (state/action slices, camera naming, task annotation).
# Copied into the dataset so the split + loader can find it at meta/modality.json.
cp "$MODALITY_JSON" "$DATASET_ROOT/meta/modality.json"

# Train/val split: every 20th episode held out (~67 val eps, ~13 per subtask;
# subtasks are interleaved in episode order so the split stays proportional).
if [ ! -d "$TRAIN_DATASET" ]; then
    echo "Train split not found — creating ${TRAIN_DATASET} (+_val) ..."
    python examples/unitree_g1_dex1/split_train_val.py --src "$DATASET_ROOT" --val-every 20
else
    # keep splits' modality.json in sync with the selected variant
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
echo "=== g1_dex1 BCT finetune [$VARIANT] ==="
echo "  config          : $CONFIG"
echo "  dataset         : $TRAIN_DATASET"
echo "  effective batch : $((GLOBAL_BATCH_SIZE * GRAD_ACCUM)) (global $GLOBAL_BATCH_SIZE x accum $GRAD_ACCUM)"
echo "  steps           : $MAX_STEPS (save every $SAVE_STEPS)"
echo "  output          : $OUTPUT_DIR"

# No numactl node binding: 16 workers x 2 ranks of video decode need both NUMA nodes.
env \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    NUM_GPUS="${NUM_GPUS:-2}" USE_WANDB=1 GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
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
    -- --save-total-limit "${SAVE_TOTAL_LIMIT:-20}" --gradient-accumulation-steps "$GRAD_ACCUM" \
    --val-dataset-path "$VAL_DATASET" --eval-steps "${EVAL_STEPS:-2500}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    2>&1 | tee "$OUTPUT_DIR/train.log"
