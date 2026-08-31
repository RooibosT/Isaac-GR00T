#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on our IKEA data using **Team-RAMEN's training recipe**.
#
# `run_finetune_ikea.sh`의 파생본이다. 데이터·state·action·lr·wd·warmup·state_dropout은
# 전부 그대로 두고, RAMEN_COMPARISON.md §2에서 "다름"으로 분류된 항목 중 데이터 변환이
# 필요 없는 세 가지만 바꾼다:
#
#   1. global batch 16, gradient accumulation 없음  (우리: eff 64 = global 16 x accum 4)
#   2. 200,000 steps                                 (우리: 20~40k)
#   3. RandomAffine 회전 ±5°                         (우리: color jitter + random crop만)
#   + embodiment tag를 REAL_G1으로 바꿔 projector slot 25(사전학습됨)에서 시작
#
# 실행창(16 vs 8)은 배포 설정이라 학습에 안 들어간다. EEF auxiliary action은 action 배열에
# eef 컬럼을 새로 만들어야 해서(FK(action.arm_q) -> xyz+rot6d 18칸) 2단계로 미뤘다.
#
# **왜 200k인가.** RAMEN이 세 태스크 모두에 쓴 값인데, 우리 데이터 크기에서 마침 정상
# epoch이 된다. leg_30hz_train은 129,974개의 H40 윈도우이므로
#     200,000 step x batch 16 = 3.2M sample = 24.6 epoch
# 이고 우리 검증 최적(armvel 26k/eff64 = 25.5 epoch)과 사실상 같다. RAMEN 자신의 insert
# 세트에서는 이 값이 4.7~6.0 epoch밖에 안 된다(그쪽 데이터가 4.4배 크다).
#
# **체크포인트 격자가 두 번째 비교를 공짜로 준다.** 20k마다 저장하면 control과 샘플 수를
# 맞춘 지점이 격자 위에 온다:
#     ctl leg_armvel 40k x 64 = 2.56M  ->  이 런의 160k
#     armvel v2 최적 26k x 64 = 1.66M  ->  이 런의 104k
# 그래서 "같은 샘플 수에서 배치가 작은 게 나은가"와 "레시피대로 끝까지 갔을 때"를 한 런에서
# 둘 다 읽을 수 있다. §21에서 쓴 것과 같은 방식.
#
# **속도는 실측할 것.** §3 표에서 역산하면 1 GPU micro-step(8)이 ~0.31s, eff 64/2GPU/accum4가
# 1.46 s/it이므로 통신이 ~0.23s -> batch 16/accum 1은 ~0.54 s/it, 200k에 30시간으로 추정된다.
# 다만 accum 1은 all-reduce가 매 스텝이고 다른 micro-step의 backward와 겹칠 수 없어서 노출
# 통신이 더 클 수 있다. 30스텝 재서 §3 표에 한 줄 추가할 것.
#
# 사용:
#   bash examples/unitree_g1_dex1_ikea/launch_ramen.sh
# 또는 직접:
#   CONFIG=... DATASET_ROOT=... EXP_SUFFIX=_leg \
#   bash examples/unitree_g1_dex1_ikea/run_finetune_ramen.sh --use-ddp --ddp-comm-bf16
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLE_DIR="$REPO_ROOT/examples/unitree_g1_dex1_ikea"
EXTRA_ARGS=("$@")

# torchcodec needs the ffmpeg 7 shared libs; the venv ships none.
if [ -d "$HOME/micromamba/envs/ffmpeg7/lib" ]; then
    export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ---- RAMEN 레시피에서 바뀌는 것 ------------------------------------------------
# run_finetune_ikea.sh는 embodiment tag를 두 군데에 하드코딩한다. 여기서는 파라미터다.
EMBODIMENT_TAG="${EMBODIMENT_TAG:-REAL_G1}"
# RandomAffine 회전. FinetuneConfig에 이미 있던 노브인데 기본값이 None이라 한 번도 켠 적이
# 없다. 평행이동(RAMEN은 5%)에 해당하는 노브는 없고 crop_fraction의 random crop이 부분적으로
# 대신한다. 0이나 빈 값이면 끈다.
ROTATION_ANGLE="${ROTATION_ANGLE:-5}"
# global 16 x accum 1 = effective 16. RAMEN은 8/GPU x 2 GPU, accum 없음.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-200000}"
SAVE_STEPS="${SAVE_STEPS:-20000}"
# 200k에서 2,000마다 eval하면 100번이다. RAMEN은 10k마다 512샘플로 본다.
EVAL_STEPS="${EVAL_STEPS:-10000}"
# --------------------------------------------------------------------------------

CONFIG="${CONFIG:-$EXAMPLE_DIR/g1_dex1_ikea_ramen_config.py}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/carroll511/G1_Dex1_IKEA_leg_30hz}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$REPO_ROOT/models/GR00T-N1.7-3B}"
TRAIN_DATASET="${DATASET_ROOT}_train"
VAL_DATASET="${DATASET_ROOT}_val"

NUM_GPUS="${NUM_GPUS:-2}"
NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
STATE_DROPOUT="${STATE_DROPOUT:-0.2}"
COLOR_JITTER_PARAMS="${COLOR_JITTER_PARAMS:-brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08}"
EP_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.2}"
SHARD="${SHARD_SIZE:-1024}"
FFMPEG_THREADS="${DATALOADER_FFMPEG_THREADS:-1}"

EXP_NAME="g1_dex1_ikea_ramen_b$((GLOBAL_BATCH_SIZE * GRAD_ACCUM))${EXP_SUFFIX:-}"
OUTPUT_DIR="$REPO_ROOT/outputs/${EXP_NAME}"
WANDB_PROJECT_NAME="${WANDB_PROJECT:-gr00t-n1.7-g1-dex1-ikea}"

for DS in "$TRAIN_DATASET" "$VAL_DATASET"; do
    [ -f "$DS/meta/episodes.jsonl" ] || {
        echo "ERROR: $DS missing (LeRobot v2.1 layout)." >&2
        exit 1; }
    # meta/stats.json은 embodiment 태그와 무관하다 (feature 이름으로만 fingerprint를 잡는다,
    # gr00t/data/stats.py). NEW_EMBODIMENT로 만들어 둔 파일을 그대로 쓴다.
    if [ ! -f "$DS/meta/stats.json" ]; then
        echo "Generating statistics for ${DS} ..."
        python -m gr00t.data.stats --dataset-path "$DS" \
            --embodiment-tag "$EMBODIMENT_TAG" --modality-config-path "$CONFIG"
    fi
done

mkdir -p "$OUTPUT_DIR"

CKPT_GB="${CKPT_GB:-12}"
NEED_GB=$(( (MAX_STEPS / SAVE_STEPS + 1) * CKPT_GB ))
FREE_GB=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
    echo "ERROR: need ~${NEED_GB} GB for $((MAX_STEPS / SAVE_STEPS + 1)) checkpoints, only ${FREE_GB} GB free." >&2
    echo "       Free space or raise SAVE_STEPS, then retry. Set SKIP_DISK_CHECK=1 to override." >&2
    [ "${SKIP_DISK_CHECK:-0}" = "1" ] || exit 1
fi
echo "disk: ${FREE_GB} GB free, this run needs ~${NEED_GB} GB"

# examples/finetune.sh는 `--` **앞의** 미지 인자에서 죽는다(case의 `*)` 분기).
# --random-rotation-angle은 launch_finetune.py가 받는 FinetuneConfig 필드이므로 `--` 뒤로
# 넘겨야 한다. 거기 온 것은 EXTRA_ARGS로 모여 LAUNCH_CMD에 그대로 붙는다.
ROT_ARGS=()
if [ -n "$ROTATION_ANGLE" ] && [ "$ROTATION_ANGLE" != "0" ]; then
    ROT_ARGS=(--random-rotation-angle "$ROTATION_ANGLE")
fi

echo "=== g1_dex1 IKEA finetune — RAMEN recipe ==="
echo "  config          : $CONFIG"
echo "  embodiment tag  : $EMBODIMENT_TAG  (projector slot 25 if REAL_G1, 10 if NEW_EMBODIMENT)"
echo "  dataset         : $TRAIN_DATASET"
echo "  gpus            : $NUM_GPUS (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all})"
echo "  effective batch : $((GLOBAL_BATCH_SIZE * GRAD_ACCUM)) (global $GLOBAL_BATCH_SIZE x accum $GRAD_ACCUM)"
echo "  steps           : $MAX_STEPS (save every $SAVE_STEPS, eval every $EVAL_STEPS)"
echo "  rotation aug    : ${ROTATION_ANGLE:-off} deg"
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
    --embodiment-tag "$EMBODIMENT_TAG" \
    --modality-config-path "$CONFIG" \
    --state-dropout-prob "$STATE_DROPOUT" \
    --wandb-project "$WANDB_PROJECT_NAME" \
    --experiment-name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --save-only-model \
    -- --save-total-limit "${SAVE_TOTAL_LIMIT:-10}" --gradient-accumulation-steps "$GRAD_ACCUM" \
    --val-dataset-path "$VAL_DATASET" --eval-steps "$EVAL_STEPS" \
    ${ROT_ARGS[@]+"${ROT_ARGS[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    2>&1 | tee "$OUTPUT_DIR/train.log"
