#!/usr/bin/env bash
# RAMEN 레시피 완전판 — EEF auxiliary 포함, leg 세트, GPU 2,3.
#
# GPU 0,1의 `launch_ramen.sh` 런과 **EEF 하나만 다르다.** 배치 16 / accum 없음 / 200k /
# 회전 5도 / REAL_G1 slot 25가 전부 동일하고, action에 손목 EEF 9D 두 블록이 더 붙는다
# (16 -> 34차원). 그래서 두 런의 차이가 EEF 중복 지도의 값을 그대로 가격하고, 동시에
# 이쪽이 RAMEN 레시피의 완전판이다.
#
#   ctl (leg_armvel, 완료)  eff 64 / 40k / 증강 기본 / slot 10 / action 16
#   R   (GPU 0,1)           b16  / 200k / 회전 5도  / slot 25 / action 16
#   R+EEF (여기, GPU 2,3)   b16  / 200k / 회전 5도  / slot 25 / action 34   <- RAMEN 전체
#
# 읽는 법: R vs ctl = 나머지 세 변경의 합, R+EEF vs R = EEF 단독, R+EEF vs ctl = 레시피 전체.
#
# 데이터: G1_Dex1_IKEA_leg_30hz_eef — `make_eef_action_variant.py` 산출.
# state 117 -> 135, action 33 -> 51 (뒤에 `{left,right}_eef_9d` 9칸씩 덧붙임).
# 프레임/에피소드/윈도우 수는 원본과 동일하므로(307 ep / 141,947 frame / 129,974 윈도우)
# 200k x 16 = 24.6 epoch도 그대로다.
#
# **EEF 통계는 미리 만들어 둘 것.** relative 통계 계산이 joint 키는 9 it/s인데 EEF 키는
# 에피소드당 4.8초다(포즈 합성이 프레임 x horizon마다 들어간다). train 307 에피소드면
# 두 EEF 키에 ~50분이 걸린다. 학습 시작 시점에 만들게 두면 GPU가 그만큼 논다:
#     python -m gr00t.data.stats \
#       --dataset-path datasets/carroll511/G1_Dex1_IKEA_leg_30hz_eef_train \
#       --embodiment-tag REAL_G1 \
#       --modality-config-path examples/unitree_g1_dex1_ikea/g1_dex1_ikea_eefaux_config.py
# 두 프로세스가 같은 파일에 동시에 쓰면 깨지므로(§21의 pnp 런처 주석) 이 스크립트는
# relative_stats.json이 없으면 아예 실행을 거부한다.
#
# 저장 격자 8,000 / limit 18은 R과 같아야 한다. 체크포인트를 서로 맞춰 비교하기 때문이다.
#   18 x 12.58 = 226 GB. (모델은 항상 132차원으로 패딩되므로 state/action 폭이 늘어도
#   체크포인트 크기는 R과 같다.)
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
cd "$ROOT"
export PATH="$HOME/micromamba/envs/ffmpeg7/bin:$PATH"
source "$ROOT/.venv/bin/activate"

STEPS="${MAX_STEPS:-200000}"
KEEP="${SAVE_TOTAL_LIMIT:-18}"
# GPU 0,1의 R 런 및 GPU 6,7의 pi05 런과 96코어를 나눠 쓴다
WORKERS="${DATALOADER_NUM_WORKERS:-10}"
GPUS="${GPUS:-2,3}"
PORT="${MASTER_PORT:-29551}"

DS="$ROOT/datasets/carroll511/G1_Dex1_IKEA_leg_30hz_eef"
CFG="$ROOT/examples/unitree_g1_dex1_ikea/g1_dex1_ikea_eefaux_config.py"

for split in train val; do
    for f in stats.json relative_stats.json; do
        if [ ! -f "${DS}_${split}/meta/$f" ]; then
            echo "refusing to launch: ${DS}_${split}/meta/$f 가 없다" >&2
            echo "  gr00t.data.stats 를 먼저 돌릴 것 (헤더 주석 참고)" >&2
            exit 1
        fi
    done
done

NEED=$(( (KEEP + 1) * 12 ))
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "disk: ${FREE} GB free now, this run needs ~${NEED} GB on top of the R run and pi05"
if [ "$FREE" -lt $(( NEED + 150 )) ]; then
    echo "ERROR: too tight next to the R run on GPU 0,1." >&2
    exit 1
fi

echo "[$(date '+%F %T')] launching ramen+eef on GPU $GPUS port $PORT <- $(basename "$DS")  [$(basename "$CFG")]"
CUDA_VISIBLE_DEVICES="$GPUS" \
MASTER_PORT="$PORT" \
CONFIG="$CFG" \
DATASET_ROOT="$DS" \
EMBODIMENT_TAG=REAL_G1 \
ROTATION_ANGLE=5 \
GLOBAL_BATCH_SIZE=16 \
GRAD_ACCUM=1 \
EXP_SUFFIX="_leg_reef" \
MAX_STEPS="$STEPS" \
SAVE_STEPS="${SAVE_STEPS:-8000}" \
EVAL_STEPS="${EVAL_STEPS:-10000}" \
SAVE_TOTAL_LIMIT="$KEEP" \
NUM_GPUS=2 \
DATALOADER_NUM_WORKERS="$WORKERS" \
nohup bash examples/unitree_g1_dex1_ikea/run_finetune_ramen.sh \
    --use-ddp --ddp-comm-bf16 \
    > "$ROOT/datasets/train_ramen_leg_reef.log" 2>&1 &

echo "[$(date '+%F %T')] launched; log in datasets/train_ramen_leg_reef.log"
