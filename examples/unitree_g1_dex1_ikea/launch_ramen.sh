#!/usr/bin/env bash
# RAMEN 레시피 한 방 — leg 세트, GPU 0,1.
#
# 세 변경을 **한 런에 다 넣는다.** 이기면 그때 ablation으로 쪼갠다:
#
#   1. global batch 16, accum 없음        (ctl: eff 64 = global 16 x accum 4)
#   2. 200,000 steps                      (ctl: 40,000)
#   3. RandomAffine 회전 5도               (ctl: color jitter + random crop만)
#   + projector slot 25 (REAL_G1)         (ctl: slot 10 = 초기화 상태)
#
# 대조군은 새로 안 돌린다 — `g1_dex1_ikea_relarm_3view_aug_b64_leg_armvel`이 GPU 6,7에서
# 같은 데이터·같은 config(armvel 60차원)로 돌고 있다. 모달리티 정의는 한 글자도 안 바꿨으므로
# (`g1_dex1_ikea_ramen_config.py`는 등록 태그만 다르다) 두 런은 위 네 항목에서만 갈린다.
#
# 데이터: G1_Dex1_IKEA_leg_30hz — train 307 ep / 141,947 frame / 129,974 H40 윈도우,
# val 15 ep / 7,490 frame (pick·insert·rotate leg 각 5개). val은 예전 것과 바이트 동일이라
# §12 이후 모든 숫자와 연결된다.
#
# 200k x 16 = 3.2M sample = 24.6 epoch. 우리 검증 최적(armvel 26k/eff64 = 25.5 epoch)과
# 사실상 같다. `pnp` 세트를 안 쓴 이유: turn-tabletop / flip-table이 붙어 있는데 §18이 그
# 둘을 `insert`에 EE8 +12.06%의 간섭원으로 측정했다. RAMEN 레시피를 재는 자리에 이미 해롭다고
# 아는 교란을 넣을 이유가 없다.
#
# **저장 격자는 8,000이다. 자유 선택이 아니다.** ctl은 스캔이 끝난 뒤 체크포인트가 지워져
# 22k/24k/26k/38k/40k 다섯 개의 *측정치*만 남아 있다(datasets/scan_leg_s7{a,b}.log). 그 다섯
# 지점의 샘플 수를 이 런의 스텝으로 환산하면
#     ctl 22k -> 88k    24k -> 96k    26k -> 104k    38k -> 152k    40k -> 160k
# 이고 최대공약수가 8,000이다. 20k 격자로 저장하면 다섯 중 160k 하나만 맞는다. 8k로 저장하면
# 다섯 개가 전부 격자 위에 올라와 "같은 샘플 수에서 배치 16이 배치 64를 이기는가"를 점 하나가
# 아니라 곡선으로 읽을 수 있다.
#
# **저장은 25번 하되 18개만 남긴다.** 체크포인트는 실측 12.58 GB이고 25개면 314.5 GB인데,
# HF Trainer는 오래된 것부터 지우므로 limit 18이면 64k~200k가 남는다. 매칭 5개(88k~160k)가
# 전부 그 안에 들어오고, 덤으로 64k도 남는다 -- §6/§14의 "팔/EE는 16k에 포화"가 eff 64에서
# 1.024M 샘플이므로 이 런에서는 step 64,000이다. 즉 작은 배치가 *같은 샘플 수*에서 같은 지점에
# 포화하는지 볼 수 있다. 버리는 8k~56k는 1~7 epoch 구간이라 최적일 가능성이 없다.
#   18 x 12.58 = 226 GB (25개 대비 -88 GB).
#
# 판정은 `scan_ikea.py`로. eval_loss로 고르지 말 것(§4). `OMP_NUM_THREADS=4` 잊지 말 것(§18).
# ctl 기준선 (같은 스캔, val 15 ep):
#   ckpt    mse      arm      ee        grip
#   22000   0.06638  3.192    20.33mm   0.2313
#   24000   0.06621  3.181    20.18mm   0.2227
#   26000   0.06651  3.176    20.56mm   0.2168
#   38000   0.06767  3.167    19.87mm   0.2183
#   40000   0.06714  3.160    19.84mm   0.2168
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
cd "$ROOT"
export PATH="$HOME/micromamba/envs/ffmpeg7/bin:$PATH"
source "$ROOT/.venv/bin/activate"

STEPS="${MAX_STEPS:-200000}"
KEEP="${SAVE_TOTAL_LIMIT:-18}"
# GPU 6,7의 ctl과 96코어를 나눠 쓴다
WORKERS="${DATALOADER_NUM_WORKERS:-12}"
GPUS="${GPUS:-0,1}"
# examples/finetune.sh는 MASTER_PORT 기본이 29500이고 launch_leg.sh가 29543을 쓴다
PORT="${MASTER_PORT:-29550}"

DS="$ROOT/datasets/carroll511/G1_Dex1_IKEA_leg_30hz"
CFG="$ROOT/examples/unitree_g1_dex1_ikea/g1_dex1_ikea_ramen_config.py"

if [ ! -f "${DS}_train/meta/stats.json" ]; then
    echo "refusing to launch: ${DS}_train has no stats.json" >&2
    exit 1
fi

NEED=$(( (KEEP + 1) * 12 ))
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "disk: ${FREE} GB free now, this run needs ~${NEED} GB on top of the ctl still writing"
if [ "$FREE" -lt $(( NEED + 150 )) ]; then
    echo "ERROR: too tight next to the leg_armvel run." >&2
    exit 1
fi

echo "[$(date '+%F %T')] launching ramen recipe on GPU $GPUS port $PORT <- $(basename "$DS")  [$(basename "$CFG")]"
CUDA_VISIBLE_DEVICES="$GPUS" \
MASTER_PORT="$PORT" \
CONFIG="$CFG" \
DATASET_ROOT="$DS" \
EMBODIMENT_TAG=REAL_G1 \
ROTATION_ANGLE=5 \
GLOBAL_BATCH_SIZE=16 \
GRAD_ACCUM=1 \
EXP_SUFFIX="_leg_r" \
MAX_STEPS="$STEPS" \
SAVE_STEPS="${SAVE_STEPS:-8000}" \
EVAL_STEPS="${EVAL_STEPS:-10000}" \
SAVE_TOTAL_LIMIT="$KEEP" \
NUM_GPUS=2 \
DATALOADER_NUM_WORKERS="$WORKERS" \
nohup bash examples/unitree_g1_dex1_ikea/run_finetune_ramen.sh \
    --use-ddp --ddp-comm-bf16 \
    > "$ROOT/datasets/train_ramen_leg_r.log" 2>&1 &

echo "[$(date '+%F %T')] launched; log in datasets/train_ramen_leg_r.log"
