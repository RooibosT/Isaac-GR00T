#!/usr/bin/env bash
# RAMEN 레시피 두 런이 끝나는 대로 각자의 GPU에서 스캔한다.
#
#   R      GPU 0,1   g1_dex1_ikea_ramen_b16_leg_r      leg_30hz_val      ramen config
#   R+EEF  GPU 2,3   g1_dex1_ikea_ramen_b16_leg_reef   leg_30hz_eef_val  eefaux config
#
# 두 런은 완주 시각이 1시간쯤 다르므로 각각 독립적으로 기다린다. 자기 학습이 쓰던 GPU가
# 그대로 비므로 스캔을 거기에 붙인다.
#
# ## stride 7 — 자유 선택이 아니다
#
# 대조군 `leg_armvel`의 수치가 stride 7 / 994 윈도우로 측정됐다
# (datasets/scan_leg_s7{a,b}.log). scan_ikea.py 기본값은 10이고, 그러면 윈도우 집합 자체가
# 달라져 대조군과 나란히 놓을 수 없다. 다른 IKEA 스캔들은 대부분 stride 10이므로 이 세트만
# 예외라는 점에 주의.
#
#   ctl 기준선 (stride 7, val 15 ep / 994 윈도우):
#     ckpt    mse      arm      ee        grip
#     22000   0.06638  3.192    20.33mm   0.2313
#     24000   0.06621  3.181    20.18mm   0.2227
#     26000   0.06651  3.176    20.56mm   0.2168
#     38000   0.06767  3.167    19.87mm   0.2183
#     40000   0.06714  3.160    19.84mm   0.2168
#
# 샘플 수를 맞춘 비교 지점(ctl step x 64 = 이 런 step x 16):
#     ctl 22k -> 88k    24k -> 96k    26k -> 104k    38k -> 152k    40k -> 160k
#
# ## ⚠️ 대기 패턴의 접미 공백
#
# `..._leg_r`은 `..._leg_reef`의 **접두사**다. 공백 없이 pgrep하면 50개(양쪽 전부)가 잡히고,
# 공백을 붙이면 27개(R만)가 잡힌다 — 실측으로 확인함. 공백을 빼면 R의 스캔이 R+EEF까지
# 끝나기를 기다리게 된다. scan_when_done.sh가 경고한 바로 그 함정이다.
#
# ## ⚠️ --embodiment-tag
#
# scan_ikea.py는 `MODALITY_CONFIGS["new_embodiment"]`를 하드코딩하고 있었다. RAMEN 레시피
# config들은 REAL_G1으로 등록하므로 그대로 두면 체크포인트를 하나도 못 읽고 KeyError로 죽는다
# (실제로 R의 첫 스캔이 5초 만에 0개를 병합하고 끝났다). scan_ikea.py에 인자를 추가했고
# 기본값은 new_embodiment라 기존 스캔들은 그대로 동작한다.
#
# ## OMP_NUM_THREADS
#
# §18: 이게 없으면 프로세스당 331 스레드를 잡아 체크포인트당 59분 -> 5.9분, 10배 차이가 난다.
# 스캔은 윈도우마다 FK를 80회 부르므로 이 경로가 핫패스다.
#
# 사용:
#   nohup bash examples/unitree_g1_dex1_ikea/scan_when_ramen_done.sh > datasets/scan_waiter_ramen.log 2>&1 &
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
cd "$ROOT"
export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$ROOT/.venv/bin/activate"

# §18. 학습 런처는 항상 설정했고 스캔 경로만 빠져 있었다.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

STRIDE="${STRIDE:-7}"
E="$ROOT/examples/unitree_g1_dex1_ikea"

scan_one() {   # exp  gpuA  gpuB  val  config  [tag]
    local EXP="$1" GA="$2" GB="$3" VAL="$4" CFG="$5" TAG="${6:-REAL_G1}"
    local OUT="$ROOT/outputs/$EXP/$EXP"          # HF Trainer가 이름을 한 번 더 중첩한다
    local LOG="$ROOT/datasets/scan_${EXP}.log"

    echo "[$(date '+%F %T')] $EXP: 학습 종료 대기 ..." | tee -a "$LOG"
    # 접미 공백 필수 — 위 주석 참고
    while pgrep -f "output_dir $ROOT/outputs/$EXP " > /dev/null 2>&1; do sleep 60; done
    sleep 60   # 마지막 체크포인트 flush + wandb sync

    if [ ! -d "$OUT" ]; then
        echo "[$(date '+%F %T')] $EXP: $OUT 이 없다 — 중단" | tee -a "$LOG"; return 1
    fi
    local STEPS N HALF A B
    STEPS=$(find "$OUT" -maxdepth 1 -name 'checkpoint-*' -type d -printf '%f\n' \
            | sed 's/checkpoint-//' | sort -n)
    N=$(echo "$STEPS" | grep -c .)
    if [ "$N" -eq 0 ]; then
        echo "[$(date '+%F %T')] $EXP: 체크포인트가 없다 — 중단" | tee -a "$LOG"; return 1
    fi
    HALF=$(( (N + 1) / 2 ))
    A=$(echo "$STEPS" | head -n "$HALF" | paste -sd,)
    B=$(echo "$STEPS" | tail -n +$((HALF + 1)) | paste -sd,)
    echo "[$(date '+%F %T')] $EXP 완료; ${N}개 -> GPU $GA [$A] | GPU $GB [$B]" | tee -a "$LOG"

    for pair in "$GA:$A:a" "$GB:$B:b"; do
        local g=${pair%%:*} rest=${pair#*:}
        local steps=${rest%:*} tagname=${rest##*:}
        [ -z "$steps" ] && continue
        CUDA_VISIBLE_DEVICES="$g" python "$E/scan_ikea.py" \
            --checkpoints-dir "$OUT" --dataset-path "$VAL" --config "$CFG" \
            --embodiment-tag "$TAG" --stride "$STRIDE" --steps "$steps" \
            --output "$OUT/scan_$tagname.json" >> "$LOG" 2>&1 &
    done
    wait

    python - "$OUT" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); merged = {}
for f in sorted(out.glob("scan_?.json")):
    merged.update(json.loads(f.read_text()))
(out / "scan.json").write_text(json.dumps(merged, indent=1))
print(f"merged {len(merged)} checkpoints -> {out/'scan.json'}")
PY
    echo "[$(date '+%F %T')] $EXP 스캔 완료" | tee -a "$LOG"
}

scan_one g1_dex1_ikea_ramen_b16_leg_r 0 1 \
    "$ROOT/datasets/carroll511/G1_Dex1_IKEA_leg_30hz_val" \
    "$E/g1_dex1_ikea_ramen_config.py" &
PID_R=$!

scan_one g1_dex1_ikea_ramen_b16_leg_reef 2 3 \
    "$ROOT/datasets/carroll511/G1_Dex1_IKEA_leg_30hz_eef_val" \
    "$E/g1_dex1_ikea_eefaux_config.py" &
PID_E=$!

wait $PID_R $PID_E
echo "[$(date '+%F %T')] 두 런 스캔 모두 완료"
echo "  R      : outputs/g1_dex1_ikea_ramen_b16_leg_r/g1_dex1_ikea_ramen_b16_leg_r/scan.json"
echo "  R+EEF  : outputs/g1_dex1_ikea_ramen_b16_leg_reef/g1_dex1_ikea_ramen_b16_leg_reef/scan.json"
