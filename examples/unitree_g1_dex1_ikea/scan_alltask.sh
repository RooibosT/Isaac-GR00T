#!/usr/bin/env bash
# Scan the four task-label runs, plus the control the ablation does not train.
#
# The control is the existing g1_dex1_ikea_relarm_3view_aug_b64_v2: it was trained
# at the identical recipe (eff 64, 20k, cosine, seed 42, same 46-dim config) on
# the older three tasks alone, so re-scoring its checkpoint-20000 against the new
# val split gives "what the old tasks looked like before the new ones arrived" for
# a GPU-hour instead of a nine-hour rerun. Scoring uses the policy's own processor
# statistics, not the val set's, so pointing it at a different val directory is
# legitimate.
#
# It must be the 46-dim run, not `..._v2_armvel`: the four runs here carry no arm
# velocity, and a control with an input they lack would confound the comparison
# with the 15%-on-arm8 velocity effect.
#
# Every run is scored on the SAME val split -- the unified one -- whatever labels
# it was trained with. That matters: scan_ikea.py reads the task string out of
# the val set's meta and reports per task, so scoring the split run against the
# split val would bucket its rotate episodes under two names and stop them lining
# up with the other runs. The instruction the policy is fed comes from the val
# set too, which is the point for the control (it has never seen these strings)
# and is what "unified at inference" means for run S.
#
#   bash scan_alltask.sh [run ...]     default: all four plus the control
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
cd "$ROOT"
export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$ROOT/.venv/bin/activate"

CONFIG="${CONFIG:-$ROOT/examples/unitree_g1_dex1_ikea/g1_dex1_ikea_relarm_3view_aug_config.py}"
VAL="${VAL:-$ROOT/datasets/carroll511/G1_Dex1_IKEA_all_30hz_unified_val}"
STRIDE="${STRIDE:-10}"

# Cap the maths threads. torch and BLAS both size their pools from the core count,
# so eight concurrent scans opened 331 threads each and asked for ~1200% CPU
# apiece; the box ran at load 268 on 96 cores and the GPUs sat at 0-14% while the
# per-window FK loop (80 wrist solves x 1,513 windows per checkpoint) thrashed.
# Measured cost of leaving this unset: 78 min per checkpoint against 37 expected.
# The training launcher has always set these; the scan path never did.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# name -> output dir holding the checkpoints
declare -A DIRS=(
  [m16]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_m16/g1_dex1_ikea_relarm_3view_aug_b64_alltask_m16"
  [m30]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_m30/g1_dex1_ikea_relarm_3view_aug_b64_alltask_m30"
  [uv]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_uv/g1_dex1_ikea_relarm_3view_aug_b64_alltask_uv"
  [mv]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_mv/g1_dex1_ikea_relarm_3view_aug_b64_alltask_mv"
  [ctlv]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_v2_armvel/g1_dex1_ikea_relarm_3view_aug_b64_v2_armvel"
  [u]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_u/g1_dex1_ikea_relarm_3view_aug_b64_alltask_u"
  [s]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_s/g1_dex1_ikea_relarm_3view_aug_b64_alltask_s"
  [n]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_n/g1_dex1_ikea_relarm_3view_aug_b64_alltask_n"
  [x]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_x/g1_dex1_ikea_relarm_3view_aug_b64_alltask_x"
  [ctl]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_v2/g1_dex1_ikea_relarm_3view_aug_b64_v2"
)
# one GPU pair per run, matching how they were trained
declare -A GPUS=([u]="0 1" [s]="2 3" [n]="4 5" [x]="6 7" [ctl]="0 1" \
                 [m16]="0 1" [m30]="2 3" [uv]="4 5" [mv]="6 7" [ctlv]="0 1")

# Runs whose config carries arm velocity; the rest use the 46-dim one. Scoring a
# 60-dim checkpoint against the 46-dim config silently feeds it the wrong state.
declare -A CFGS=(
  [uv]="$ROOT/examples/unitree_g1_dex1_ikea/g1_dex1_ikea_armvel_config.py"
  [mv]="$ROOT/examples/unitree_g1_dex1_ikea/g1_dex1_ikea_armvel_config.py"
  [ctlv]="$ROOT/examples/unitree_g1_dex1_ikea/g1_dex1_ikea_armvel_config.py"
)


RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(u s n x ctl)

# Each run owns a GPU pair, so the four training runs scan at the same time
# rather than one after another. The merged val split carries 14,970 windows
# against the older split's 614, so a checkpoint costs about 2.4x what it used to
# and a sequential pass over five runs would take most of a day. The control goes
# last because it shares GPUs 0,1 with run u.
MERGE='
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); merged = {}
for f in sorted(out.glob("scan_alltask_?.json")):
    merged.update(json.loads(f.read_text()))
(out / "scan_alltask.json").write_text(json.dumps(merged, indent=1))
print(f"merged {len(merged)} checkpoints -> {out}/scan_alltask.json")
'

scan_one() {   # name
    local name="$1"
    local OUT="${DIRS[$name]:-}"
    if [ -z "$OUT" ] || [ ! -d "$OUT" ]; then
        echo "skip $name: no output directory at ${OUT:-<unset>}" >&2
        return
    fi
    local LOG="$ROOT/datasets/scan_alltask_$name.log"
    local GA GB STEPS N HALF A B CFG
    read -r GA GB <<< "${GPUS[$name]}"
    CFG="${CFGS[$name]:-$CONFIG}"

    STEPS=$(find "$OUT" -maxdepth 1 -name 'checkpoint-*' -type d -printf '%f\n' \
            | sed 's/checkpoint-//' | sort -n)
    N=$(echo "$STEPS" | grep -c .)
    if [ "$N" -eq 0 ]; then echo "skip $name: no checkpoints" >&2; return; fi
    HALF=$(( (N + 1) / 2 ))
    A=$(echo "$STEPS" | head -n "$HALF" | paste -sd,)
    B=$(echo "$STEPS" | tail -n +$((HALF + 1)) | paste -sd,)
    echo "[$(date '+%F %T')] $name: $N ckpts ($(basename "$CFG")) -> GPU $GA [$A] | GPU $GB [$B]" | tee -a "$LOG"

    local pair g rest steps tag
    for pair in "$GA:$A:a" "$GB:$B:b"; do
        g=${pair%%:*}; rest=${pair#*:}; steps=${rest%:*}; tag=${rest##*:}
        [ -z "$steps" ] && continue
        CUDA_VISIBLE_DEVICES="$g" python "$ROOT/examples/unitree_g1_dex1_ikea/scan_ikea.py" \
            --checkpoints-dir "$OUT" --dataset-path "$VAL" --config "$CFG" \
            --stride "$STRIDE" --steps "$steps" \
            --output "$OUT/scan_alltask_$tag.json" >> "$LOG" 2>&1 &
    done
    wait

    python -c "$MERGE" "$OUT" | tee -a "$LOG"
}

# controls share GPU pairs with the training runs, so they go last
for name in "${RUNS[@]}"; do
    case "$name" in ctl|ctlv) continue;; esac
    scan_one "$name" &
done
wait

for name in "${RUNS[@]}"; do
    case "$name" in ctl|ctlv) scan_one "$name";; esac
done

echo "[$(date '+%F %T')] scans complete"
