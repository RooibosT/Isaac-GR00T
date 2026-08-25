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

# name -> output dir holding the checkpoints
declare -A DIRS=(
  [u]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_u/g1_dex1_ikea_relarm_3view_aug_b64_alltask_u"
  [s]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_s/g1_dex1_ikea_relarm_3view_aug_b64_alltask_s"
  [n]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_n/g1_dex1_ikea_relarm_3view_aug_b64_alltask_n"
  [x]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_x/g1_dex1_ikea_relarm_3view_aug_b64_alltask_x"
  [ctl]="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_v2/g1_dex1_ikea_relarm_3view_aug_b64_v2"
)
# one GPU pair per run, matching how they were trained
declare -A GPUS=([u]="0 1" [s]="2 3" [n]="4 5" [x]="6 7" [ctl]="0 1")

RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(u s n x ctl)

for name in "${RUNS[@]}"; do
    OUT="${DIRS[$name]:-}"
    if [ -z "$OUT" ] || [ ! -d "$OUT" ]; then
        echo "skip $name: no output directory at ${OUT:-<unset>}" >&2
        continue
    fi
    LOG="$ROOT/datasets/scan_alltask_$name.log"
    read -r GA GB <<< "${GPUS[$name]}"

    STEPS=$(find "$OUT" -maxdepth 1 -name 'checkpoint-*' -type d -printf '%f\n' \
            | sed 's/checkpoint-//' | sort -n)
    N=$(echo "$STEPS" | grep -c .)
    [ "$N" -eq 0 ] && { echo "skip $name: no checkpoints" >&2; continue; }
    HALF=$(( (N + 1) / 2 ))
    A=$(echo "$STEPS" | head -n "$HALF" | paste -sd,)
    B=$(echo "$STEPS" | tail -n +$((HALF + 1)) | paste -sd,)
    echo "[$(date '+%F %T')] $name: $N ckpts -> GPU $GA [$A] | GPU $GB [$B]" | tee -a "$LOG"

    for pair in "$GA:$A:a" "$GB:$B:b"; do
        g=${pair%%:*}; rest=${pair#*:}; steps=${rest%:*}; tag=${rest##*:}
        [ -z "$steps" ] && continue
        CUDA_VISIBLE_DEVICES="$g" python "$ROOT/examples/unitree_g1_dex1_ikea/scan_ikea.py" \
            --checkpoints-dir "$OUT" --dataset-path "$VAL" --config "$CONFIG" \
            --stride "$STRIDE" --steps "$steps" \
            --output "$OUT/scan_alltask_$tag.json" >> "$LOG" 2>&1 &
    done
    wait

    python - "$OUT" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); merged = {}
for f in sorted(out.glob("scan_alltask_*.json")):
    merged.update(json.loads(f.read_text()))
(out / "scan_alltask.json").write_text(json.dumps(merged, indent=1))
print(f"merged {len(merged)} checkpoints -> {out/'scan_alltask.json'}")
PY
done
echo "[$(date '+%F %T')] scans complete"
