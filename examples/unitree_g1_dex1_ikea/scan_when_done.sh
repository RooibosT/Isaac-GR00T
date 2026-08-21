#!/usr/bin/env bash
# Wait for one IKEA training run to exit, then scan its checkpoints on two GPUs.
#
# Waits on the process rather than on checkpoint-20000 appearing, so a run that
# dies early still gets whatever checkpoints it wrote scanned instead of hanging
# here forever.
#
# The wait pattern needs the trailing space after the output dir: the baseline
# run's name is a prefix of the ablations' ("..._b64" vs "..._b64_torsograv"),
# so a bare match would wait on all three.
#
#   bash scan_when_done.sh <experiment_name> <config.py> <gpuA> <gpuB>
set -uo pipefail

EXP="$1"; CONFIG="$2"; GPU_A="$3"; GPU_B="$4"

# GPU 7 is reserved for the user's own work on this host — never schedule onto it.
for g in "$GPU_A" "$GPU_B"; do
    if [ "$g" = "7" ]; then
        echo "refusing to use GPU 7 (reserved); pick another" >&2
        exit 1
    fi
done
ROOT="/home/chan/IKEA/Isaac-GR00T"
# HF Trainer nests the run under a second copy of the experiment name.
OUT="$ROOT/outputs/$EXP/$EXP"
# The waist-aligned runs emit 19 dims and must be scored on the matching
# *_wa_val split; scoring them against the 16-dim baseline split silently
# compares different action vectors. Override VAL for those.
VAL="${VAL:-$ROOT/datasets/carroll511/G1_Dex1_IKEA_table_30hz_val}"
LOG="$ROOT/datasets/scan_${EXP}.log"

cd "$ROOT"
export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[$(date '+%F %T')] waiting for $EXP ..." | tee -a "$LOG"
while pgrep -f "output_dir $ROOT/outputs/$EXP " > /dev/null 2>&1; do sleep 60; done
sleep 45   # let the final checkpoint and wandb sync flush

STEPS=$(find "$OUT" -maxdepth 1 -name 'checkpoint-*' -type d -printf '%f\n' \
        | sed 's/checkpoint-//' | sort -n)
N=$(echo "$STEPS" | wc -l)
HALF=$(( (N + 1) / 2 ))
A=$(echo "$STEPS" | head -n "$HALF" | paste -sd,)
B=$(echo "$STEPS" | tail -n +$((HALF + 1)) | paste -sd,)
echo "[$(date '+%F %T')] $EXP done; $N ckpts -> GPU $GPU_A [$A] | GPU $GPU_B [$B]" | tee -a "$LOG"

source "$ROOT/.venv/bin/activate"
for pair in "$GPU_A:$A:a" "$GPU_B:$B:b"; do
    g=${pair%%:*}; rest=${pair#*:}; steps=${rest%:*}; tagname=${rest##*:}
    [ -z "$steps" ] && continue
    CUDA_VISIBLE_DEVICES="$g" python "$ROOT/examples/unitree_g1_dex1_ikea/scan_ikea.py" \
        --checkpoints-dir "$OUT" --dataset-path "$VAL" --config "$CONFIG" \
        --stride 10 --steps "$steps" \
        --output "$OUT/scan_$tagname.json" >> "$LOG" 2>&1 &
done
wait

python - "$OUT" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); merged = {}
for f in sorted(out.glob("scan_*.json")):
    merged.update(json.loads(f.read_text()))
(out / "scan.json").write_text(json.dumps(merged, indent=1))
print(f"merged {len(merged)} checkpoints -> {out/'scan.json'}")
PY
echo "[$(date '+%F %T')] scan complete for $EXP" | tee -a "$LOG"
