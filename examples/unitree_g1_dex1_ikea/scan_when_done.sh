#!/usr/bin/env bash
# Wait for one IKEA training run to exit, then scan its checkpoints, several processes per GPU.
#
# Waits on the process rather than on checkpoint-20000 appearing, so a run that
# dies early still gets whatever checkpoints it wrote scanned instead of hanging
# here forever.
#
# The wait pattern needs the trailing space after the output dir: the baseline
# run's name is a prefix of the ablations' ("..._b64" vs "..._b64_torsograv"),
# so a bare match would wait on all three.
#
#   bash scan_when_done.sh <experiment_name> <config.py> <gpu> [<gpu> ...]
#
# Env overrides: VAL (val split), STRIDE (window stride, default 10), TAG
# (embodiment tag, default new_embodiment -- the RAMEN configs register REAL_G1),
# SCAN_PER_GPU (scan processes per GPU, default 4), DRY_RUN=1 (print the shards
# and commands, scan nothing).
# STRIDE is not free choice when comparing against recorded numbers: the
# `leg_30hz` control was measured at stride 7 / 994 windows, so a scan meant to
# sit beside those figures must pass STRIDE=7.
#
# One scan process holds ~7 GB of an 80 GB H100 and keeps it ~16% busy (one window
# at a time, FK on the CPU), so a single process per GPU left most of the card idle.
# Checkpoints are dealt round-robin over GPU x SCAN_PER_GPU shards, each shard its own
# process and its own scan_<letter>.json; scan_ikea.py seeds every window the same way
# regardless of which process scores it, so the shard layout does not change a number.
set -uo pipefail

EXP="$1"; CONFIG="$2"; shift 2; GPUS=("$@")
PER_GPU="${SCAN_PER_GPU:-4}"

# GPUs the operator has claimed for their own work; scans refuse to schedule onto
# them. GPU 7 was reserved for a stretch and is not any more, so this is empty by
# default — set RESERVED_GPUS="7" (space separated) to bring the guard back.
for g in "${GPUS[@]}"; do
    for r in ${RESERVED_GPUS:-}; do
        if [ "$g" = "$r" ]; then
            echo "refusing to use GPU $g (reserved); pick another" >&2
            exit 1
        fi
    done
done
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# HF Trainer nests the run under a second copy of the experiment name.
OUT="$ROOT/outputs/$EXP/$EXP"
# The waist-aligned runs emit 19 dims and must be scored on the matching
# *_wa_val split; scoring them against the 16-dim baseline split silently
# compares different action vectors. Override VAL for those.
VAL="${VAL:-$ROOT/datasets/carroll511/G1_Dex1_IKEA_table_30hz_val}"
LOG="$ROOT/datasets/scan_${EXP}.log"
[ -n "${DRY_RUN:-}" ] && LOG=/dev/null

cd "$ROOT"
# torchcodec wants the ffmpeg 7 libs where they exist; a system ffmpeg 6 also
# decodes these clips, so this is added only if the env is actually installed.
if [ -d "$HOME/micromamba/envs/ffmpeg7/lib" ]; then
    export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# EXPERIMENTS.md section 18: without this the scan grabs 331 threads per process
# and takes 59 min per checkpoint instead of 5.9 -- the FK call is the hot path,
# 80 per window. The training launcher always set it; only the scan path missed it.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

if [ -z "${DRY_RUN:-}" ]; then
    echo "[$(date '+%F %T')] waiting for $EXP ..." | tee -a "$LOG"
    while pgrep -f "output_dir $ROOT/outputs/$EXP " > /dev/null 2>&1; do sleep 60; done
    sleep 45   # let the final checkpoint and wandb sync flush
fi

mapfile -t STEPS < <(find "$OUT" -maxdepth 1 -name 'checkpoint-*' -type d -printf '%f\n' \
                     | sed 's/checkpoint-//' | sort -n)
N=${#STEPS[@]}
# Shard s runs on GPUS[s % #GPUS], so consecutive shards alternate GPUs, and step i goes
# to shard i % S: every shard gets early and late checkpoints alike.
S=$(( ${#GPUS[@]} * PER_GPU ))
LETTERS=(a b c d e f g h i j k l m n o p q r s t u v w x y z)
[ "$S" -le "${#LETTERS[@]}" ] || { echo "too many shards ($S)" >&2; exit 1; }
declare -a SHARD
for i in "${!STEPS[@]}"; do
    s=$(( i % S ))
    SHARD[$s]="${SHARD[$s]:+${SHARD[$s]},}${STEPS[$i]}"
done
echo "[$(date '+%F %T')] $EXP done; $N ckpts over ${#GPUS[@]} GPU(s) x $PER_GPU:" | tee -a "$LOG"

[ -z "${DRY_RUN:-}" ] && source "$ROOT/.venv/bin/activate"
for (( s = 0; s < S; s++ )); do
    steps=${SHARD[$s]:-}
    [ -z "$steps" ] && continue
    g=${GPUS[$(( s % ${#GPUS[@]} ))]}
    echo "    scan_${LETTERS[$s]}: GPU $g [$steps]" | tee -a "$LOG"
    cmd=(python "$ROOT/examples/unitree_g1_dex1_ikea/scan_ikea.py"
         --checkpoints-dir "$OUT" --dataset-path "$VAL" --config "$CONFIG"
         --embodiment-tag "${TAG:-new_embodiment}"
         --stride "${STRIDE:-10}" --steps "$steps"
         --output "$OUT/scan_${LETTERS[$s]}.json")
    if [ -n "${DRY_RUN:-}" ]; then
        echo "      CUDA_VISIBLE_DEVICES=$g ${cmd[*]}"
    else
        CUDA_VISIBLE_DEVICES="$g" "${cmd[@]}" >> "$LOG" 2>&1 &
    fi
done
[ -n "${DRY_RUN:-}" ] && exit 0
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
