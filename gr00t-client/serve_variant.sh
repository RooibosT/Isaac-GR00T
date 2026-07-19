#!/usr/bin/env bash
# Serve one of the finetuned G1-Dex1 N1.7 variants with the stock Isaac-GR00T
# policy server. Run this ON THE INFERENCE GPU SERVER, inside the Isaac-GR00T
# repo with its venv activated (source .venv/bin/activate).
#
# The checkpoint is downloaded from its (private) HF Hub repo into
# $CHECKPOINT_ROOT on first use — make sure `hf auth login` / HF_TOKEN is set
# up for an account with access. Gr00tPolicy resolves its processor from a
# local "processor/" subdirectory, so we always serve from a local snapshot
# rather than passing the hub id straight through.
#
# Usage:
#   bash serve_variant.sh A     [port] [gpu]   # rel joint arms, horizon 16
#   bash serve_variant.sh B     [port] [gpu]   # abs joint arms, horizon 16
#   bash serve_variant.sh A40   [port] [gpu]   # rel joint arms, horizon 40
#   bash serve_variant.sh B40   [port] [gpu]   # abs joint arms, horizon 40
# Defaults: port 5555, gpu 0.
#
# Env overrides:
#   MODEL_PATH       existing local checkpoint dir (skips hub download)
#   HUB_USER         hub namespace for the variant repos (default RooibosT)
#   CHECKPOINT_ROOT  where hub snapshots are stored (default ~/gr00t_checkpoints)

set -euo pipefail

VARIANT="${1:?usage: serve_variant.sh <A|B|A40|B40> [port] [gpu]}"
PORT="${2:-5555}"
GPU="${3:-0}"
HUB_USER="${HUB_USER:-RooibosT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$HOME/gr00t_checkpoints}"

case "$VARIANT" in
    A|B|A40|B40) ;;
    *) echo "Unknown variant: $VARIANT (expected A, B, A40, B40)" >&2; exit 1 ;;
esac

if ! python -c "import gr00t" 2>/dev/null; then
    echo "The Isaac-GR00T venv is not active. From the Isaac-GR00T repo run:" >&2
    echo "  source .venv/bin/activate" >&2
    exit 1
fi

if [ -z "${MODEL_PATH:-}" ]; then
    REPO_ID="${HUB_USER}/gr00t-n1.7-g1-dex1-${VARIANT}"
    MODEL_PATH="$CHECKPOINT_ROOT/gr00t-n1.7-g1-dex1-${VARIANT}"
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "Downloading $REPO_ID -> $MODEL_PATH"
        python - "$REPO_ID" "$MODEL_PATH" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
EOF
    fi
fi

echo "Serving variant $VARIANT from $MODEL_PATH on port $PORT (GPU $GPU)"

exec env CUDA_VISIBLE_DEVICES="$GPU" python gr00t/eval/run_gr00t_server.py \
    --model_path "$MODEL_PATH" \
    --embodiment_tag new_embodiment \
    --port "$PORT" \
    --device cuda
