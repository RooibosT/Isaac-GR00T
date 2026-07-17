#!/usr/bin/env bash
# Entrypoint for the gr00t-inference image.
#
# Env vars:
#   MODEL_PATH      checkpoint — local dir inside the container OR an HF hub
#                   repo id (e.g. RooibosT/gr00t-n1.7-g1-dex1-A). Hub ids are
#                   snapshotted into /checkpoints/<name> on first start
#                   (Gr00tPolicy needs a local processor/ dir, so hub ids are
#                   never passed straight to the server). Default: variant A.
#   EMBODIMENT_TAG  default new_embodiment
#   PORT            default 5555
#   HF_TOKEN        required for private hub checkpoints
#
# Extra args are appended to run_gr00t_server.py.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-RooibosT/gr00t-n1.7-g1-dex1-A}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
PORT="${PORT:-5555}"

# GPU / attention sanity check (fail fast with a readable message).
python - <<'PYEOF'
import torch

assert torch.cuda.is_available(), (
    "CUDA is not available in the container. Run with --gpus all and check "
    "`nvidia-smi` on the host."
)
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print(f"GPU: {name} (sm{cc[0]}{cc[1]})")
if cc < (8, 0):
    raise SystemExit(
        f"flash-attn 2 requires an Ampere-or-newer GPU (sm80+); got sm{cc[0]}{cc[1]}. "
        "This image cannot serve GR00T N1.7 on this GPU."
    )
import flash_attn  # noqa: F401
x = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
from flash_attn import flash_attn_func
flash_attn_func(x, x, x)  # exercises the CUDA kernel, not just the import
print(f"flash-attn {flash_attn.__version__} kernel OK")
PYEOF

# Resolve hub repo ids to a local snapshot.
if [ ! -d "$MODEL_PATH" ]; then
    LOCAL_DIR="/checkpoints/$(basename "$MODEL_PATH")"
    if [ ! -f "$LOCAL_DIR/config.json" ]; then
        echo "Downloading $MODEL_PATH -> $LOCAL_DIR"
        python - "$MODEL_PATH" "$LOCAL_DIR" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PYEOF
    fi
    MODEL_PATH="$LOCAL_DIR"
fi

echo "Serving $MODEL_PATH (tag=$EMBODIMENT_TAG) on port $PORT"
exec python gr00t/eval/run_gr00t_server.py \
    --model_path "$MODEL_PATH" \
    --embodiment_tag "$EMBODIMENT_TAG" \
    --port "$PORT" \
    --device cuda \
    "$@"
