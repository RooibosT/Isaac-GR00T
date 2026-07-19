#!/usr/bin/env bash
# Push a trained G1 Dex1 checkpoint to the Hugging Face Hub.
#
# A gr00t checkpoint dir is already a COMPLETE, loadable model:
#   config.json, model-*.safetensors (+ index), processor_config.json, statistics.json,
#   embodiment_id.json, experiment_cfg/ (conf/config yaml + dataset_statistics.json).
# This uploads the whole dir as-is.
#
# WHY a chosen checkpoint (not auto-push the last one): with 30fps data the val loss
# usually bottoms mid-run (e.g. ckpt-4000..7000), not at 13000. Select the best ckpt
# from the val-mse scan first, then push THAT one. See RETRAIN_NOTES.md §5-1.
#
# Usage:
#   bash examples/unitree_g1_dex1/push_to_hub.sh <checkpoint-dir> <repo-id> [--public] [--dry-run]
#
# Example (push the val-selected checkpoint of a config-a run):
#   bash examples/unitree_g1_dex1/push_to_hub.sh \
#       /NHNHOME/WORKSPACE/chan/outputs/g1_dex1_v2_a_b256/g1_dex1_v2_a_b256/checkpoint-6000 \
#       chan-VLA/gr00t-n1.7-g1-dex1-a-ckpt6000
#
# List a run's checkpoints:
#   ls -d /NHNHOME/WORKSPACE/chan/outputs/<exp>/<exp>/checkpoint-*
#
# Auth : already logged in as RooibosT (huggingface-cli login). Override with HF_TOKEN=hf_xxx.
# Repo : PRIVATE by default (publishing is outward-facing / hard to fully undo).
#        Add --public to publish openly. --dry-run validates + prints without uploading.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/NHNHOME/WORKSPACE/chan/.cache/huggingface}"

CKPT_DIR=""; REPO_ID=""; PRIVATE="True"; DRYRUN="0"
for arg in "$@"; do
    case "$arg" in
        --public)  PRIVATE="False" ;;
        --dry-run) DRYRUN="1" ;;
        -*)        echo "unknown flag: $arg" >&2; exit 2 ;;
        *) if   [ -z "$CKPT_DIR" ]; then CKPT_DIR="$arg";
           elif [ -z "$REPO_ID"  ]; then REPO_ID="$arg";
           else echo "unexpected extra arg: $arg" >&2; exit 2; fi ;;
    esac
done

if [ -z "$CKPT_DIR" ] || [ -z "$REPO_ID" ]; then
    echo "usage: push_to_hub.sh <checkpoint-dir> <repo-id> [--public] [--dry-run]" >&2
    exit 2
fi

# Absolutize (fails loudly if the dir is missing) and validate it is a real checkpoint,
# so we never push a half-written or wrong directory.
CKPT_DIR="$(cd "$CKPT_DIR" && pwd)"
for f in model.safetensors.index.json config.json processor_config.json statistics.json experiment_cfg; do
    if [ ! -e "$CKPT_DIR/$f" ]; then
        echo "ERROR: '$CKPT_DIR' is missing '$f' -- not a complete gr00t checkpoint." >&2
        exit 1
    fi
done

echo "=== push checkpoint to HF hub ==="
echo "  from    : $CKPT_DIR"
echo "  repo    : $REPO_ID"
echo "  private : $PRIVATE"
du -sh "$CKPT_DIR" 2>/dev/null | awk '{print "  size    : "$1}'
[ "$DRYRUN" = "1" ] && echo "  MODE    : DRY-RUN (no upload)"

python - "$CKPT_DIR" "$REPO_ID" "$PRIVATE" "$DRYRUN" <<'PY'
import sys
from huggingface_hub import HfApi

ckpt, repo_id, private, dryrun = sys.argv[1], sys.argv[2], sys.argv[3] == "True", sys.argv[4] == "1"
api = HfApi()

# whoami is only an informational check; a rate limit (429) or offline blip must not
# kill the push -- create_repo/upload_folder authenticate via the stored token/HF_TOKEN.
try:
    print(f"  auth    : {api.whoami().get('name')}")
except Exception as e:
    print(f"  auth    : (whoami unavailable: {type(e).__name__} -- e.g. HF rate limit; "
          "upload still uses the stored token/HF_TOKEN)")

if dryrun:
    print("\nDRY-RUN ok: checkpoint valid. Re-run without --dry-run to upload.")
    sys.exit(0)

api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
print("  uploading (~7GB, a few minutes)...")
api.upload_folder(
    folder_path=ckpt,
    repo_id=repo_id,
    repo_type="model",
    commit_message=f"Add G1 Dex1 GR00T-N1.7 checkpoint ({ckpt.split('/')[-1]})",
    # save-only-model runs carry no optimizer state; guard anyway.
    ignore_patterns=["*.lock", "optimizer.pt", "scheduler.pt", "rng_state*.pth"],
)
vis = "private" if private else "public"
print(f"\nDONE ({vis}) -> https://huggingface.co/{repo_id}")
PY
