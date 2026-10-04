#!/usr/bin/env bash
# Upload the scan-selected checkpoint of the nature_fliptable run, once its chain has
# finished training, scanning and the v3.1 baseline scan.
#
# Waits on the chain's own completion line rather than pgrep (see chain_threadsnap_upload.sh).
# --task-title, --save-steps, --max-steps and --scan-stride are passed explicitly: the v3.1
# card left them at the script defaults and so reads "IKEA stage 1" and "every 2,500".
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_naturefliptable_absarm
# The user named it without the `ikea` segment the earlier uploads carry.
REPO=${REPO:-RooibosT/gr00t-n1.7-g1-dex1-naturefliptable-absarm-30hz-h40}
CHAIN_LOG=$ROOT/datasets/chain_nature_fliptable_absarm.log
LOG=$ROOT/datasets/upload_naturefliptable.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the training + scan chain to log its completion"
until grep -q "chain complete" "$CHAIN_LOG" 2>/dev/null; do sleep 60; done
say "it finished"

source "$ROOT/.venv/bin/activate"
SCAN=$ROOT/outputs/$B/$B/scan.json
[ -f "$SCAN" ] || { say "NO scan.json at $SCAN -- nothing to upload"; exit 1; }

say "uploading best checkpoint of $B -> $REPO"
python "$HUB/upload_best_ckpt.py" \
    --exp "$B" \
    --repo "$REPO" \
    --config g1_dex1_ikea_absarm_3view_aug_config.py \
    --dataset-repo RooibosT/nature_fliptable \
    --train-eps 105 --train-frames 69946 --val-eps 10 --val-frames 7353 \
    --max-steps 16000 --save-steps 1000 --scan-stride 4 \
    --task-title "IKEA flip the table, new room" \
    --data-note "$(cat "$HUB/naturefliptable_data_note.txt")" \
    --state-dim 46 \
    --state-desc "legs 12, waist 3, both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6." \
    --extra-note "$(cat "$HUB/naturefliptable_extra_note.txt")" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} ${DRY_RUN:+--dry-run} \
    >> "$LOG" 2>&1 && say "upload done" || say "UPLOAD FAILED (see $LOG)"
