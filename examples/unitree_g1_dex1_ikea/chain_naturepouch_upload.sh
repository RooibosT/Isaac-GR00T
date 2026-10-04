#!/usr/bin/env bash
# Upload the scan-selected checkpoint of the nature_pouch run, once its chain has finished
# training and scanning.
#
# Waits on the chain's own completion line rather than pgrep (see chain_threadsnap_upload.sh).
# Without CHECKPOINT, upload_best_ckpt.py takes the mae_arm_first8 argmin; pass CHECKPOINT=<step>
# to override it when that minimum is a one-point dip and EE8 and grip are clearly better later.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_naturepouch_absarm
REPO=${REPO:-RooibosT/gr00t-n1.7-g1-dex1-naturepouch-absarm-30hz-h40}
CHAIN_LOG=$ROOT/datasets/chain_nature_pouch_absarm.log
LOG=$ROOT/datasets/upload_naturepouch.log
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
    --dataset-repo RooibosT/nature_pouch \
    --train-eps 90 --train-frames 23650 --val-eps 10 --val-frames 2612 \
    --max-steps 5000 --save-steps 250 --scan-stride 2 \
    --task-title "IKEA pick and place the pouch" \
    --data-note "$(cat "$HUB/naturepouch_data_note.txt")" \
    --state-dim 46 \
    --state-desc "legs 12, waist 3, both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6." \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    ${EXTRA_NOTE:+--extra-note "$EXTRA_NOTE"} \
    >> "$LOG" 2>&1 && say "upload done" || say "UPLOAD FAILED (see $LOG)"
