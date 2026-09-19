#!/usr/bin/env bash
# Upload the scan-selected checkpoint of the single-instruction thread_snapshot
# run, once its chain has finished training and scanning.
#
# The wait is on the chain's own completion line rather than on pgrep: a pattern
# matching a script name also matches any other shell whose command line quotes
# that name, which stalled the fliptable follow-up for six minutes.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_thread_snapshot_uni_armvel
CHAIN_LOG=$ROOT/datasets/chain_thread_snapshot_uni.log
LOG=$ROOT/datasets/chain_threadsnap_upload.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the training + scan chain to log its completion"
until grep -q "chain complete" "$CHAIN_LOG" 2>/dev/null; do sleep 60; done
say "it finished"

source "$ROOT/.venv/bin/activate"
SCAN=$ROOT/outputs/$B/$B/scan.json
if [ ! -f "$SCAN" ]; then
    say "NO scan.json at $SCAN -- nothing to upload"
    exit 1
fi

say "uploading best checkpoint of $B"
python "$HUB/upload_best_ckpt.py" \
    --exp "$B" \
    --repo RooibosT/gr00t-n1.7-g1-dex1-ikea-threadsnap-unified-armvel-30hz-h40 \
    --config g1_dex1_ikea_absarm_armvel_config.py \
    --dataset-repo RooibosT/thread_snapshot \
    --train-eps 73 --train-frames 14240 --val-eps 12 --val-frames 2375 \
    --max-steps 10000 \
    --data-note "$(cat "$HUB/thread_snapshot_uni_data_note.txt")" \
    --state-dim 60 \
    --state-desc "the 46-dim set (legs 12, waist 3, both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6) plus left_arm_vel and right_arm_vel, 7+7. **Inference must feed real arm_dq** -- zeros make it worse than the 46-dim baseline (EXPERIMENTS.md 16); the deploy-side 60-dim ObservationSpec is IKEA_ARMVEL_SPEC." \
    >> "$LOG" 2>&1 && say "upload done" || say "UPLOAD FAILED (see $LOG)"
