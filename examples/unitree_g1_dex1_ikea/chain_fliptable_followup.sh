#!/usr/bin/env bash
# After the fliptable run: upload its scan-selected checkpoint, then start
# thread_snapshot.
#
# The two are chained rather than run side by side because each training uses
# both GPUs, and the fliptable scan uses both as well.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_fliptable_armvel
LOG=$ROOT/datasets/chain_fliptable_followup.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for chain_fliptable.sh (training + scan) to exit"
while pgrep -f "chain_fliptable[.]sh" > /dev/null 2>&1; do sleep 60; done
say "it exited"

source "$ROOT/.venv/bin/activate"
SCAN=$ROOT/outputs/$B/$B/scan.json
if [ ! -f "$SCAN" ]; then
    say "NO scan.json at $SCAN -- skipping the upload, still starting thread_snapshot"
else
    say "uploading best checkpoint of $B"
    python "$HUB/upload_best_ckpt.py" \
        --exp "$B" \
        --repo RooibosT/gr00t-n1.7-g1-dex1-ikea-fliptable-armvel-30hz-h40 \
        --config g1_dex1_ikea_absarm_armvel_config.py \
        --dataset-repo URL-RFM/IKEA_fliptable \
        --train-eps 38 --train-frames 23693 --val-eps 5 --val-frames 2712 \
        --max-steps 16000 \
        --data-note "$(cat "$HUB/fliptable_data_note.txt")" \
        --state-dim 60 \
        --state-desc "the 46-dim set (legs 12, waist 3, both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6) plus left_arm_vel and right_arm_vel, 7+7. **Inference must feed real arm_dq** -- zeros make it worse than the 46-dim baseline (EXPERIMENTS.md 16); the deploy-side 60-dim ObservationSpec is IKEA_ARMVEL_SPEC." \
        >> "$LOG" 2>&1 && say "upload done" || say "UPLOAD FAILED (see $LOG)"
fi

say "starting thread_snapshot"
bash "$EX/chain_thread_snapshot.sh" >> "$LOG" 2>&1
say "followup chain complete (exit $?)"
