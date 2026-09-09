#!/usr/bin/env bash
# Upload the best checkpoint of the stage1_v2 ABS+armvel run once its scan lands.
#
# Separate from chain_stage1v2_followup.sh because that script was already
# running when the upload was asked for; it waits on the followup to exit rather
# than editing a live process's script.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_stage1v2_absarm_armvel
LOG=$ROOT/datasets/chain_stage1v2_upload_armvel.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for chain_stage1v2_followup.sh (armvel training + scan) to exit"
while pgrep -f "chain_stage1v2_followup[.]sh" > /dev/null 2>&1; do sleep 60; done
say "it exited"

source "$ROOT/.venv/bin/activate"
SCAN=$ROOT/outputs/$B/$B/scan.json
if [ ! -f "$SCAN" ]; then
    say "NO scan.json at $SCAN -- nothing to upload"
    exit 1
fi

say "uploading best checkpoint of $B"
python "$HUB/upload_best_ckpt.py" \
    --exp "$B" \
    --repo RooibosT/gr00t-n1.7-g1-dex1-ikea-stage1v2-absarm-armvel-30hz-h40 \
    --config g1_dex1_ikea_absarm_armvel_config.py \
    --state-dim 60 \
    --state-desc "the 46-dim set (legs 12, waist 3, both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6) plus left_arm_vel and right_arm_vel, 7+7. **Inference must feed real arm_dq** -- zeros make it worse than the 46-dim baseline (EXPERIMENTS.md 16); the deploy-side 60-dim ObservationSpec is IKEA_ARMVEL_SPEC." \
    2>&1 | tee -a "$LOG"
say "upload chain complete (exit ${PIPESTATUS[0]})"
