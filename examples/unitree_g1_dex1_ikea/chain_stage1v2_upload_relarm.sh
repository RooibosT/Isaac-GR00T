#!/usr/bin/env bash
# Upload the best checkpoint of the 46-dim REL run once its scan lands.
#
# Waits on chain_stage1v2_rel_followup.sh, which launched that run through
# run_finetune_ikea.sh and so exits only after the auto-scan.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_stage1v2_relarm
LOG=$ROOT/datasets/chain_stage1v2_upload_relarm.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# upload_best_ckpt.py takes the dataset facts as arguments rather than baking
# them into its card, so every caller has to supply them or argparse rejects the
# run. All the stage1_v2 runs share one set; the note lives in a file because
# four scripts repeating a paragraph is four places for it to drift.
V2_DATA=(--dataset-repo RooibosT/IKEA-pick-leg-stage1_v2
         --train-eps 162 --train-frames 222090
         --val-eps 10 --val-frames 11508
         --max-steps 45000
         --data-note "$(cat /root/02_hub/datasets/stage1v2_data_note.txt)")


say "waiting for chain_stage1v2_rel_followup.sh (46-dim training + scan) to exit"
while pgrep -f "chain_stage1v2_rel_followup[.]sh" > /dev/null 2>&1; do sleep 60; done
say "it exited"

source "$ROOT/.venv/bin/activate"
SCAN=$ROOT/outputs/$B/$B/scan.json
[ -f "$SCAN" ] || { say "NO scan.json at $SCAN -- nothing to upload"; exit 1; }

say "uploading best checkpoint of $B"
python "$HUB/upload_best_ckpt.py" "${V2_DATA[@]}" \
    --exp "$B" \
    --repo RooibosT/gr00t-n1.7-g1-dex1-ikea-stage1v2-relarm-30hz-h40 \
    --config g1_dex1_ikea_relarm_3view_aug_config.py \
    --state-dim 46 --action-rep RELATIVE \
    --state-desc "legs 12, waist 3, both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6. No arm velocity, no joint torque -- the baseline the other stage1_v2 runs add to." \
    2>&1 | tee -a "$LOG"
say "upload chain complete (exit ${PIPESTATUS[0]})"
