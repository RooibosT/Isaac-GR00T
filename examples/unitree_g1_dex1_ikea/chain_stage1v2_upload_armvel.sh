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

# upload_best_ckpt.py takes the dataset facts as arguments rather than baking
# them into its card, so every caller has to supply them or argparse rejects the
# run. All the stage1_v2 runs share one set; the note lives in a file because
# four scripts repeating a paragraph is four places for it to drift.
V2_DATA=(--dataset-repo RooibosT/IKEA-pick-leg-stage1_v2
         --train-eps 162 --train-frames 222090
         --val-eps 10 --val-frames 11508
         --max-steps 45000
         --data-note "$(cat /root/02_hub/datasets/stage1v2_data_note.txt)")


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
# The dataset facts moved from constants inside upload_best_ckpt.py to arguments
# here, so that pointing it at a different export cannot silently emit a card
# describing this one. These are stage1_v2's.
python "$HUB/upload_best_ckpt.py" "${V2_DATA[@]}" \
    --exp "$B" \
    --repo RooibosT/gr00t-n1.7-g1-dex1-ikea-stage1v2-absarm-armvel-30hz-h40 \
    --config g1_dex1_ikea_absarm_armvel_config.py \
    --dataset-repo RooibosT/IKEA-pick-leg-stage1_v2 \
    --train-eps 162 --train-frames 222090 --val-eps 10 --val-frames 11508 \
    --data-note "The IROS challenge stage-1 recording (assemble the first table leg) re-issued with 45 more episodes. The three subtasks (pick the leg, insert it, rotate to tighten) are carried under a single instruction, \`stage1: assemble the first leg on the table base\`, so episodes are the whole ~45 s sequence rather than per-subtask cuts. The v2 export prepends its new episodes, so its last 127 are bit-identical to \`RooibosT/IKEA-pick-leg-stage1\` and the validation split is **the same ten recordings** the v1 stage-1 models were scored on (v1's seed-20260904 draw, +45). Only train grew: 117 -> 162 episodes, 157,663 -> 222,090 frames (+41%)." \
    --state-dim 60 \
    --state-desc "the 46-dim set (legs 12, waist 3, both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6) plus left_arm_vel and right_arm_vel, 7+7. **Inference must feed real arm_dq** -- zeros make it worse than the 46-dim baseline (EXPERIMENTS.md 16); the deploy-side 60-dim ObservationSpec is IKEA_ARMVEL_SPEC." \
    2>&1 | tee -a "$LOG"
say "upload chain complete (exit ${PIPESTATUS[0]})"
