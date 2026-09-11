#!/usr/bin/env bash
# After the 60-dim REL run scans: upload its best checkpoint, then train the
# 46-dim REL variant without arm velocity.
#
# Separate from chain_stage1v2_rel.sh because that one was already running when
# this was asked for; editing a live script's file moves the shell's read
# position under it.
#
# The 46-dim run needs no stats work -- the REL chain already regenerated
# relative_stats.json, and both configs take the relative targets from the same
# two arm blocks.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=/root/02_hub/datasets/IKEA_pick_leg_stage1_v2
A=g1_dex1_ikea_relarm_3view_aug_b64_stage1v2_relarm_armvel
LOG=$ROOT/datasets/chain_stage1v2_rel_followup.log
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


say "waiting for chain_stage1v2_rel.sh (training + scan) to exit"
while pgrep -f "chain_stage1v2_rel[.]sh" > /dev/null 2>&1; do sleep 60; done
say "it exited"

source "$ROOT/.venv/bin/activate"
SCAN=$ROOT/outputs/$A/$A/scan.json
if [ -f "$SCAN" ]; then
    say "uploading best checkpoint of $A"
    python "$HUB/upload_best_ckpt.py" "${V2_DATA[@]}" \
        --exp "$A" \
        --repo RooibosT/gr00t-n1.7-g1-dex1-ikea-stage1v2-relarm-armvel-30hz-h40 \
        --config g1_dex1_ikea_armvel_config.py \
        --state-dim 60 --action-rep RELATIVE \
        --state-desc "the 46-dim set (legs 12, waist 3, both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6) plus left_arm_vel and right_arm_vel, 7+7. **Inference must feed real arm_dq**; zeros cost more than never having had the block (EXPERIMENTS.md 16). The deploy-side 60-dim ObservationSpec is IKEA_ARMVEL_SPEC." \
        2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -ne 0 ] && say "UPLOAD FAILED -- continuing to the 46-dim run anyway"
else
    say "NO scan.json at $SCAN -- skipping upload"
fi

say "launching 46-dim REL (no arm velocity) on stage1_v2"
cd "$ROOT"
CONFIG=$EX/g1_dex1_ikea_relarm_3view_aug_config.py \
DATASET_ROOT=$DS \
EXP_SUFFIX=_stage1v2_relarm \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29676 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "rel followup complete ($?)"
