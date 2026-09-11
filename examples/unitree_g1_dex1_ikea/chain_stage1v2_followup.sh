#!/usr/bin/env bash
# After the stage1_v2 ABS run finishes and scans: upload its best checkpoint,
# then train the same thing with arm velocities in the state.
#
# The upload is attempted first but is not a gate -- RooibosT has hit its private
# storage limit mid-push before (a 403 on the LFS batch endpoint, hours in), and
# a failed push is no reason to leave two H100s idle. The upload result is logged
# either way and the armvel run starts regardless.
#
# The armvel run needs no data work: meta/stats.json carries all 117 state and 33
# action dims of the export, so widening the state from 46 to 60 is config-only.
# Both configs are ABSOLUTE, so the empty relative_stats.json stays correct too.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_pick_leg_stage1_v2
A=g1_dex1_ikea_relarm_3view_aug_b64_stage1v2_absarm
LOG=$ROOT/datasets/chain_stage1v2_followup.log
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


# The bracket keeps this pattern from matching the pgrep process's own argv.
say "waiting for chain_stage1v2_absarm.sh (training + scan) to exit"
while pgrep -f "chain_stage1v2_absarm[.]sh" > /dev/null 2>&1; do sleep 60; done
say "it exited"

source "$ROOT/.venv/bin/activate"

SCAN=$ROOT/outputs/$A/$A/scan.json
if [ -f "$SCAN" ]; then
    say "uploading best checkpoint of $A"
    python "$HUB/upload_best_ckpt.py" "${V2_DATA[@]}" \
        --exp "$A" \
        --repo RooibosT/gr00t-n1.7-g1-dex1-ikea-stage1v2-absarm-30hz-h40 \
        --config g1_dex1_ikea_absarm_3view_aug_config.py \
        --state-dim 46 \
        --state-desc "legs 12, waist 3, both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6. No arm velocity, no joint torque." \
        2>&1 | tee -a "$LOG"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        say "UPLOAD FAILED -- continuing to the armvel run anyway; retry by hand"
    fi
else
    say "NO scan.json at $SCAN -- skipping upload (did the scan run?)"
fi

say "launching 60-dim ABS+armvel on stage1_v2"
cd "$ROOT"
CONFIG=$EX/g1_dex1_ikea_absarm_armvel_config.py \
DATASET_ROOT=$DS \
EXP_SUFFIX=_stage1v2_absarm_armvel \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29672 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "followup chain complete (exit $?)"
