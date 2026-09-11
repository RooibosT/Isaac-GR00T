#!/usr/bin/env bash
# Upload the best checkpoint of the 47-dim ABS + phase run once its scan lands.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_stage1v2_absarm_phase
LOG=$ROOT/datasets/chain_stage1v2_upload_phase.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for chain_stage1v2_phase.sh (training + scan) to exit"
while pgrep -f "chain_stage1v2_phase[.]sh" > /dev/null 2>&1; do sleep 60; done
say "it exited"

source "$ROOT/.venv/bin/activate"
SCAN=$ROOT/outputs/$B/$B/scan.json
[ -f "$SCAN" ] || { say "NO scan.json at $SCAN -- nothing to upload"; exit 1; }

say "uploading best checkpoint of $B"
python "$HUB/upload_best_ckpt.py" \
    --exp "$B" \
    --repo RooibosT/gr00t-n1.7-g1-dex1-ikea-stage1v2-absarm-phase-30hz-h40 \
    --config g1_dex1_ikea_absarm_phase_config.py \
    --state-dim 47 --action-rep ABSOLUTE \
    --state-desc "legs 12, waist 3, both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6, and a **subtask phase bit** -- 0 while the leg is still being turned, 1 once an external module has counted four turns. The phase is exempt from state dropout, so the model was never trained to work without it." \
    --dataset-repo RooibosT/IKEA-pick-leg-stage1_v2_subtask \
    --train-eps 158 --train-frames 217430 --val-eps 9 --val-frames 10604 \
    --max-steps 45000 \
    --data-note "$(cat "$HUB/phase_data_note.txt")" \
    --extra-note "$(cat "$HUB/phase_extra_note.txt")" \
    2>&1 | tee -a "$LOG"
say "upload chain complete (exit ${PIPESTATUS[0]})"
