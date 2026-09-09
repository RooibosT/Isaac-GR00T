#!/usr/bin/env bash
# stage1_v2, 74-dim ABS state read over a one-second history.
#
# The 74-dim torque run reproduced the section 23 shortcut signature -- EE8 -12%
# but the gripper worse and the gain decaying with horizon -- which is consistent
# with the model having torque's magnitude and never its trend: every observation
# it has ever had is a single instant. This gives it four.
#
# History carries only arm torque, both EEF and both grippers (28 of the 74 dims).
# Measured on this dataset, that subset is 58% of what a dense history offers in
# the insertion phase, and the 42% given up is arm joint and velocity history --
# the channel a policy uses to extrapolate its own motion rather than read the
# scene. `history_dropout_prob` collapses the history onto the current frame 30%
# of the time; that form is not synthetic, it is what an episode's opening steps
# look like once indices clamp and what a robot sees before its buffer fills.
#
# Judged on the gripper, the late chunk and probe_insert_phase.py -- not arm8.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
DS=/root/02_hub/datasets/IKEA_pick_leg_stage1_v2
CFG=$EX/g1_dex1_ikea_absarm_hist_config.py
H=g1_dex1_ikea_relarm_3view_aug_b64_stage1v2_absarm_hist
T=g1_dex1_ikea_relarm_3view_aug_b64_stage1v2_absarm_torque
LOG=$ROOT/datasets/chain_stage1v2_hist.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "launching 74-dim ABS + 1s history on stage1_v2"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_stage1v2_absarm_hist \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29674 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 \
    --state-history-keys left_arm_torque right_arm_torque left_gripper right_gripper left_eef right_eef \
    --history-dropout-prob 0.3 >> "$LOG" 2>&1
say "training+scan exited ($?)"

# Phase probe against the torque run, gripper-selected on both -- arm8 is where
# a leak would sit, so selecting on it would pick whichever checkpoint reads the
# log best rather than whichever handles the contact best.
pick() { python - "$ROOT/outputs/$1/$1/scan.json" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(min(d, key=lambda k: d[k]["__all__"][sys.argv[2]]))
PY
}
for pair in "$H:$CFG:0" "$T:$EX/g1_dex1_ikea_absarm_armvel_torque_config.py:1"; do
    exp=${pair%%:*}; rest=${pair#*:}; cfg=${rest%:*}; gpu=${rest##*:}
    [ -f "$ROOT/outputs/$exp/$exp/scan.json" ] || { say "no scan.json for $exp, skipping probe"; continue; }
    ck=$(pick "$exp" mae_grip)
    say "probe: $exp / $ck on GPU $gpu"
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
    python "$EX/probe_insert_phase.py" \
        --checkpoint "$ROOT/outputs/$exp/$exp/$ck" \
        --dataset-path "${DS}_val" --config "$cfg" \
        --output "$ROOT/outputs/$exp/$exp/insert_phase.json" \
        >> "$ROOT/datasets/probe_stage1v2_${exp##*b64_}.log" 2>&1 &
done
wait
say "history chain complete"
