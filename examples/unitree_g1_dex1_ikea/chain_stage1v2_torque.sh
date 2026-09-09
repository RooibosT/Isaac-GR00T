#!/usr/bin/env bash
# stage1_v2, 74-dim: ABSOLUTE arms + arm velocity + arm joint torque.
#
# Motivated by a deployment failure, not a table: the policy inserts the leg and
# does not go on to rotate it tight. Sections 21/23 rejected torque on
# whole-episode mean MAE, which is not what this asks -- see the config's
# docstring for why that rejection does not cover the contact component, and for
# why the first-8 arm error is the wrong metric here.
#
# No data work: IKEA_pick_leg_stage1_v2 carries all 117 state dims and
# meta/stats.json covers them, so this is config-only, same as the armvel run.
# 45,000 / 2,500 / stride 7 keeps it directly beside the 46- and 60-dim runs.
#
# After the scan, both this run and the 60-dim baseline go through
# probe_insert_phase.py so there is a like-for-like number at the approach and
# the contact -- the part the aggregate scan cannot see. No upload here: whether
# this checkpoint is worth publishing is a judgement to make on those numbers.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
DS=/root/02_hub/datasets/IKEA_pick_leg_stage1_v2
CFG=$EX/g1_dex1_ikea_absarm_armvel_torque_config.py
T=g1_dex1_ikea_relarm_3view_aug_b64_stage1v2_absarm_torque
B=g1_dex1_ikea_relarm_3view_aug_b64_stage1v2_absarm_armvel
LOG=$ROOT/datasets/chain_stage1v2_torque.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "launching 74-dim ABS+armvel+torque on stage1_v2"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_stage1v2_absarm_torque \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29673 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "training+scan exited ($?)"

# Phase probe on the scan-selected checkpoint of each, one per GPU.
pick() {   # best checkpoint of $1 by the metric named in $2
    python - "$ROOT/outputs/$1/$1/scan.json" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(min(d, key=lambda k: d[k]["__all__"][sys.argv[2]]))
PY
}
# Selected on the gripper, not on arm8: arm8 is where the torque leak lives, so
# using it here would pick the checkpoint that reads the log best.
for pair in "$T:$CFG:0" "$B:$EX/g1_dex1_ikea_absarm_armvel_config.py:1"; do
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
say "torque chain complete"
