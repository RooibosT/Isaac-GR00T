#!/usr/bin/env bash
# stage1_v2, 60-dim RELATIVE joint targets with arm velocity.
#
# The missing cell. Section 31 measured three ABSOLUTE variants on this dataset
# and could say how much armvel buys *on ABS* (-7.7%), but not whether ABS has
# closed the gap to RELATIVE, because no v2 REL run existed. This is the direct
# counterpart of `_stage1v2_absarm_armvel` -- same data, same val, same 45,000
# steps, same stride-7 scan -- so the two differ in the action representation
# alone.
#
# On v1 this config was the overall winner (arm8 1.153 / EE8 9.74 at 35,000)
# while 46-dim ABS was the worst corner at 2.261 / 15.48. Whether that ordering
# survives +41% data is the question. Note that section 27's late correction
# still stands: the real-robot observation ran opposite to the open-loop
# ranking, so this settles a number, not a deployment choice.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
DS=/root/02_hub/datasets/IKEA_pick_leg_stage1_v2
CFG=$EX/g1_dex1_ikea_armvel_config.py
LOG=$ROOT/datasets/chain_stage1v2_rel.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# REL targets need relative_stats.json, and the split's copy is empty -- it was
# generated with an ABS config, which has no relative keys to compute. Nothing
# downstream would say so, the normaliser would just find no entry.
if ! grep -q "left_arm" "${DS}_train/meta/relative_stats.json"; then
    say "regenerating stats for the REL config (relative_stats.json is empty)"
    for D in "${DS}_train" "${DS}_val"; do
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_stage1v2_rel.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    done
    say "stats ready"
fi

say "launching 60-dim REL+armvel on stage1_v2"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_stage1v2_relarm_armvel \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29675 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "rel chain complete ($?)"
