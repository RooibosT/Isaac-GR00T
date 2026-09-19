#!/usr/bin/env bash
# stage1_v2_subtask demonstrations + the human-intervention (DAgger) clips, 46-dim ABS.
#
# The pair to `chain_s1v2sub_abs.sh`'s first run (`_s1v2sub_abs`): same config,
# same schedule (45,000 / save 2,500 / scan stride 7), same untouched val. The
# only change is 61 extra episodes -- the teleop takeovers recorded just before
# the deployed policy would have failed.
#
# This is the subtask sibling of `chain_dagger.sh`, and it is the variant that
# matches the deployment the intervention data came from: that session ran the
# `dex1:ikea_stage1_subtask` profile, whose two instructions are exactly this
# export's. The expert clips carry only `assemble`, which `merge_stage1v2_dagger.py`
# maps onto the base's index for it -- the `align` tail is always under policy
# control, so no expert frame has it.
#
# The subtask boundary lives PER FRAME in `task_index`, so the merge copies that
# column through rather than flattening it; the merge asserts both instructions
# survive (219 episodes touch `assemble`, 139 touch `align`).
#
# Stats are generated with the RELATIVE config, as `chain_s1v2sub_abs.sh` did, so
# that adding a REL column later needs no second pass.
#
# The scan cannot settle this. Val is 9 demonstration episodes and holds no
# intervention states at all, so it measures whether the extra data BROKE
# anything, not whether it fixed the failure. The verdict is on the robot.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_pick_leg_stage1_v2_subtask_dagger
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
REL_CFG=$EX/g1_dex1_ikea_armvel_config.py
LOG=$ROOT/datasets/chain_subtask_dagger.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ] || [ ! -f "$D/meta/relative_stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$REL_CFG" \
            >> "$ROOT/datasets/stats_subtask_dagger.log" 2>&1 \
            || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

say "launching 46-dim ABS on stage1_v2_subtask + intervention expert"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_s1v2sub_abs_dagger \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29688 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
