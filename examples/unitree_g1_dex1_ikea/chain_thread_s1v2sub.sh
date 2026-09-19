#!/usr/bin/env bash
# target_thread_subtask + stage1_v2_subtask from the left-hand base grasp on, 46-dim ABS.
#
# The deploy SOTA `...-ikea-stage1-absarm-subtask` is `_s1v2sub_abs` at 37,500, so this
# is that recipe unchanged -- same config, 45,000 / save 2,500 / scan stride 7, both GPUs.
# What changes is the data in front of the base grasp: stage1_v2_subtask's own approach
# is cut away (`make_s1v2sub_fromgrasp.py`) and target_thread's leg orientation and last
# threading turn take its place (`merge_thread_s1v2sub.py`).
#
# Three instructions, per frame via `task_index`:
#   0 `stage1: assemble the first leg on the table base`  77.5% of train frames
#   1 `stage1: align the table base`                        9.6%
#   2 `stage0.5: orient table leg to find grasp pose`      12.9%
# 0 and 1 keep the SOTA's indices. Train is 201 episodes / 196,030 frames (149 + 52),
# 9% fewer frames than `_s1v2sub_abs` trained on, so 45,000 steps is ~14.7 epochs.
#
# The val is NOT the SOTA's: 8 of its 9 stage1_v2 episodes (151 has no base grasp), cut
# at the grasp, plus 5 target_thread episodes. Scan numbers therefore do not sit beside
# `_s1v2sub_abs`'s; rescore the SOTA on `thread_s1v2sub_val` before comparing.
#
# Known gap at the seam (memory `target-thread-seam-mismatch`): at the base grasp the
# target_thread left hand is ~8 cm nearer and ~5 cm further left than stage1_v2's, and
# the room background differs. Posture and head camera are the same.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/thread_s1v2sub
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
REL_CFG=$EX/g1_dex1_ikea_armvel_config.py
LOG=$ROOT/datasets/chain_thread_s1v2sub.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ] || [ ! -f "$D/meta/relative_stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$REL_CFG" \
            >> "$ROOT/datasets/stats_thread_s1v2sub.log" 2>&1 \
            || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

say "launching 46-dim ABS on target_thread + stage1_v2_subtask from the base grasp"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_thread_s1v2sub_abs \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29689 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
