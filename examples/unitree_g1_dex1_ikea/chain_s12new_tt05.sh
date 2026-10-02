#!/usr/bin/env bash
# stage1_2_new_tt05: A's data plus target_thread's stage 0.5, 46-dim ABS.
#
# target_thread_subtask opens 56 episodes with `stage0.5: orient table leg to find grasp
# pose` on a stage-1 scene (bare tabletop, four loose legs), so `make_stage1_2_new_tt05.py`
# appends those segments to A's train set under the `stage1.1` string. Val is A's val and
# the statistics are A's, copied, so against `_s12new_abs` the only change is the added
# frames: 27,736 frames / 25,552 windows, +5.9%.
#
# It is the same phase but not the same distribution -- the pose separates it from
# stage1.1 at AUC 0.963 (arm joints), and the room and standing posture differ -- so
# whether it helps or is kept apart as its own mode is what the run measures.
#
# 95,000 steps. The windows grow to 456,241, and 95,000 x 64 / 456,241 = 13.3 epochs
# against A's 13.4, so A's own frames get the same number of passes they got in A.
# Everything else is A's launch line, on the other two GPUs of this host.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/stage1_2_new_tt05
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_s12new_tt05.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ] || [ ! -f "$D/meta/relative_stats.json" ]; then
        say "$D has no stats -- run make_stage1_2_new_tt05.py, which copies A's"
        exit 1
    fi
done
if ! cmp -s "$HUB/stage1_2_new_train/meta/stats.json" "${DS}_train/meta/stats.json"; then
    say "stats differ from stage1_2_new_train -- A's samples would be normalised differently"
    exit 1
fi
say "stats identical to A"

say "launching 46-dim ABS on stage1_2_new + target_thread stage0.5 (95k steps = 13.3 epochs)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
# scan_when_done.sh sets CUDA_VISIBLE_DEVICES per scan GPU, so SCAN_GPUS takes physical ids.
GPUS="${GPUS:-2,3}"
CUDA_VISIBLE_DEVICES="$GPUS" \
NUM_GPUS=2 \
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_s12new_tt05_abs \
MAX_STEPS=95000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="${GPUS//,/ }" \
USE_WANDB=1 MASTER_PORT=29691 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
