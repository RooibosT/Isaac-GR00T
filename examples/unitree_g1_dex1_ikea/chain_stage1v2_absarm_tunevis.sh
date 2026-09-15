#!/usr/bin/env bash
# stage1_v2 + 46-dim ABSOLUTE joint actions, with the vision encoder unfrozen.
#
# The pair to `chain_stage1v2_absarm.sh`: same export, split, config and schedule
# (45,000 / save 2,500 / scan stride 7). The one change is `--tune-visual`, which
# trains the 407M ViT -- its 24 blocks, the merger and the three deepstack mergers --
# alongside the action head. The LLM stays frozen, but backward still runs through
# all 16 of its layers to reach the ViT.
#
# Why the ViT: read-outs of the frozen features locate the insertion hole to ~28.7 mm
# and the robot's own, plainly visible wrist to ~26.9 mm, and neither a 384 px input
# nor a fourth view moved that wall by more than 3-11%. Every head-side loss tried
# (goal pose at two weights) left hole aiming unchanged, and none of them can reach
# the encoder.
#
# Micro-batch 16 x accum 2, so effective batch 64 is unchanged. 32/GPU runs out of
# memory on the first step once the ViT is trainable (79.1 of 79.2 GiB); 16/GPU
# peaks at 78.2 GB, a 1 GB margin, hence expandable_segments against fragmentation.
# Measured 0.954 s/step over steps 100-600 against 0.48 frozen: ~12 h for 45,000.
#
# Outcome (EXPERIMENTS.md section 35): open loop 8-10% worse than the frozen run and
# hole aiming unchanged. Kept for the record, not as a recipe.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
DS=/root/02_hub/datasets/IKEA_pick_leg_stage1_v2
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_stage1v2_absarm_tunevis.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    [ -f "$D/meta/stats.json" ] || { say "missing $D/meta/stats.json -- run chain_stage1v2_absarm.sh first"; exit 1; }
done

say "launching 46-dim ABS + tune_visual on stage1_v2"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_stage1v2_absarm_tunevis \
GLOBAL_BATCH_SIZE=32 GRAD_ACCUM=2 \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29681 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 --tune-visual >> "$LOG" 2>&1
say "chain complete (exit $?)"
