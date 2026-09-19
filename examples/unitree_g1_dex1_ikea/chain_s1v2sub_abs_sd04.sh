#!/usr/bin/env bash
# stage1_v2_subtask, 46-dim ABSOLUTE, state_dropout 0.4 -- one knob off `_s1v2sub_abs`.
#
# The baseline is the run uploaded as `...-ikea-stage1-absarm-subtask-30hz-h40`
# (checkpoint-37500). Everything else is copied from its saved conf.yaml and launch
# line: same config, same train/val split, 45,000 steps / save+eval 2,500, effective
# batch 64 (2 GPU x micro 32), lr 1e-4, warmup 0.05, frozen LLM and ViT, same
# color jitter, DDP with bf16 comm, and a stride-7 scan on finish.
#
# The code has moved since that run (loss_weight, state_dropout_keep_keys,
# tune_top_llm_layers), but each of those defaults to the old behaviour, and the
# train split's stats.json predates the baseline's launch, so normalisation is the
# same too.
#
# What 0.4 actually means: the drop is drawn twice, once in the processor on the
# state values and once in the action head on the state embedding (commit
# e786bc2). The baseline's 0.2 hid the state on 1-0.8^2 = 36% of samples; 0.4
# hides it on 1-0.6^2 = 64%. It matches how the stage1 `_sd04` pair was run.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
DS=/root/02_hub/datasets/IKEA_pick_leg_stage1_v2_subtask
LOG=$ROOT/datasets/chain_s1v2sub_abs_sd04.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

SUFFIX=_s1v2sub_abs_sd04
EXP="g1_dex1_ikea_relarm_3view_aug_b64${SUFFIX}"
OUT="$ROOT/outputs/$EXP/$EXP"

if [ -f "$OUT/scan.json" ]; then
    say "=== $SUFFIX already scanned, nothing to do ==="
    exit 0
fi
if pgrep -f "output_dir $ROOT/outputs/$EXP " > /dev/null 2>&1; then
    say "=== $SUFFIX already running; not launching a second copy ==="
    exit 1
fi

cd "$ROOT"
# finetune.sh ends in `exec torchrun`, which is only on PATH inside the venv.
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

say "=== launching 46-dim ABSOLUTE, state_dropout 0.4  (suffix $SUFFIX) ==="
CONFIG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py \
DATASET_ROOT=$DS \
EXP_SUFFIX=$SUFFIX \
STATE_DROPOUT=0.4 \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29683 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "=== $SUFFIX finished (exit $?) ==="
