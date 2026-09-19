#!/usr/bin/env bash
# One model, two jobs: the stage-1 sequence and the table flip. 46-dim ABS.
#
# `thread_ins_flip` is `thread_ins_s1v2sub` (the target_thread_insert + stage1_v2
# splice, 3 instructions) merged with `IKEA_fliptable_v3` (1 instruction) by
# merge_ikea_tasksets.py. 350 episodes / 329,169 frames / 315,519 H40 windows,
# stage-1 material 62.3% of the windows and flip 37.7%.
#
#   0 `stage1: assemble the first leg on the table base`   48.7% of train frames
#   1 `stage1: align the table base`                        5.7%
#   2 `stage0.5: orient table leg to find grasp pose`       7.7%
#   3 `flip the table`                                     37.9%
#
# 0-2 keep `thread_ins_s1v2sub`'s indices and 3 is appended, so a client built for
# either parent keeps working and only gains the other's string. `flip the table` is
# v2/v3's own string, not v1's `flip table`; `formalize_language` only lowercases and
# strips punctuation, so send exactly this one.
#
# Schedule by epochs, as every run here does: 74,000 steps is 15.0 epochs of 315,519
# windows, the same place `_thread_ins_s1v2sub_abs` (14.7) and `_fliptablev3_absarm`
# (15.1) sat. save 4,000 puts 19 checkpoints on the grid (~247 GB).
#
# Scan stride 10, not 7: val is 24,914 windows here against the stage-1 val's 1,734,
# and stride 10 leaves ~2,500 -- the stride the older multi-task scans used.
#
# ⚠️ **Read the per-instruction rows, not the headline.** Val is 28 episodes, and flip
# is 51.5% of its frames, so the `__all__` number is half a flip score. The question
# this run exists to answer -- does one model hold both -- lives in the per-instruction
# table, against the two single-purpose models:
#   `_thread_ins_s1v2sub_abs`  ckpt 45000: assemble 2.111 / orient 2.243 / align 3.154 deg
#   `_fliptablev3_absarm`      ckpt 22500: flip 3.708 deg arm8, 28.18 mm EE8
# Neither was scored on THIS val, so those numbers are a sanity reference, not a
# baseline. Rescore both here before claiming a win or a loss.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/thread_ins_flip
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_thread_ins_flip.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    [ -d "$D" ] || { say "$D missing -- run merge_ikea_tasksets.py --sets thread_ins_s1v2sub IKEA_fliptable_v3 --out $DS"; exit 1; }
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_thread_ins_flip.log" 2>&1 \
            || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

EXP=g1_dex1_ikea_relarm_3view_aug_b64_thread_ins_flip_abs
if [ -f "$ROOT/outputs/$EXP/$EXP/scan.json" ]; then
    say "=== already trained and scanned, nothing to do ==="
    exit 0
fi

say "launching 46-dim ABS on stage-1 + flip, four instructions"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_thread_ins_flip_abs \
MAX_STEPS=74000 SAVE_STEPS=4000 EVAL_STEPS=4000 \
SCAN_STRIDE=10 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29693 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
