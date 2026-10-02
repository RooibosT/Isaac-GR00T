#!/usr/bin/env bash
# stage1_2_new_desc: the B arm of the stage1_2_new instruction A/B, 46-dim ABS.
#
# A is `chain_s12new.sh` (`_s12new_abs`): the seven exported strings, where three pairs
# share their body and differ only in the `stageX.Y` prefix -- after formalize_language,
# two digit tokens. On the frozen layer-12 features, pooled over the text tokens, those
# pairs sit 0.00002 / 0.00006 / 0.00001 apart against a median 0.0004 between different
# subtasks, so A is the arm where the repeats are expected to train as one instruction.
#
# B rewrites all seven from what the cameras show about the stage (bare tabletop and four
# loose legs; turned tabletop carrying one leg and three loose), with the wording varied
# inside each pair. The pairs move to 0.00046 / 0.00073 / 0.00056 -- as far apart as
# different subtasks. `make_stage1_2_new_desc.py` derives it from A's split, so episodes,
# frames, val and statistics are identical and only the text differs.
#
# Everything else is A's launch line: 90,000 steps, save/eval 2,500, stride 7, 2 GPUs at
# micro 32 and 12 workers, seed 42. The GPU count is held on purpose. The shard schedule is
# seed-determined, but shards are dealt to `rank * workers + worker`, so the same count
# replays A's sample stream and the A/B difference is the instruction alone. At 4 GPUs the
# stream reshuffles, and retraining variance has never been measured on this data (§38),
# so a small A/B gap could not be told apart from it.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/stage1_2_new_desc
CFG=$EX/g1_dex1_ikea_absarm_3view_aug_config.py
LOG=$ROOT/datasets/chain_s12new_desc.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ] || [ ! -f "$D/meta/relative_stats.json" ]; then
        say "$D has no stats -- run make_stage1_2_new_desc.py, which copies A's"
        exit 1
    fi
done
if ! cmp -s "$HUB/stage1_2_new_train/meta/stats.json" "${DS}_train/meta/stats.json"; then
    say "stats differ from stage1_2_new_train -- the A/B would not share normalisation"
    exit 1
fi
say "stats identical to A"

say "launching 46-dim ABS on stage1_2_new_desc (7 rewritten instructions, 90k steps = 13.4 epochs)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
# scan_when_done.sh sets CUDA_VISIBLE_DEVICES per scan GPU, so SCAN_GPUS takes physical
# ids and has to follow GPUS rather than stay at "0 1".
GPUS="${GPUS:-0,1}"
CUDA_VISIBLE_DEVICES="$GPUS" \
NUM_GPUS=2 \
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_s12new_desc_abs \
MAX_STEPS=90000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="${GPUS//,/ }" \
USE_WANDB=1 MASTER_PORT=29690 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
