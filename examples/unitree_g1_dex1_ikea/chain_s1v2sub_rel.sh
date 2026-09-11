#!/usr/bin/env bash
# stage1_v2_subtask: the RELATIVE column, with and without arm velocity.
#
# `RooibosT/IKEA-pick-leg-stage1_v2_subtask` is the stage1_v2 recording re-exported
# under **two** instructions -- "stage1: assemble the first leg on the table base"
# and "stage1: align the table base" -- instead of one. The frames are the same
# material: 167 of stage1_v2's 172 episodes, in order, five dropped. So §31's
# schedule is reused unchanged (45,000 / save 2,500 / stride 7) and the val set is
# carried over rather than redrawn: 9 episodes / 10,604 frames against §31's 10 /
# 11,508, because v2 118 is one of the five this export drops. Train is 158 eps /
# 217,430 frames. Read the table beside §27/28/31 as the same held-out recordings
# minus one, not as the same number.
#
# What this measures that §31 could not. §31 ran the ABS column only and closed
# with "v2 REL+armvel 은 안 돌렸으므로 ABS 가 REL 을 따라잡았다고는 말할 수 없다" --
# it re-measured the armvel gain *on top of ABS* (-4.9% -> -8.3% once data grew
# 41%) and left the REL half open. These two runs are that half. §28 put the
# armvel gain on REL at -16.3% on v1 data; the question here is whether it holds,
# grows or shrinks at +41% data, and the pair is directly comparable to §31's
# 46-vs-60 ABS pair on the same schedule and near-identical val.
#
# The subtask is carried **per frame** by `task_index`, not per episode: 139 of the
# 158 train episodes hold both labels (8 of 9 in val), task 1 being the last ~10%
# of the episode. `modality.json` maps `annotation.human.task_description` to
# `task_index` via `original_key` and `lerobot_episode_loader` resolves it frame by
# frame, so the `language` modality already in both configs picks the boundary up
# with no config change -- verified on the loader: episode 1 of val reads task 0
# from frame 0 and task 1 from frame 1575 of 1675.
#
# The *scan* did need a change. `scan_ikea.py` grouped its per-task table by
# `episodes.jsonl["tasks"][0]`, one label per episode, which on this export files
# every window under "assemble the first leg" and hides the align phase inside its
# numbers. It now groups by the per-frame instruction; on a single-instruction
# split that is the same string, and a checkpoint-45000 control re-scanned on
# `IKEA_pick_leg_stage1_v2_val` returns the one group it always did.
#
# Sequential, not parallel, and both GPUs per run. Measured on this host at
# effective batch 64: 2 GPU micro 32 x accum 1 is 0.486 s/step against 1 GPU's
# 1.375, so two concurrent single-GPU runs would finish later than two sequential
# two-GPU ones. The scan wants both GPUs too. §31's two 45,000-step runs took
# 6:10:25 (46-dim) and 6:10:38 (60-dim) -- the 14 extra state dims are free -- and
# their 18-checkpoint stride-7 scans 40 and 43 min.
#
# Restartable. A run whose `scan.json` already exists is skipped, and a run already
# in flight is waited on rather than launched twice.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_pick_leg_stage1_v2_subtask
LOG=$ROOT/datasets/chain_s1v2sub_rel.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

REL_CFG=$EX/g1_dex1_ikea_armvel_config.py

if [ ! -d "${DS}_train" ]; then
    say "splitting $DS"
    "$ROOT/.venv/bin/python" "$HUB/split_stage1_v2_subtask.py" 2>&1 | tee -a "$LOG" || exit 1
fi

# Generated here, once, before any run starts: concurrent runs writing the same
# stats.json corrupt it (DATASETS.md). Both runs are RELATIVE, so both need
# `relative_stats.json` as well as `stats.json`; the stats pass is done with the
# 60-dim config because `generate_stats` covers every key in the dataset
# regardless of config and `generate_rel_stats` needs the RELATIVE action keys,
# which both configs share. Normally already satisfied here -- the ABS chain's
# pass wrote both files -- so this is a guard, not work.
for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ] || [ ! -f "$D/meta/relative_stats.json" ]; then
        say "stats for $D"
        "$ROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$REL_CFG" \
            >> "$ROOT/datasets/stats_s1v2sub.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

cd "$ROOT"
# finetune.sh ends in `exec torchrun`, which is only on PATH inside the venv;
# nohup-ing this script from a bare shell does not inherit an activated one.
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# suffix | config | port | description
RUNS=(
  "_s1v2sub_rel|g1_dex1_ikea_relarm_3view_aug_config.py|29683|46-dim RELATIVE, no arm velocity"
  "_s1v2sub_rel_armvel|g1_dex1_ikea_armvel_config.py|29684|60-dim RELATIVE + arm velocity"
)

for spec in "${RUNS[@]}"; do
    IFS='|' read -r SUFFIX CFGNAME PORT DESC <<< "$spec"
    EXP="g1_dex1_ikea_relarm_3view_aug_b64${SUFFIX}"
    OUT="$ROOT/outputs/$EXP/$EXP"

    if [ -f "$OUT/scan.json" ]; then
        say "=== $SUFFIX already scanned, skipping ==="
        continue
    fi
    # The trailing space matters: `_s1v2sub_rel` is a prefix of
    # `_s1v2sub_rel_armvel`, so a bare match would wait on the wrong run
    # (scan_when_done.sh, same reason).
    if pgrep -f "output_dir $ROOT/outputs/$EXP " > /dev/null 2>&1; then
        say "=== $SUFFIX already running; waiting for it (train + its own scan) ==="
        while pgrep -f "output_dir $ROOT/outputs/$EXP " > /dev/null 2>&1; do sleep 60; done
        # run_finetune_ikea.sh scans in-process after training, so give that a
        # moment to appear before deciding whether it happened.
        sleep 90
        if [ -f "$OUT/scan.json" ]; then
            say "=== $SUFFIX finished and scanned ==="
            continue
        fi
        say "=== $SUFFIX finished without a scan; scanning now ==="
        VAL="${DS}_val" STRIDE=7 \
            bash "$EX/scan_when_done.sh" "$EXP" "$EX/$CFGNAME" 0 1 >> "$LOG" 2>&1
        continue
    fi

    say "=== launching $DESC  (suffix $SUFFIX) ==="
    CONFIG=$EX/$CFGNAME \
    DATASET_ROOT=$DS \
    EXP_SUFFIX=$SUFFIX \
    MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
    SCAN_STRIDE=7 SCAN_GPUS="0 1" \
    USE_WANDB=1 MASTER_PORT=$PORT \
    bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
    say "=== $SUFFIX finished (exit $?) ==="
done
say "both RELATIVE runs complete"
