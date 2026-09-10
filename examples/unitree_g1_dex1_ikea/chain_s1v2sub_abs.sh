#!/usr/bin/env bash
# stage1_v2_subtask: the ABSOLUTE column, with and without arm velocity.
#
# `RooibosT/IKEA-pick-leg-stage1_v2_subtask` is the stage1_v2 recording re-exported
# under **two** instructions -- "stage1: assemble the first leg on the table base"
# and "stage1: align the table base" -- instead of one. The frames are the same
# material: 167 of stage1_v2's 172 episodes, in order, five dropped. So §31's
# schedule is reused unchanged (45,000 / save 2,500 / stride 7) and the val set is
# carried over rather than redrawn.
#
# The subtask is carried **per frame** by `task_index`, not per episode: 147 of the
# 167 episodes hold both labels, task 1 being the last ~10% of the episode, and 20
# never leave task 0. `modality.json` maps `annotation.human.task_description` to
# `task_index` via `original_key` and `lerobot_episode_loader` resolves it frame by
# frame, so the language modality already in both configs picks the boundary up with
# no config change. What it needed was a split script that does not flatten the
# column -- `split_stage1_v2.py` writes `task_index = 0`, which would have trained
# both of these on a single instruction. `split_stage1_v2_subtask.py` preserves it
# and asserts both labels survive into each split.
#
# Scope: the ABS column only. This started as the full representation x velocity 2x2
# and was cut to these two runs on request, so the RELATIVE half is not measured
# here and §31's open question -- whether ABS caught up with REL once the data grew
# -- stays open. What these two do measure is the armvel gain under ABS, now with
# the subtask boundary as language, against §31's 60-vs-46 ABS pair on the same
# schedule and near-identical val.
#
# Sequential, not parallel, and both GPUs per run. Measured on this host at effective
# batch 64: 2 GPU micro 32 x accum 1 is 0.486 s/step against 1 GPU's 1.375, so two
# concurrent single-GPU runs would finish later than two sequential two-GPU ones. The
# scan wants both GPUs too.
#
# Restartable. A run whose `scan.json` already exists is skipped, and a run already
# in flight is waited on rather than launched twice -- which is how this script picks
# up an ABS run started by an earlier driver instead of fighting it for the GPUs.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/IKEA_pick_leg_stage1_v2_subtask
LOG=$ROOT/datasets/chain_s1v2sub_abs.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

REL_CFG=$EX/g1_dex1_ikea_armvel_config.py

if [ ! -d "${DS}_train" ]; then
    say "splitting $DS"
    "$ROOT/.venv/bin/python" "$HUB/split_stage1_v2_subtask.py" 2>&1 | tee -a "$LOG" || exit 1
fi

# Generated here, once, before any run starts: concurrent runs writing the same
# stats.json corrupt it (DATASETS.md). One pass with a RELATIVE config covers more
# than these two ABS runs need -- `generate_stats` covers every key in the dataset
# regardless of config, and `generate_rel_stats` only has work to do for RELATIVE
# action keys, which neither config here has. It is done with the relative config
# anyway so that adding the REL column later needs no second stats pass.
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
  "_s1v2sub_abs|g1_dex1_ikea_absarm_3view_aug_config.py|29681|46-dim ABSOLUTE, no arm velocity"
  "_s1v2sub_abs_armvel|g1_dex1_ikea_absarm_armvel_config.py|29682|60-dim ABSOLUTE + arm velocity"
)

for spec in "${RUNS[@]}"; do
    IFS='|' read -r SUFFIX CFGNAME PORT DESC <<< "$spec"
    EXP="g1_dex1_ikea_relarm_3view_aug_b64${SUFFIX}"
    OUT="$ROOT/outputs/$EXP/$EXP"

    if [ -f "$OUT/scan.json" ]; then
        say "=== $SUFFIX already scanned, skipping ==="
        continue
    fi
    # The trailing space matters: `_s1v2sub_abs` is a prefix of `_s1v2sub_abs_armvel`,
    # so a bare match would wait on the wrong run (scan_when_done.sh, same reason).
    if pgrep -f "output_dir $ROOT/outputs/$EXP " > /dev/null 2>&1; then
        say "=== $SUFFIX already running; waiting for it (train + its own scan) ==="
        while pgrep -f "output_dir $ROOT/outputs/$EXP " > /dev/null 2>&1; do sleep 60; done
        # Training has exited, but `run_finetune_ikea.sh` scans in-process straight
        # afterwards and that scan does NOT carry `output_dir` on its command line --
        # it runs as `scan_ikea.py --checkpoints-dir $OUT`. Waiting only on the loop
        # above therefore returns while the scan is still going, and since a scan
        # takes ~40 min the `scan.json` test below then reads "no scan happened" and
        # launches a second one against the same checkpoints. That is what this
        # script did to the first ABS run: two scans, concurrent, on the same 18
        # checkpoints. They are deterministic so the merged scan.json was still
        # correct, but it was 40 minutes of duplicated GPU work.
        sleep 90
        while pgrep -f "scan_ikea.py --checkpoints-dir $OUT" > /dev/null 2>&1; do sleep 60; done
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
say "both ABS runs complete"
