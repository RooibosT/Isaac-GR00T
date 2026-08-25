#!/usr/bin/env bash
# Four concurrent 2-GPU runs over the enlarged task set.
#
#   U  unified   everything, one string for all three rotate sets   GPU 0,1
#   S  split     everything, set 2 given its own string             GPU 2,3
#   N  renamed   as U, rotate-table under its natural name          GPU 4,5
#   X  new only  rotate-table and flip, without the older three     GPU 6,7
#
# What each pairing buys:
#
#   S vs U  the direction split. The two datasets differ in the label of 27
#           training episodes and nothing else -- same episode order, same
#           lengths, same frames, videos hardlinked off one copy.
#   N vs U  the wording, priced on its own. `rotate table base` sits 0.056 from
#           the existing `rotate leg to tighten` in this model's frozen text
#           stack, closer than `insert`/`rotate leg`, the pair already measured
#           to interfere (ego-view distance 1.03x, cross-task retrieval error
#           only 20-30% above own-task). U and S instead use `turn the tabletop
#           square`, at least 0.091 from every existing task.
#   X vs U  whether the older three tasks earn their place for the new ones, at
#           a fixed compute budget. Note X sees 24.7 epochs against U's 10.9,
#           because it holds 51,844 windows against 117,202 -- so this answers
#           "given 20k steps, train on what?", not "does extra data help per
#           epoch". The two cannot both be held fixed.
#
# There is deliberately no run for "the older three on their own": that is the
# existing g1_dex1_ikea_relarm_3view_aug_b64_v2_armvel, already trained at the
# identical recipe (eff 64, 20k, cosine, seed 42, armvel config, same data). Its
# checkpoint-20000 only needs re-scanning against the new val split, which costs
# a GPU-hour rather than nine. See scan_alltask.sh.
#
# 2 GPUs per run, not 4: at fixed effective batch this host's all-reduce rides
# PCIe (no NVLink) and 4 GPUs measured 2.24 s/it against 1.46 (EXPERIMENTS.md
# section 3). Three concurrent 2-GPU runs held 1.43-1.45 s/it there.
#
# 30k steps, and the reason is epochs rather than steps. The merged set holds
# 117,202 windows against the older set's 65,358, so at effective batch 64 an
# epoch costs 1831 steps instead of 1021. The deployed model needed 26k steps =
# 25.5 epochs before its gripper stopped improving (section 14); 20k here would
# be 10.9 epochs, under half of that, and the gripper is the metric that kept
# moving after arm and EE had settled. 30k buys 16.4 epochs. Going all the way to
# 25.5 epochs would cost 46k steps and ~21 h, which is not worth it for a run
# whose only job is to rank labelling schemes -- whichever wins gets a proper
# long run afterwards.
#
# Whether epochs is even the right invariant is untested: 1.79x the data at the
# same step count means each sample is seen less often but more samples are seen,
# and nothing here has measured which of those the gripper follows.
#
# Checkpoints: save every 2000 but keep only the last 8, i.e. 16k-30k. Nothing in
# sections 12 or 14 ever used a checkpoint from the first half -- the noise band
# came from 16/18/20k and the gripper saturation point from 22k-30k -- and a
# diverging run shows up in the loss log, not in an early checkpoint. Eight
# rather than three so the ranking between variants can be checked for stability
# across the second half instead of only at the end.
#   8 x 12 GB + the root final model 12 GB = 108 GB per run, 432 GB for four.
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
cd "$ROOT"
export PATH="$HOME/micromamba/envs/ffmpeg7/bin:$PATH"
# torchrun lives in the venv; without this every run dies instantly on
# "exec: torchrun: not found" and leaves the GPUs idle with no error in sight
source "$ROOT/.venv/bin/activate"
# The 46-dim state, NOT the 60-dim arm-velocity one that the deployed model uses.
# The competition boundary (EXPERIMENTS.md section 11) hands over body_q(29) and
# base_quat(4) and no velocity at all, so until that is settled a model that needs
# arm_dq cannot be entered -- and feeding it zeros costs +26% on the executed
# window, worse than never having had it. Arm velocity is worth 15% on arm8 and
# 10% on EE8 (section 14), so it goes back in once the rules are known: rerun the
# winning label scheme with
#   CONFIG=.../g1_dex1_ikea_armvel_config.py EXP_PREFIX=_armvel bash this
CONFIG="${CONFIG:-$ROOT/examples/unitree_g1_dex1_ikea/g1_dex1_ikea_relarm_3view_aug_config.py}"
SUFFIX_EXTRA="${EXP_PREFIX:-}"
STEPS="${MAX_STEPS:-30000}"
KEEP="${SAVE_TOTAL_LIMIT:-8}"
# 96 cores shared four ways, against 16 for a run on its own
WORKERS="${DATALOADER_NUM_WORKERS:-10}"

# examples/finetune.sh defaults MASTER_PORT to 29500, so four concurrent torchrun
# jobs would all try to bind the same rendezvous port. One per run.
PORT_BASE="${MASTER_PORT_BASE:-29510}"
port_offset=0

launch() {   # name gpus dataset_prefix
    local name="$1" gpus="$2" ds="$3"
    if [ ! -f "${ds}_train/meta/stats.json" ]; then
        echo "refusing to launch $name: ${ds}_train has no stats.json" >&2
        echo "  generate it first -- four runs racing to write it corrupts the file" >&2
        return 1
    fi
    echo "[$(date '+%F %T')] launching $name on GPU $gpus port $((PORT_BASE + port_offset))" \
         "<- $(basename "$ds")  [$(basename "$CONFIG")]"
    CUDA_VISIBLE_DEVICES="$gpus" \
    MASTER_PORT="$((PORT_BASE + port_offset))" \
    CONFIG="$CONFIG" \
    DATASET_ROOT="$ds" \
    EXP_SUFFIX="_alltask_$name$SUFFIX_EXTRA" \
    MAX_STEPS="$STEPS" \
    SAVE_TOTAL_LIMIT="$KEEP" \
    NUM_GPUS=2 \
    DATALOADER_NUM_WORKERS="$WORKERS" \
    nohup bash examples/unitree_g1_dex1_ikea/run_finetune_ikea.sh \
        --use-ddp --ddp-comm-bf16 \
        > "$ROOT/datasets/train_alltask_$name.log" 2>&1 &
    port_offset=$((port_offset + 1))
    sleep 20   # stagger, so four ranks do not open the same shards at once
}

# run_finetune_ikea.sh preflights disk for one run at a time, so four of them
# each see enough space and together fill the volume. Check the total here.
NEED=$(( 4 * (KEEP + 1) * 12 ))
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "disk: ${FREE} GB free, four runs need ~${NEED} GB"
if [ "$FREE" -lt "$NEED" ]; then
    echo "ERROR: not enough space for four concurrent runs." >&2
    echo "       free space, lower SAVE_TOTAL_LIMIT, or run fewer at once." >&2
    exit 1
fi

D="$ROOT/datasets/carroll511"
launch u 0,1 "$D/G1_Dex1_IKEA_all_30hz_unified"
launch s 2,3 "$D/G1_Dex1_IKEA_all_30hz_split"
launch n 4,5 "$D/G1_Dex1_IKEA_all_30hz_renamed"
launch x 6,7 "$D/G1_Dex1_IKEA_newonly_30hz"

echo "[$(date '+%F %T')] launched; logs in datasets/train_alltask_*.log"
