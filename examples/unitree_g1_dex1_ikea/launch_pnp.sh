#!/usr/bin/env bash
# Two concurrent 2-GPU runs over the pick-and-place merge.
#
#   P   46-dim state              GPU 0,1   vs the existing alltask_u
#   PV  60-dim, arm velocity      GPU 2,3   vs the existing alltask_uv
#
# What the merge is. `pick table leg` and `insert table leg to table base` were
# two labels over one continuous motion, and the recording proves it: the IKEA
# source is two videos, file-000 holding 42,943 frames whose episode lengths sum
# to exactly 42,943 over [0, 42943), file-001 likewise. No frame was ever
# discarded; an episode boundary is a label cut.
#
# Why cutting there costs something. The sampler takes a window start only where
# the whole horizon fits (`get_effective_episode_length = length - horizon + 1`,
# and `allow_padding` is False), so the last 39 frames of every episode are never
# a start. Cut at the grasp, and the moment between closing the gripper and
# lifting toward the base has no supervision at all -- which is what showed up on
# hardware as a bad hand-off between the two instructions.
#
# Not every boundary is joinable. A cut was also made whenever an attempt failed
# and had to be restarted, and those carry a real discontinuity even though the
# frame count cannot see one: 14 of the 63 pick/insert boundaries step an arm or
# gripper reading further in one frame than that dim ever steps inside an
# episode, by 1.1x to 15.3x. 13 of those 14 also jump in pixels, 4.6x to 14.3x
# the neighbouring frame gap, so the two signals agree. The converter refuses
# them (`--merge-jump-tol`) and leaves those pairs as separate episodes.
#
# Net: 178 source episodes -> 129, 48 fused groups, 128,200 frames unchanged and
# meta/stats.json bit-identical to the unified set. Window starts 117,202 ->
# 118,996; the 1,794 new ones all straddle a grasp.
#
# The instruction. `pick and place the table leg` sits 0.164 from the nearest
# surviving task in the frozen text stack, against the 0.1447 that `insert` and
# `rotate leg to tighten` already have -- the pair measured to interfere. So the
# merge does not make the label space tighter, it loosens it by retiring the
# closer of the two strings.
#
# Deliberately not run: a label-only arm, i.e. the same cuts with both episodes
# renamed. It would price the instruction separately from the recovered windows,
# but either result ships the merged dataset, so it buys nothing here. The
# converter keeps `--merge-label-only` if that question comes back.
#
# 2 GPUs per run rather than 4: at fixed effective batch this host has no NVLink
# and 4 GPUs measured 2.24 s/it against 1.46 (EXPERIMENTS.md section 3). Two runs
# therefore leave GPUs 4-7 idle and still finish sooner than spreading over 8.
#
# 30k steps and SAVE_TOTAL_LIMIT 8 are not free choices: they reproduce the
# checkpoint grid (16k-30k) that alltask_u and alltask_uv already sit on, so the
# comparison is checkpoint-for-checkpoint rather than best-against-best.
#   2 runs x 9 x 12 GB = 216 GB.
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
cd "$ROOT"
export PATH="$HOME/micromamba/envs/ffmpeg7/bin:$PATH"
# torchrun lives in the venv; without this every run dies instantly on
# "exec: torchrun: not found" and leaves the GPUs idle with no error in sight
source "$ROOT/.venv/bin/activate"

STEPS="${MAX_STEPS:-30000}"
KEEP="${SAVE_TOTAL_LIMIT:-8}"
# 96 cores shared two ways
WORKERS="${DATALOADER_NUM_WORKERS:-16}"
# examples/finetune.sh defaults MASTER_PORT to 29500, so concurrent torchrun jobs
# would all try to bind the same rendezvous port. One per run.
PORT_BASE="${MASTER_PORT_BASE:-29540}"
port_offset=0

DS="$ROOT/datasets/carroll511/G1_Dex1_IKEA_all_30hz_pnp"
E="$ROOT/examples/unitree_g1_dex1_ikea"

launch() {   # name gpus config
    local name="$1" gpus="$2" cfg="$3"
    if [ ! -f "${DS}_train/meta/stats.json" ]; then
        echo "refusing to launch $name: ${DS}_train has no stats.json" >&2
        echo "  generate it first -- two runs racing to write it corrupts the file" >&2
        return 1
    fi
    echo "[$(date '+%F %T')] launching $name on GPU $gpus port $((PORT_BASE + port_offset))" \
         "<- $(basename "$DS")  [$(basename "$cfg")]"
    CUDA_VISIBLE_DEVICES="$gpus" \
    MASTER_PORT="$((PORT_BASE + port_offset))" \
    CONFIG="$cfg" \
    DATASET_ROOT="$DS" \
    EXP_SUFFIX="_pnp_$name" \
    MAX_STEPS="$STEPS" \
    SAVE_TOTAL_LIMIT="$KEEP" \
    NUM_GPUS=2 \
    DATALOADER_NUM_WORKERS="$WORKERS" \
    nohup bash examples/unitree_g1_dex1_ikea/run_finetune_ikea.sh \
        --use-ddp --ddp-comm-bf16 \
        > "$ROOT/datasets/train_pnp_$name.log" 2>&1 &
    port_offset=$((port_offset + 1))
    sleep 20   # stagger, so both runs do not open the same shards at once
}

# run_finetune_ikea.sh preflights disk for one run at a time, so two of them each
# see enough space and together fill the volume. Check the total here.
NEED=$(( 2 * (KEEP + 1) * 12 ))
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "disk: ${FREE} GB free, two runs need ~${NEED} GB"
if [ "$FREE" -lt "$NEED" ]; then
    echo "ERROR: not enough space for two concurrent runs." >&2
    echo "       free space, lower SAVE_TOTAL_LIMIT, or run one at a time." >&2
    exit 1
fi

launch p  0,1 "$E/g1_dex1_ikea_relarm_3view_aug_config.py"
launch pv 2,3 "$E/g1_dex1_ikea_armvel_config.py"

echo "[$(date '+%F %T')] launched; logs in datasets/train_pnp_*.log"
