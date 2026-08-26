#!/usr/bin/env bash
# Follow-up to the label ablation (EXPERIMENTS.md section 18): can the older
# three tasks be weighted back up, and what does arm velocity add on top?
#
#   m16  46-dim, older three weighted x1.6   GPU 0,1
#   m30  46-dim, older three weighted x3.0   GPU 2,3
#   uv   60-dim arm velocity, natural mix    GPU 4,5
#   mv   60-dim arm velocity, weighted x1.6  GPU 6,7
#
# Why weight at all: `insert` lost 12.1% on EE against a model trained on the
# older three alone, and it also dropped from 19.6 epochs to 16.4 when the new
# tasks joined. x1.6 puts exactly those epochs back; x3.0 overshoots to 23.2 so
# the two bracket the effect rather than testing one guess.
#
# The honest expectation is low. Section 18 showed `insert` flat from 22k to 30k
# (EE8 +0.30%) while every other task kept improving, so it is saturated rather
# than starved -- which points at interference, not dilution, and weighting only
# addresses dilution. More `insert` episodes would address both. This runs
# because the data collection has not happened yet and the GPUs are free.
#
# Why arm velocity again: it is worth 15% on arm error and 10% on EE (section
# 14) and cannot be entered until the competition boundary is settled, so these
# are the models to have ready if it is. `..._v2_armvel/checkpoint-20000` is the
# matching old-tasks-only control and only needs re-scanning.
#
# Caveat on m16/m30: they feed two datasets through the mixture path while U fed
# one merged copy of the same episodes. At ratio 1.0 the two are equivalent in
# sampling, but `merge_statistics` weights the normalisation by the same ratio,
# so a weighted run also shifts its statistics toward the older data. The two
# changes are not separated here.
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
cd "$ROOT"
export PATH="$HOME/micromamba/envs/ffmpeg7/bin:$PATH"
source "$ROOT/.venv/bin/activate"       # torchrun lives here

BASE_CFG="$ROOT/examples/unitree_g1_dex1_ikea/g1_dex1_ikea_relarm_3view_aug_config.py"
VEL_CFG="$ROOT/examples/unitree_g1_dex1_ikea/g1_dex1_ikea_armvel_config.py"
MIX="$ROOT/examples/unitree_g1_dex1_ikea/mixtures/ikea_plus_new_tasks.json"
OLD="$ROOT/datasets/carroll511/G1_Dex1_IKEA_table_30hz_v2"
MERGED="$ROOT/datasets/carroll511/G1_Dex1_IKEA_all_30hz_unified"
STEPS="${MAX_STEPS:-30000}"
KEEP="${SAVE_TOTAL_LIMIT:-8}"
WORKERS="${DATALOADER_NUM_WORKERS:-10}"

NEED=$(( 4 * (KEEP + 1) * 12 ))
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "disk: ${FREE} GB free, four runs need ~${NEED} GB"
[ "$FREE" -lt "$NEED" ] && { echo "ERROR: not enough space" >&2; exit 1; }

PORT_BASE="${MASTER_PORT_BASE:-29520}"
off=0

launch() {   # name gpus config dataset_root [mix_ratio]
    local name="$1" gpus="$2" cfg="$3" ds="$4" ratio="${5:-}"
    [ -f "${ds}_train/meta/stats.json" ] || { echo "refusing $name: ${ds}_train has no stats" >&2; return 1; }
    local extra=(--use-ddp --ddp-comm-bf16)
    [ -n "$ratio" ] && extra+=(--mixture-spec "$MIX" --mix-ratio "$ratio")
    echo "[$(date '+%F %T')] $name  GPU $gpus  port $((PORT_BASE+off))  $(basename "$cfg")  $(basename "$ds")${ratio:+  x$ratio}"
    CUDA_VISIBLE_DEVICES="$gpus" MASTER_PORT="$((PORT_BASE+off))" \
    CONFIG="$cfg" DATASET_ROOT="$ds" EXP_SUFFIX="_alltask_$name" \
    MAX_STEPS="$STEPS" SAVE_TOTAL_LIMIT="$KEEP" NUM_GPUS=2 \
    DATALOADER_NUM_WORKERS="$WORKERS" \
    nohup bash examples/unitree_g1_dex1_ikea/run_finetune_ikea.sh "${extra[@]}" \
        > "$ROOT/datasets/train_alltask_$name.log" 2>&1 &
    off=$((off+1)); sleep 20
}

launch m16 0,1 "$BASE_CFG" "$OLD"    1.6
launch m30 2,3 "$BASE_CFG" "$OLD"    3.0
launch uv  4,5 "$VEL_CFG"  "$MERGED"
launch mv  6,7 "$VEL_CFG"  "$OLD"    1.6

echo "[$(date '+%F %T')] launched; logs in datasets/train_alltask_{m16,m30,uv,mv}.log"
