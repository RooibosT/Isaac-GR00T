#!/usr/bin/env bash
# Sequential action-representation ablations for G1 Dex1 pick-redblock.
# Runs variants B, A40, B40 (5000 steps each) one after another.
#
# Usage (after the baseline A run has finished):
#   bash examples/unitree_g1_dex1/run_ablations.sh
#
# A failed variant is logged and the next one still runs. Relative-action
# stats are regenerated automatically at training start when a variant's
# fingerprint (rep/horizon) differs from the cached one.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export HF_HOME=/NHNHOME/WORKSPACE/chan/.cache/huggingface

BASE_MODEL_PATH=/NHNHOME/WORKSPACE/chan/models/GR00T-N1.7-3B
DATASET_PATH=demo_data/RooibosT/g1_pick_redblock_dex1_sim_merged_107demo
OUTPUT_ROOT=/NHNHOME/WORKSPACE/chan/outputs
WANDB_PROJECT_NAME=gr00t-n1.7-g1-dex1
MAX_STEPS="${MAX_STEPS:-5000}"

# Refuse to start while another training run is still using the GPUs.
if [ "${FORCE:-0}" != "1" ]; then
    BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "$BUSY" -gt 0 ]; then
        echo "GPUs are busy ($BUSY compute process(es) running)." >&2
        echo "Wait for the current run to finish, or re-run with FORCE=1." >&2
        exit 1
    fi
fi

# variant_name : modality config path
VARIANTS=(
    "B:examples/unitree_g1_dex1/variants/g1_dex1_config_b.py"
    "A40:examples/unitree_g1_dex1/variants/g1_dex1_config_a40.py"
    "B40:examples/unitree_g1_dex1/variants/g1_dex1_config_b40.py"
)

declare -A RESULTS

for entry in "${VARIANTS[@]}"; do
    name="${entry%%:*}"
    config="${entry#*:}"
    output_dir="$OUTPUT_ROOT/g1_dex1_ablation_${name}"
    experiment_name="g1_dex1_${name}"

    echo "=============================================================="
    echo "[$(date '+%F %T')] Starting variant ${name} (${config})"
    echo "  output: ${output_dir}"
    echo "=============================================================="
    mkdir -p "$output_dir"

    numactl --cpunodebind=1 --membind=1 -- env \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
        NUM_GPUS=2 USE_WANDB=1 GLOBAL_BATCH_SIZE=32 DATALOADER_NUM_WORKERS=4 \
        MAX_STEPS="$MAX_STEPS" SAVE_STEPS=1000 \
        bash examples/finetune.sh \
        --base-model-path "$BASE_MODEL_PATH" \
        --dataset-path "$DATASET_PATH" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "$config" \
        --wandb-project "$WANDB_PROJECT_NAME" \
        --experiment-name "$experiment_name" \
        --output-dir "$output_dir" 2>&1 | tee "$output_dir/train.log"
    status=${PIPESTATUS[0]}

    if [ "$status" -eq 0 ]; then
        RESULTS[$name]="OK"
        echo "[$(date '+%F %T')] Variant ${name} finished."
    else
        RESULTS[$name]="FAILED (exit ${status})"
        echo "[$(date '+%F %T')] Variant ${name} FAILED with exit ${status} — see $output_dir/train.log" >&2
    fi
done

echo
echo "================== Ablation summary =================="
for entry in "${VARIANTS[@]}"; do
    name="${entry%%:*}"
    printf '  %-5s %s\n' "$name" "${RESULTS[$name]}"
done
echo "======================================================"
