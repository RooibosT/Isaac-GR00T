#!/usr/bin/env bash
# Scan each state-dropout 0.4 run as it finishes.
#
# The two runs train in parallel on one GPU each, so their scans cannot both
# claim both GPUs. A finishes first and is scanned on GPU 0 alone (two shards on
# the same card); B is scanned on both once nothing else is training.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
LOG=$ROOT/datasets/scan_both_sd04.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
wait_for() { while pgrep -f "output_dir $ROOT/outputs/$1 " >/dev/null 2>&1; do sleep 60; done; sleep 45; }

A=g1_dex1_ikea_relarm_3view_aug_b64_stage1_absarm_armvel_sd04
B=g1_dex1_ikea_relarm_3view_aug_b64_stage1_absarm_sd04

say "waiting for A ($A)"
wait_for "$A"
say "A finished; scanning on GPU 0"
VAL=/root/02_hub/datasets/IKEA_pick_leg_stage1_val STRIDE=7 \
  bash "$EX/scan_when_done.sh" "$A" "$EX/g1_dex1_ikea_absarm_armvel_config.py" 0 0

say "waiting for B ($B)"
wait_for "$B"
say "B finished; scanning on GPU 0 and 1"
VAL=/root/02_hub/datasets/IKEA_pick_leg_stage1_val STRIDE=7 \
  bash "$EX/scan_when_done.sh" "$B" "$EX/g1_dex1_ikea_absarm_3view_aug_config.py" 0 1
say "both scans complete"
