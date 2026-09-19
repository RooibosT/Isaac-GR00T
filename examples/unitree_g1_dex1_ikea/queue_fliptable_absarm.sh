#!/usr/bin/env bash
# Start the 46-dim fliptable run once the thread_snapshot upload has finished,
# so the two never share the GPUs. Waits on log lines, not process names.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
UP=$ROOT/datasets/chain_threadsnap_upload.log
LOG=$ROOT/datasets/queue_fliptable_absarm.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the thread_snapshot upload to finish"
until grep -qE "upload done|UPLOAD FAILED|nothing to upload" "$UP" 2>/dev/null; do sleep 60; done
say "it finished: $(grep -oE 'upload done|UPLOAD FAILED|nothing to upload' "$UP" | tail -1)"
bash "$EX/chain_fliptable_absarm.sh" >> "$LOG" 2>&1
say "queued run complete (exit $?)"
