#!/usr/bin/env bash
# Wait for the four label runs to exit, then scan them and the control.
#
# Waits on the processes rather than on checkpoint-30000 appearing, so a run
# that dies early still gets whatever it wrote scanned instead of hanging here
# for ever. Each run is matched on its own --output_dir with a trailing space,
# because the four names share the prefix g1_dex1_ikea_relarm_3view_aug_b64_
# and a bare match would wait on all of them at once.
#
# The pattern is bracket-escaped so pgrep cannot match this script's own command
# line -- an unescaped one matches itself and the wait returns immediately, or
# never.
#
#   nohup bash scan_when_alltask_done.sh > datasets/scan_waiter.log 2>&1 &
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
cd "$ROOT"
RUNS="${RUNS:-u s n x}"
# what to scan once those exit; controls are appended so they run last
SCAN="${SCAN:-$RUNS ctl}"

for r in $RUNS; do
    pat="output_dir $ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltas[k]_$r "
    echo "[$(date '+%F %T')] waiting for run $r ..."
    while pgrep -f "$pat" > /dev/null 2>&1; do sleep 120; done
    echo "[$(date '+%F %T')] run $r has exited"
done

sleep 90   # let the final checkpoint and the wandb sync flush

echo "[$(date '+%F %T')] all runs done; checkpoints written:"
for r in $RUNS; do
    d="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_alltask_$r/g1_dex1_ikea_relarm_3view_aug_b64_alltask_$r"
    n=$(find "$d" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | wc -l)
    echo "   $r: $n checkpoints"
done

# shellcheck disable=SC2086
bash "$ROOT/examples/unitree_g1_dex1_ikea/scan_alltask.sh" $SCAN
echo "[$(date '+%F %T')] scanning finished"
