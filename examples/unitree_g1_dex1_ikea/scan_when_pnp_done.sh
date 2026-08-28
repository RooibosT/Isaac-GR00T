#!/usr/bin/env bash
# Wait for the pick-and-place runs to exit, then scan them.
#
# Same shape as scan_when_alltask_done.sh: wait on the processes rather than on
# checkpoint-30000 appearing, so a run that dies early still gets whatever it
# wrote scanned instead of hanging here for ever, and bracket-escape the pattern
# so pgrep cannot match this script's own command line.
#
#   nohup bash scan_when_pnp_done.sh > datasets/scan_waiter_pnp.log 2>&1 &
set -uo pipefail

ROOT="/home/chan/IKEA/Isaac-GR00T"
cd "$ROOT"
RUNS="${RUNS:-p pv pvt leg}"
SCAN="${SCAN:-$RUNS}"

# the leg run lives under a different suffix, so its pattern is built separately
suffix_of() { case "$1" in leg) echo "leg_armvel";; rv|r|r1v) echo "2h_$1";; *) echo "pnp_$1";; esac; }

for r in $RUNS; do
    pat="output_dir $ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_$(suffix_of "$r" | sed 's/^\(.\)/[\1]/') "
    echo "[$(date '+%F %T')] waiting for run $r ..."
    while pgrep -f "$pat" > /dev/null 2>&1; do sleep 120; done
    echo "[$(date '+%F %T')] run $r has exited"
done

sleep 90   # let the final checkpoint and the wandb sync flush

echo "[$(date '+%F %T')] all runs done; checkpoints written:"
for r in $RUNS; do
    d="$ROOT/outputs/g1_dex1_ikea_relarm_3view_aug_b64_$(suffix_of "$r")/g1_dex1_ikea_relarm_3view_aug_b64_$(suffix_of "$r")"
    n=$(find "$d" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | wc -l)
    echo "   $r: $n checkpoints"
done

# shellcheck disable=SC2086
bash "$ROOT/examples/unitree_g1_dex1_ikea/scan_alltask.sh" $SCAN
echo "[$(date '+%F %T')] scanning finished"
