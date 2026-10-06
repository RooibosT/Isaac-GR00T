#!/usr/bin/env bash
# Upload the scan-selected checkpoint of the nature_pouch_new waist run, once its chain has
# finished training and scanning.
#
# Waits on the chain's own completion line rather than pgrep (see chain_threadsnap_upload.sh).
#
# The pick is arm8 first, as for every IKEA card, but this model has a second output the arm
# metric cannot see: `waist_yaw`. Checkpoints within 1% of the arm8 minimum are inside the
# noise of that metric (see chain_alltask4_allvel_upload.sh), so among them the one with the
# lowest `mae_waist_first8` is taken. When that is not the arm8 argmin itself, it is passed as
# --checkpoint and the card says why. The card's scan table has no waist column, so a per-
# checkpoint waist table goes in --extra-note. Set CHECKPOINT=<step> to override the pick.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_naturepouchnew_absarm_waist
REPO=${REPO:-RooibosT/gr00t-n1.7-g1-dex1-naturepouchnew-absarm-waist-30hz-h40}
CHAIN_LOG=$ROOT/datasets/chain_nature_pouch_new_absarm_waist.log
LOG=$ROOT/datasets/upload_naturepouchnew.log
NOTE=$ROOT/datasets/naturepouchnew_extra_note.md
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the training + scan chain to log its completion"
until grep -q "chain complete" "$CHAIN_LOG" 2>/dev/null; do sleep 60; done
say "it finished"

source "$ROOT/.venv/bin/activate"
SCAN=$ROOT/outputs/$B/$B/scan.json
[ -f "$SCAN" ] || { say "NO scan.json at $SCAN -- nothing to upload"; exit 1; }

PICK=$(python - "$SCAN" "$NOTE" <<'EOF'
import json, sys
import numpy as np

scan, note = sys.argv[1], sys.argv[2]
d = json.load(open(scan))
a = {k: v["__all__"] for k, v in d.items()}
step = lambda k: int(k.split("-")[1])
ks = sorted(a, key=step)
arm_best = min(ks, key=lambda k: a[k]["mae_arm_first8"])
if "mae_waist_first8" not in a[arm_best]:
    print(f"no waist metric in the scan; keeping the arm8 pick {arm_best}", file=sys.stderr)
    open(note, "w").write("")
    print("")
    sys.exit(0)
band = [k for k in ks if a[k]["mae_arm_first8"] <= 1.01 * a[arm_best]["mae_arm_first8"]]
pick = min(band, key=lambda k: a[k]["mae_waist_first8"])
waist_best = min(ks, key=lambda k: a[k]["mae_waist_first8"])
deg = np.degrees
rows = "\n".join(
    (f"| **`{k}`** (this card) |" if k == pick else f"| `{k}` |")
    + f" {deg(a[k]['mae_arm_first8']):.3f} | {deg(a[k]['mae_waist_first8']):.3f} |"
    f" {deg(a[k]['mae_waist']):.3f} |"
    for k in ks)
why = (
    f"`{pick}` is the `mae_arm_first8` minimum and also the lowest waist8 among checkpoints "
    f"within 1% of it."
    if pick == arm_best else
    f"`mae_arm_first8` bottoms at `{arm_best}` ({deg(a[arm_best]['mae_arm_first8']):.3f}°). "
    f"{len(band)} checkpoints sit within 1% of that, which is inside the metric's noise, and "
    f"among them `{pick}` has the lowest waist8: {deg(a[pick]['mae_waist_first8']):.3f}° "
    f"against {deg(a[arm_best]['mae_waist_first8']):.3f}°, at an arm8 cost of "
    f"{100 * (a[pick]['mae_arm_first8'] / a[arm_best]['mae_arm_first8'] - 1):.1f}%."
)
text = f"""
---

## Waist command (`waist_yaw`, action dim 16)

The table above scores the arms only. This one scores the waist yaw output on the same
windows, in degrees (the full turn is 45°).

| ckpt | arm8° | **waist8°** | waist° |
|---|---:|---:|---:|
{rows}

## How this checkpoint was chosen

{why} The overall waist8 minimum is `{waist_best}` ({deg(a[waist_best]['mae_waist_first8']):.3f}°).
"""
open(note, "w").write(text)
print("" if pick == arm_best else step(pick))
EOF
) || { say "pick step FAILED"; exit 1; }
CKPT=${CHECKPOINT:-$PICK}
say "pick: ${CKPT:-arm8 argmin}"

ACTION_DESC='**Action: 17-dim, ABSOLUTE joint targets** -- left arm 7, right arm 7, both
  grippers 1 each, then **`waist_yaw` 1 (dim 16)**, horizon 40 at 30 Hz. `waist_yaw` is the
  absolute waist yaw target in radians (0 = facing forward, -0.785 = turned 45 deg to the
  robot'"'"'s right) that `teleop_ikea.py` sends on `rt/arm_sdk` slot 12. **Deployment must unpack
  17 dims and send the last one to the waist**; a 16-dim client drops it and the robot never turns.'

say "uploading ${CKPT:+checkpoint-$CKPT of }$B -> $REPO"
python "$HUB/upload_best_ckpt.py" \
    --exp "$B" \
    --repo "$REPO" \
    --config g1_dex1_ikea_absarm_waist_config.py \
    --dataset-repo RooibosT/nature_pouch_new \
    --train-eps 89 --train-frames 39002 --val-eps 10 --val-frames 4504 \
    --max-steps 9000 --save-steps 500 --scan-stride 4 \
    --task-title "IKEA pick and place the pouch, with a waist turn" \
    --data-note "$(cat "$HUB/naturepouchnew_data_note.txt")" \
    --state-dim 46 \
    --state-desc "legs 12, waist 3 (yaw, roll, pitch), both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6." \
    --action-desc "$ACTION_DESC" \
    --extra-note "$(cat "$NOTE")" \
    ${CKPT:+--checkpoint "$CKPT"} \
    >> "$LOG" 2>&1 && say "upload done" || say "UPLOAD FAILED (see $LOG)"
