#!/usr/bin/env bash
# Upload the scan-selected checkpoint of the nature_pouch_new REL-arm waist run, once its chain
# has finished training and scanning.
#
# The REL twin of chain_naturepouchnew_upload.sh, with the same pick rule: arm8 first, and
# among the checkpoints within 1% of the arm8 minimum the one with the lowest
# `mae_waist_first8`. When the ABS run's scan.json exists, the card also carries a short table of
# both runs' picks. Their val windows are the same, so the rows compare directly.
# Set CHECKPOINT=<step> to override the pick.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_naturepouchnew_relarm_waist
ABS_B=g1_dex1_ikea_relarm_3view_aug_b64_naturepouchnew_absarm_waist
REPO=${REPO:-RooibosT/gr00t-n1.7-g1-dex1-naturepouchnew-relarm-waist-30hz-h40}
CHAIN_LOG=$ROOT/datasets/chain_nature_pouch_new_relarm_waist.log
LOG=$ROOT/datasets/upload_naturepouchnew_relarm.log
NOTE=$ROOT/datasets/naturepouchnew_relarm_extra_note.md
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the training + scan chain to log its completion"
until grep -q "chain complete" "$CHAIN_LOG" 2>/dev/null; do sleep 60; done
say "it finished"

source "$ROOT/.venv/bin/activate"
SCAN=$ROOT/outputs/$B/$B/scan.json
ABS_SCAN=$ROOT/outputs/$ABS_B/$ABS_B/scan.json
[ -f "$SCAN" ] || { say "NO scan.json at $SCAN -- nothing to upload"; exit 1; }

PICK=$(python - "$SCAN" "$NOTE" "$ABS_SCAN" <<'EOF'
import json, os, sys
import numpy as np

scan, note, abs_scan = sys.argv[1], sys.argv[2], sys.argv[3]
deg = np.degrees
step = lambda k: int(k.split("-")[1])


def pick_of(d):
    """arm8 argmin, then the lowest waist8 among checkpoints within 1% of it."""
    a = {k: v["__all__"] for k, v in d.items()}
    ks = sorted(a, key=step)
    arm_best = min(ks, key=lambda k: a[k]["mae_arm_first8"])
    if "mae_waist_first8" not in a[arm_best]:
        return a, ks, arm_best, arm_best, [arm_best]
    band = [k for k in ks if a[k]["mae_arm_first8"] <= 1.01 * a[arm_best]["mae_arm_first8"]]
    return a, ks, arm_best, min(band, key=lambda k: a[k]["mae_waist_first8"]), band


a, ks, arm_best, pick, band = pick_of(json.load(open(scan)))
if "mae_waist_first8" not in a[arm_best]:
    print(f"no waist metric in the scan; keeping the arm8 pick {arm_best}", file=sys.stderr)
    open(note, "w").write("")
    print("")
    sys.exit(0)
waist_best = min(ks, key=lambda k: a[k]["mae_waist_first8"])
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

twin = ""
if os.path.exists(abs_scan):
    b, _, _, b_pick, _ = pick_of(json.load(open(abs_scan)))
    if "mae_waist_first8" in b[b_pick]:
        r, s = a[pick], b[b_pick]
        pct = lambda m: 100 * (r[m] / s[m] - 1)
        twin = f"""
## Against the ABSOLUTE twin

The same split, schedule, state and 17-dim layout, with the two arm blocks ABSOLUTE
(`{os.path.basename(os.path.dirname(abs_scan))}`). Each row is that run's pick under the same
rule. The two runs were scored on the same val windows.

| run | ckpt | arm8° | EE8 mm | grip | waist8° |
|---|---|---:|---:|---:|---:|
| **REL arms (this card)** | `{pick}` | {deg(r['mae_arm_first8']):.3f} | {r['ee_mm_first8']:.2f} | {r['mae_grip']:.4f} | {deg(r['mae_waist_first8']):.3f} |
| ABS arms | `{b_pick}` | {deg(s['mae_arm_first8']):.3f} | {s['ee_mm_first8']:.2f} | {s['mae_grip']:.4f} | {deg(s['mae_waist_first8']):.3f} |
| REL vs ABS | | {pct('mae_arm_first8'):+.1f}% | {pct('ee_mm_first8'):+.1f}% | {pct('mae_grip'):+.1f}% | {pct('mae_waist_first8'):+.1f}% |
"""
        print(f"REL {pick} vs ABS {b_pick}: arm8 {pct('mae_arm_first8'):+.1f}%  "
              f"EE8 {pct('ee_mm_first8'):+.1f}%  waist8 {pct('mae_waist_first8'):+.1f}%",
              file=sys.stderr)
else:
    print(f"no ABS scan at {abs_scan}; the card goes up without the twin table", file=sys.stderr)

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
{twin}"""
open(note, "w").write(text)
print("" if pick == arm_best else step(pick))
EOF
) || { say "pick step FAILED"; exit 1; }
CKPT=${CHECKPOINT:-$PICK}
say "pick: ${CKPT:-arm8 argmin}"

ACTION_DESC='**Action: 17-dim** -- left arm 7, right arm 7 (**RELATIVE**), both grippers 1 each
  (ABSOLUTE), then **`waist_yaw` 1 (dim 16, ABSOLUTE)**, horizon 40 at 30 Hz. The model predicts
  each arm target as an offset from the arm state at the observed frame; the policy adds that
  state back, so `get_action` returns absolute joint targets in the same 17-dim layout as the
  ABSOLUTE twin. `waist_yaw` is the absolute waist yaw target in radians (0 = facing forward,
  -0.785 = turned 45 deg to the robot'"'"'s right) that `teleop_ikea.py` sends on `rt/arm_sdk`
  slot 12. **Deployment must unpack 17 dims and send the last one to the waist**; a 16-dim
  client drops it and the robot never turns.'

say "uploading ${CKPT:+checkpoint-$CKPT of }$B -> $REPO"
python "$HUB/upload_best_ckpt.py" \
    --exp "$B" \
    --repo "$REPO" \
    --config g1_dex1_ikea_relarm_waist_config.py \
    --action-rep RELATIVE \
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
