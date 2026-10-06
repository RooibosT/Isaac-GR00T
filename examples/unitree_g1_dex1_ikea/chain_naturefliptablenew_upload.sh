#!/usr/bin/env bash
# Upload the scan-selected checkpoint of the nature_fliptable_new run, once its chain has
# finished training and scanning.
#
# Waits on the chain's own completion line rather than pgrep (see chain_threadsnap_upload.sh).
#
# The pick is arm8 first, as for every IKEA card. Checkpoints within 1% of the arm8 minimum are
# inside the noise of that metric (see chain_alltask4_allvel_upload.sh), so among them the one
# with the lowest EE8 is taken. When that is not the arm8 argmin itself, it is passed as
# --checkpoint and the card says why. The card also scores the two earlier nature models on
# this same val (the chain scanned them before training), since no other card's numbers line up
# with this val. Set CHECKPOINT=<step> to override the pick.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_naturefliptablenew_absarm
REPO=${REPO:-RooibosT/gr00t-n1.7-g1-dex1-naturefliptablenew-absarm-30hz-h40}
CHAIN_LOG=$ROOT/datasets/chain_nature_fliptable_new_absarm.log
LOG=$ROOT/datasets/upload_naturefliptablenew.log
NOTE=$ROOT/datasets/naturefliptablenew_extra_note.md
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the training + scan chain to log its completion"
until grep -q "chain complete" "$CHAIN_LOG" 2>/dev/null; do sleep 60; done
say "it finished"

source "$ROOT/.venv/bin/activate"
SCAN=$ROOT/outputs/$B/$B/scan.json
[ -f "$SCAN" ] || { say "NO scan.json at $SCAN -- nothing to upload"; exit 1; }

PICK=$(python - "$SCAN" "$NOTE" "$ROOT/outputs/$B" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np

scan, note, base_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
d = json.load(open(scan))
a = {k: v["__all__"] for k, v in d.items()}
step = lambda k: int(k.split("-")[1])
ks = sorted(a, key=step)
deg = np.degrees
arm_best = min(ks, key=lambda k: a[k]["mae_arm_first8"])
band = [k for k in ks if a[k]["mae_arm_first8"] <= 1.01 * a[arm_best]["mae_arm_first8"]]
pick = min(band, key=lambda k: a[k]["ee_mm_first8"])
p = a[pick]

why = (
    f"`{pick}` is the `mae_arm_first8` minimum and also the lowest EE8 among checkpoints "
    f"within 1% of it."
    if pick == arm_best else
    f"`mae_arm_first8` bottoms at `{arm_best}` ({deg(a[arm_best]['mae_arm_first8']):.3f}°). "
    f"{len(band)} checkpoints sit within 1% of that, which is inside the metric's noise, and "
    f"among them `{pick}` has the lowest EE8: {p['ee_mm_first8']:.2f} mm against "
    f"{a[arm_best]['ee_mm_first8']:.2f} mm, at an arm8 cost of "
    f"{100 * (p['mae_arm_first8'] / a[arm_best]['mae_arm_first8'] - 1):.1f}%."
)
last = ks[-1]
tail = (f"The last checkpoint, `{last}`, is "
        f"{100 * (a[last]['mae_arm_first8'] / p['mae_arm_first8'] - 1):+.1f}% on arm8 and "
        f"{100 * (a[last]['ee_mm_first8'] / p['ee_mm_first8'] - 1):+.1f}% on EE8 against it.")

baselines = [
    ("`RooibosT/gr00t-n1.7-g1-dex1-naturefliptable-absarm-30hz-h40` (checkpoint-13000), "
     "`nature_fliptable` alone", "naturefliptable_ckpt13000_on_new_val.json"),
    ("`RooibosT/gr00t-n1.7-g1-dex1-natureflippouch-absarm-waist-30hz-h40` (checkpoint-20000), "
     "`nature_fliptable` + `nature_pouch_new`", "natureflippouchw_ckpt20000_on_new_val.json"),
]
rows = []
for name, f in baselines:
    fp = base_dir / f
    if not fp.exists():
        continue
    r = json.loads(fp.read_text())
    b = next(iter(r.values()))["__all__"]
    rel = lambda m: 100 * (p[m] / b[m] - 1)
    rows.append(
        f"| {name} | {deg(b['mae_arm_first8']):.3f} | {b['ee_mm_first8']:.2f} | "
        f"{b['mae_grip']:.4f} | {rel('mae_arm_first8'):+.0f}% / {rel('ee_mm_first8'):+.0f}% / "
        f"{rel('mae_grip'):+.0f}% |")
cmp = ""
if rows:
    cmp = f"""
## Compared with the earlier nature models, on this same val

Neither of these trained on session `table2`, so this is a leak-free number for how they
transfer to the new background. Same windows, same stride. The last column is this card's
`{pick}` relative to each (arm8 / EE8 / grip).

| model | arm8° | EE8 mm | grip | this model vs it |
|---|---:|---:|---:|---:|
| **this model, `{pick}`** | **{deg(p['mae_arm_first8']):.3f}** | **{p['ee_mm_first8']:.2f}** | **{p['mae_grip']:.4f}** | |
""" + "\n".join(rows) + """

Open-loop, the earlier nature models do not carry over to this session's background, and this
model has never seen theirs; nothing here says it works in the `nature_fliptable` setting.
"""

text = f"""
---

## How this checkpoint was chosen

{why} {tail}
{cmp}"""
open(note, "w").write(text)
print("" if pick == arm_best else step(pick))
EOF
) || { say "pick step FAILED"; exit 1; }
CKPT=${CHECKPOINT:-$PICK}
say "pick: ${CKPT:-arm8 argmin}"

say "uploading ${CKPT:+checkpoint-$CKPT of }$B -> $REPO"
python "$HUB/upload_best_ckpt.py" \
    --exp "$B" \
    --repo "$REPO" \
    --config g1_dex1_ikea_absarm_3view_aug_config.py \
    --dataset-repo RooibosT/nature_fliptable_new \
    --train-eps 99 --train-frames 76217 --val-eps 10 --val-frames 8468 \
    --max-steps 18000 --save-steps 1000 --scan-stride 4 \
    --task-title "IKEA flip the table, nature room session 2" \
    --data-note "$(cat "$HUB/naturefliptablenew_data_note.txt")" \
    --state-dim 46 \
    --state-desc "legs 12, waist 3, both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6." \
    --extra-note "$(cat "$NOTE")" \
    ${CKPT:+--checkpoint "$CKPT"} ${DRY_RUN:+--dry-run} \
    >> "$LOG" 2>&1 && say "upload done" || say "UPLOAD FAILED (see $LOG)"
