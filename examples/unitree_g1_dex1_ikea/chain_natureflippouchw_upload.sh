#!/usr/bin/env bash
# Upload the scan-selected checkpoint of the two-task nature run (flip + pouch with the waist
# turn), once its chain has finished training and scanning.
#
# Waits on the chain's own completion line rather than pgrep (see chain_threadsnap_upload.sh).
#
# Pick rule as chain_naturepouchnew_upload.sh: `mae_arm_first8` over both instructions first,
# then, among checkpoints within 1% of that minimum (the metric's noise), the lowest
# `mae_waist_first8`. The extra note carries what the card's own tables cannot: a per-
# checkpoint split of arm8 and waist8 by instruction, and the two specialists against this
# pick on the same windows (stride 4 on the same val episodes, so the window counts match
# exactly). Set CHECKPOINT=<step> to override the pick.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_natureflippouchw_absarm_waist
REPO=${REPO:-RooibosT/gr00t-n1.7-g1-dex1-natureflippouch-absarm-waist-30hz-h40}
CHAIN_LOG=$ROOT/datasets/chain_nature_flip_pouchw_absarm_waist.log
LOG=$ROOT/datasets/upload_natureflippouchw.log
NOTE=$ROOT/datasets/natureflippouchw_extra_note.md
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for the training + scan chain to log its completion"
until grep -q "chain complete" "$CHAIN_LOG" 2>/dev/null; do sleep 60; done
say "it finished"

source "$ROOT/.venv/bin/activate"
SCAN=$ROOT/outputs/$B/$B/scan.json
[ -f "$SCAN" ] || { say "NO scan.json at $SCAN -- nothing to upload"; exit 1; }

PICK=$(python - "$SCAN" "$NOTE" "$ROOT/outputs" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np

scan, note, outputs = sys.argv[1], sys.argv[2], Path(sys.argv[3])
FLIP, POUCH = "flip the table", "pick and place the pouch"
SPECIALISTS = {  # instruction -> (run, uploaded checkpoint)
    FLIP: ("g1_dex1_ikea_relarm_3view_aug_b64_naturefliptable_absarm", "checkpoint-13000"),
    POUCH: ("g1_dex1_ikea_relarm_3view_aug_b64_naturepouchnew_absarm_waist", "checkpoint-8500"),
}
d = json.load(open(scan))
step = lambda k: int(k.split("-")[1])
ks = sorted(d, key=step)
m = lambda k, metric, g="__all__": d[k][g][metric]
arm_best = min(ks, key=lambda k: m(k, "mae_arm_first8"))
band = [k for k in ks if m(k, "mae_arm_first8") <= 1.01 * m(arm_best, "mae_arm_first8")]
pick = min(band, key=lambda k: m(k, "mae_waist_first8"))
deg = np.degrees

rows = "\n".join(
    (f"| **`{k}`** (this card) |" if k == pick else f"| `{k}` |")
    + f" {deg(m(k, 'mae_arm_first8')):.3f} | {deg(m(k, 'mae_arm_first8', FLIP)):.3f} |"
    f" {deg(m(k, 'mae_arm_first8', POUCH)):.3f} | {deg(m(k, 'mae_waist_first8', FLIP)):.3f} |"
    f" {deg(m(k, 'mae_waist_first8', POUCH)):.3f} |"
    for k in ks)

cmp_rows = []
for g, (run, ck) in SPECIALISTS.items():
    s = json.load(open(outputs / run / run / "scan.json"))[ck]["__all__"]
    assert s["n"] == d[pick][g]["n"], f"{g}: specialist scored {s['n']} windows, this run {d[pick][g]['n']}"
    a8s, a8 = deg(s["mae_arm_first8"]), deg(m(pick, "mae_arm_first8", g))
    e8s, e8 = s["ee_mm_first8"], m(pick, "ee_mm_first8", g)
    w8s = f"{deg(s['mae_waist_first8']):.3f}°" if "mae_waist_first8" in s else "— (no waist output)"
    cmp_rows.append(
        f"| `{g}` | {s['n']:,} | `{run.split('b64_')[1]}` `{ck}` | {a8s:.3f} → **{a8:.3f}** "
        f"({100 * (a8 / a8s - 1):+.1f}%) | {e8s:.2f} → **{e8:.2f}** ({100 * (e8 / e8s - 1):+.1f}%) | "
        f"{w8s} → **{deg(m(pick, 'mae_waist_first8', g)):.3f}°** |")

why = (
    f"`{pick}` is the `mae_arm_first8` minimum over both instructions and also the lowest "
    f"waist8 among checkpoints within 1% of it."
    if pick == arm_best else
    f"`mae_arm_first8` over both instructions bottoms at `{arm_best}` "
    f"({deg(m(arm_best, 'mae_arm_first8')):.3f}°). {len(band)} checkpoints sit within 1% of "
    f"that, inside the metric's noise, and among them `{pick}` has the lowest waist8: "
    f"{deg(m(pick, 'mae_waist_first8')):.3f}° against {deg(m(arm_best, 'mae_waist_first8')):.3f}°, "
    f"at an arm8 cost of {100 * (m(pick, 'mae_arm_first8') / m(arm_best, 'mae_arm_first8') - 1):.1f}%."
)
text = f"""
---

## Against the two specialists, on the same windows

Each specialist was trained on one of the two sets and scored at stride 4 on its own val. These
are the same episodes and the same stride, so the window counts match exactly. Arrows read
specialist → this model.

| instruction | windows | specialist | arm8° | EE8 mm | waist8° |
|---|---:|---|---|---|---|
{chr(10).join(cmp_rows)}

## Per instruction, every checkpoint

arm8 and waist8 in degrees. The full waist turn is 45°; for `flip the table` the target is 0
throughout, so waist8 there measures how still the model keeps the waist.

| ckpt | arm8 (both) | flip arm8 | pouch arm8 | flip waist8 | pouch waist8 |
|---|---:|---:|---:|---:|---:|
{rows}

## How this checkpoint was chosen

{why}
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
  robot'"'"'s right) that `teleop_ikea.py` sends on `rt/arm_sdk` slot 12; it is trained as 0 for
  `flip the table`. **Deployment must unpack 17 dims and send the last one to the waist**.'

say "uploading ${CKPT:+checkpoint-$CKPT of }$B -> $REPO"
python "$HUB/upload_best_ckpt.py" \
    --exp "$B" \
    --repo "$REPO" \
    --config g1_dex1_ikea_absarm_waist_config.py \
    --dataset-repo "RooibosT/nature_fliptable + nature_pouch_new (merged locally -- no single hub id holds this split)" \
    --train-eps 194 --train-frames 108948 --val-eps 20 --val-frames 11857 \
    --max-steps 22000 --save-steps 1000 --scan-stride 4 \
    --task-title "IKEA nature room, flip the table + pick and place the pouch with a waist turn" \
    --data-note "$(cat "$HUB/natureflippouchw_data_note.txt")" \
    --state-dim 46 \
    --state-desc "legs 12, waist 3 (yaw, roll, pitch), both arms 7+7, both grippers 1+1, base_gravity 3, both FK EEF 6+6." \
    --action-desc "$ACTION_DESC" \
    --extra-note "$(cat "$NOTE")" \
    ${CKPT:+--checkpoint "$CKPT"} \
    ${DRY_RUN:+--dry-run} \
    >> "$LOG" 2>&1 && say "upload done" || say "UPLOAD FAILED (see $LOG)"
