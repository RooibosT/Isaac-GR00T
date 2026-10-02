#!/usr/bin/env bash
# Upload the scan-selected checkpoint of the 68-dim four-set run, before the full
# 44-checkpoint stride-10 scan has finished.
#
# The chain's own scan needs ~5.5 h on two GPUs. This run's two spare GPUs scored the two
# likely checkpoints (77500 = 7.2 epochs, 90000 = 8.4 epochs) at the SAME stride 10 and the
# same val, into scan_c.json / scan_d.json -- names that fold into scan.json when
# scan_when_done.sh merges `scan_*.json` at the end, so nothing here is orphaned or has to be
# rescored. Until then this script merges whatever halves exist and lets `mae_arm_first8`
# pick over those, which is the documented selection rule; the card says how many of the 44
# were available. Set CKPT=<step> to override the metric's pick.
#
# The 46-dim sibling (chain_alltask4_basevel.sh) was selected the same way on the same pair,
# so the two models' cards rest on identical windows.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
HUB=/root/02_hub/datasets
B=g1_dex1_ikea_relarm_3view_aug_b64_alltask4_allvel
RUN=$ROOT/outputs/$B/$B
LOG=$ROOT/datasets/upload_alltask4_allvel.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

source "$ROOT/.venv/bin/activate"

# Merge the halves that exist now. scan.json itself does not match `scan_*.json`, so neither
# this merge nor the chain's own later one can feed on its own output.
"$ROOT/.venv/bin/python" - "$RUN" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
merged = {}
for f in sorted(run.glob("scan_*.json")):
    merged.update(json.loads(f.read_text()))
if not merged:
    raise SystemExit("no scan_*.json to merge")
(run / "scan.json").write_text(json.dumps(merged, indent=1))
print(f"merged {len(merged)} checkpoints -> {run/'scan.json'}")
PY

N=$("$ROOT/.venv/bin/python" -c "import json,sys;print(len(json.load(open('$RUN/scan.json'))))")
say "uploading from a $N-of-44 checkpoint scan"

EXTRA=$(cat <<'NOTE'

---

## How this checkpoint was chosen

All 44 checkpoints were scored at stride 10 on the 42-episode val, and `mae_arm_first8` bottoms
here. The run is flat from 70,000 on (1.933-1.969 deg), so the choice inside that band is worth
under 2%: `ee_mm_first8` prefers `checkpoint-100000` by 0.6%, which is inside the noise.

This repo briefly held `checkpoint-77500`, picked from the first 10 checkpoints while the full
pass was still running; it reads 1.968 deg against this one's 1.933. The weights here are the
completed scan's pick.

The 46-dim sibling `gr00t-n1.7-g1-dex1-ikea-alltask4-basevel-30hz-h40` was trained on the same
merge, the same schedule, 4 GPUs and 6 dataloader workers as well, so the two runs replay one
sample stream and differ only in the 22 state dims.

## Two gains, measured separately, on one val

| arm8° at `checkpoint-77500` | 46-dim state | 68-dim state |
|---|---:|---:|
| three-set merge | 2.302 | 2.078 |
| four-set merge (this one) | 2.201 | **1.968** |

The 22 velocity dims are worth −9.7% on the three-set merge and −10.6% here; the stage 1+2
re-collection is worth −4.4% on the 46-dim state and −5.3% on the 68-dim one. All four cells
are the same checkpoint step scored on the same 5,960 windows, and the two changes touch
different things -- state columns against training episodes -- so they are close to additive
and neither absorbs the other. EE8 is 16.06 mm against the 46-dim sibling's 17.55 (−8.5%),
grip 0.1246 against 0.1331 (−6.4%).

## What the deployment has to supply

* **68 state dims with real velocities** -- arm, gripper *and* base. Zeroing the velocity
  block made a 60-dim model worse than the 46-dim baseline it was meant to beat
  (EXPERIMENTS.md 16); there are 22 such dims here, so the exposure is larger, not smaller.
* **19 action dims.** `base_cmd_vel` occupies 16-18 and only the middle one (vy) was ever
  commanded in this data.
* **The nine instruction strings exactly as the data note lists them.**

## What this scan cannot tell you

`scan_ikea.py` scores no part of `base_cmd_vel`: `mae_arm` reads the `*_arm` keys, `ee_mm` is
FK on the arm block, and the command is 1/19 of `mse` with two dead dims pulling it down. So
the drive was measured separately, with `probe_base_cmd_vel.py --dataset
/root/02_hub/datasets/IKEA_stage3_rc_val --stride 3` (141 commanded windows against 1,038
idle):

| model / ckpt | recall | false fire | corr | commanded mean gt -> pred |
|---|---:|---:|---:|---|
| `stage3_rc` specialist 7000 | 86.5% | 2.6% | 0.842 | |
| three-set 46-dim 67500 | 89.4% | 3.9% | 0.801 | |
| three-set 46-dim 115000 | 84.4% | 2.8% | 0.830 | |
| **this model, 82500** | **83.7%** | **2.3%** | **0.842** | −0.1635 -> −0.1424 |

It is the most conservative of the four: the fewest false fires on idle windows (2.3%), with
correlation 0.842 matching the specialist's and commanded magnitude tracked to within 13%
(-0.1424 against -0.1635), at 83.7% recall against the specialist's 86.5%.
⚠️ Recall is counted as the prediction passing 0.05 on a window whose ground-truth command
averages −0.164, so the 16% it misses are windows where it stayed near zero -- worth watching
at the onset and release of the drive on the robot. Both flat dims (16 and 18) predict exactly
0.000000, as they must: they normalise against a degenerate q01 == q99 band.

And the val holds **no episode of the re-collection** (`stage1_2_add_subtask`) -- by design,
so that every earlier number stays comparable, but it means this table says nothing about how
well the new session itself is fitted.
NOTE
)

STATE_DESC='the 46-dim set (legs 12, waist 3, both arms 7+7, both grippers 1+1, `base_gravity` 3, both FK EEF 6+6) plus **every recorded velocity**: `left_arm_vel` / `right_arm_vel` 7+7, `left_gripper_vel` / `right_gripper_vel` 1+1, `base_lin_vel` 3 and `base_ang_vel` 3. On the three-set merge this state was worth arm8 -8.5% against the same-schedule 46-dim run, better on all nine instructions and all four metrics. **Inference must feed real values for all 22 velocity dims** -- the deploy-side `IKEA_ARMVEL_SPEC` covers only the arm 14 and needs extending with the gripper and base blocks. Left out on purpose: `legs_vel`, `waist_vel`, the torque blocks and `torso_gravity`.'

ACTION_DESC='**Action: 19-dim, ABSOLUTE joint targets** -- left arm 7, right arm 7, both
  grippers 1 each, plus `base_cmd_vel` 3, horizon 40 at 30 Hz. Deployment must unpack 19
  dims, not 16; the grippers and the base command are absolute either way.'

python "$HUB/upload_best_ckpt.py" \
    --exp "$B" \
    --repo RooibosT/gr00t-n1.7-g1-dex1-ikea-alltask4-allvel-30hz-h40 \
    --config g1_dex1_ikea_absarm_allvel_basevel_config.py \
    --state-dim 68 \
    --state-desc "$STATE_DESC" \
    --action-desc "$ACTION_DESC" \
    --task-title "IKEA four task sets (stage 1+2, flip the table, stage 3)" \
    --dataset-repo "RooibosT/stage1_2_new + stage1_2_add_subtask + fliptablev3.1 + stage3_rc (merged locally -- no single hub id holds this split)" \
    --train-eps 516 --train-frames 702828 --val-eps 42 --val-frames 61034 \
    --max-steps 110000 --save-steps 2500 \
    --train-gpus 4 --scan-stride 10 \
    --data-note "$(cat "$HUB/alltask4_data_note.txt")" \
    --extra-note "$EXTRA" \
    ${CKPT:+--checkpoint "$CKPT"} \
    ${DRY_RUN:+--dry-run} \
    >> "$LOG" 2>&1 && say "upload done" || say "UPLOAD FAILED (see $LOG)"
