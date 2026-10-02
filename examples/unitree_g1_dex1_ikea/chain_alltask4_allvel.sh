#!/usr/bin/env bash
# The 68-dim counterpart to chain_alltask4_basevel.sh: every recorded velocity in the state,
# on the FOUR-set merge.
#
# `merge_ikea_tasksets.py --sets stage1_2_new IKEA_fliptable_v31 IKEA_stage3_rc` plus
# `stage1_2_add_subtask` (make_alltask4_add.py) -- 516 eps / 702,828 frames / 682,704 H40
# windows, nine instructions, indices 0-6 from stage1_2_new so a client built for it keeps
# working; 7 is `flip the table`, 8 is `move to the stage 3 location, and rotate the table
# base`. Shares of the windows: 78.1% stage1+2 / 17.1% flip / 4.8% stage 3. The +77 episodes
# over the three-set merge are stage1_2_add_subtask, with its two orient strings rewritten to
# the merge's existing ones so the instruction set stays at nine.
#
# **68-dim state** (`g1_dex1_ikea_absarm_allvel_basevel_config.py`): the 46-dim state plus arm
# velocities 7+7, gripper velocities 1+1 and base linear/angular velocity 3+3. On the
# three-set merge this was worth **arm8 -8.5%** over the same-schedule 46-dim run, better on
# all nine tasks and all four metrics (`alltask3nt_allvel` against `alltask3nt_basevel`), and
# it beat all three specialists by 14-33%. This run repeats that A/B one set later.
#
# **19-dim action**: arms 7+7, grippers 1+1 and `base_cmd_vel` 3. stage3_rc is the one set
# where the robot drives. Deployment must unpack 19 dims, not 16.
#
# All 15 state blocks this config reads have non-degenerate q01/q99 in **both** splits,
# checked before launch and re-checked in the pre-flight below -- the velocity blocks are the
# ones at risk, since three of the four sets are stationary work (base_lin_vel spans
# 0.045-0.085 in train, base_ang_vel 0.051-0.132, so they are small but alive).
#
# ⚠️ **Do not regenerate `meta/stats.json` on this dataset.** `fix_alltask_basevel.py --step
# patch` has already given `base_cmd_vel` stage3_rc's own q01/q99; a plain
# `gr00t.data.stats` run overwrites the patch, the merged band collapses to 0/0, commanded
# and idle frames both normalise to -1 and the stage-3 drive vanishes silently. The pre-flight
# refuses to start when q01[17] != -0.2. `run_finetune_ikea.sh` only generates stats when the
# file is missing, and it is not.
#
# 110,000 steps = 10.3 epochs, **identical to chain_alltask4_basevel.sh**, and the val is the
# same 42 episodes (byte-identical `episodes.jsonl` to the three-set merges), so the two scans
# sit side by side and the only difference between the runs is the 22 state dims. Same reason
# for 4 GPUs and 6 workers: shards are dealt `rank * workers + worker`, so only the same GPU
# and worker count replays the sibling's sample stream. Do not "optimise" either number here.
#
# ⚠️ `/root` is CephFS shared with the host running the 46-dim sibling (main-wsp-9imo1jwx1krb,
# MASTER_PORT 29697, EXP_SUFFIX _alltask4_basevel, started 16:26 UTC 2026-09-24). This run
# takes its own suffix, log and port, and writes ~540 GB of checkpoints into the same volume.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/s12new_add_flip31_s3rc
CFG=$EX/g1_dex1_ikea_absarm_allvel_basevel_config.py
EXP=g1_dex1_ikea_relarm_3view_aug_b64_alltask4_allvel
LOG=$ROOT/datasets/chain_alltask4_allvel.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# An earlier attempt at this run on the sibling's host was killed at step 4,506 (15:51-16:26
# UTC) and its output directory removed; nothing may be left behind to train into.
if compgen -G "$ROOT/outputs/$EXP/checkpoint-*" > /dev/null; then
    say "REFUSING: $ROOT/outputs/$EXP already holds checkpoints -- move them aside first"
    exit 1
fi

"$ROOT/.venv/bin/python" - "$DS" <<'PY' || { say "pre-flight FAILED"; exit 1; }
import json, sys
from pathlib import Path

# The 15 state blocks g1_dex1_ikea_absarm_allvel_basevel_config reads, by column.
BLOCKS = {
    "legs": (0, 12), "waist": (12, 15), "left_arm": (15, 22), "right_arm": (22, 29),
    "left_arm_vel": (46, 53), "right_arm_vel": (53, 60),
    "left_gripper": (29, 30), "right_gripper": (30, 31),
    "left_gripper_vel": (60, 61), "right_gripper_vel": (61, 62),
    "base_lin_vel": (62, 65), "base_ang_vel": (65, 68),
    "base_gravity": (68, 71), "left_eef": (74, 80), "right_eef": (80, 86),
}
ds = Path(sys.argv[1])
for split in ("train", "val"):
    d = ds.with_name(ds.name + f"_{split}")
    stats = json.loads((d / "meta/stats.json").read_text())
    act = stats["action"]
    if abs(act["q01"][17] + 0.2) > 1e-6:
        raise SystemExit(f"{d.name}: base_cmd_vel q01[17] is {act['q01'][17]}, not -0.2 -- "
                         "run fix_alltask_basevel.py --step patch")
    st = stats["observation.state"]
    dead = [k for k, (a, b) in BLOCKS.items()
            if min(st["q99"][i] - st["q01"][i] for i in range(a, b)) <= 1e-9]
    if dead:
        raise SystemExit(f"{d.name}: degenerate q01/q99 in {dead} -- those dims would "
                         "normalise to a constant")
print("pre-flight ok: base_cmd_vel keeps stage3_rc's band and all 15 state blocks are alive, "
      "in both splits")
PY
say "pre-flight ok"

say "launching 68-dim ABS + arm/gripper/base velocities, 19-dim action on the FOUR-set merge (110k steps = 10.3 epochs, 4 GPUs)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}" \
NUM_GPUS=4 \
DATALOADER_NUM_WORKERS=6 \
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_alltask4_allvel \
MAX_STEPS=110000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=10 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29698 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
# scan_ikea.py scores no part of base_cmd_vel (mae_arm reads *_arm keys, ee_mm is FK on the
# arm block, and the command is 1/19 of mse with two dead dims pulling it down), so the
# stage-3 drive needs probe_base_cmd_vel.py on the selected checkpoint.
say "reminder: score the base command with probe_base_cmd_vel.py -- the scan cannot see it"
