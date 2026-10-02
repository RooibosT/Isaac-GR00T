#!/usr/bin/env bash
# The four-set merge with the re-collection's grasp COMMITMENT cut out, 68-dim state.
#
# Built by `/root/02_hub/datasets/make_alltask4b.py`: everything in `stage1_2_add_subtask`'s
# `stage1.2` / `stage2.3` up to one second past the left gripper closing on the base rail is
# dropped (30,592 frames), its 1.1 / 2.1 / 2.2 segments and the insert from grasp+1 s on are
# kept (74,699 frames). 581 episodes / 672,236 frames / 649,577 H40 windows, the same nine
# instructions, and the val is the three-set merge's 42 episodes byte for byte -- so this scan
# sits beside `_alltask4_allvel` and `_alltask3nt_allvel` directly.
#
# Why cut exactly there (measured 2026-09-25, after the robot grabbed the wrong part of the
# board with the left hand in 1.2 / 2.3 while 1.1 / 2.2 improved):
#   * the two sessions grip the base 6.7 cm apart (left-arm q0 17.0 deg = 1.96x the old
#     session's IQR), so the plain merge teaches one instruction two grasp targets;
#   * both are learned well -- the merged model reads 12.8 mm against old-session ground truth
#     and 7.2 mm against the re-collection's, while the model trained without it reaches for
#     the old target even on new-session frames -- so the failure is a confident choice of the
#     wrong attractor, not a blurred average;
#   * after the grasp the state alone separates the sessions (best-joint AUC 0.98-1.00), and
#     there the merged model is the more accurate of the two (1.2 arm8 1.414 -> 1.336, 2.3
#     2.042 -> 1.900), which is why the insert is kept rather than dropped with the approach.
#
# 105,000 steps = 10.34 epochs of this split, matching `_alltask4_allvel`'s 10.31 by EPOCHS,
# not by step count -- the window count changed (see EXPERIMENTS.md on the epoch rule).
#
# ⚠️ Same two data guarantees as the A run, both asserted in the pre-flight below:
# `base_cmd_vel` keeps stage3_rc's band (the merge collapsed q01 to 0 again and
# `fix_alltask_basevel.py --step patch` restored -0.2), and every state block the 68-dim
# config reads has a non-degenerate q01/q99 in both splits.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/s12new_addb_flip31_s3rc
CFG=$EX/g1_dex1_ikea_absarm_allvel_basevel_config.py
EXP=g1_dex1_ikea_relarm_3view_aug_b64_alltask4b_allvel
LOG=$ROOT/datasets/chain_alltask4b_allvel.log
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

say "launching 68-dim ABS + arm/gripper/base velocities, 19-dim action on the FOUR-set merge (105k steps = 10.3 epochs, 4 GPUs)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}" \
NUM_GPUS=4 \
DATALOADER_NUM_WORKERS=6 \
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_alltask4b_allvel \
MAX_STEPS=105000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=10 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29699 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
# scan_ikea.py scores no part of base_cmd_vel (mae_arm reads *_arm keys, ee_mm is FK on the
# arm block, and the command is 1/19 of mse with two dead dims pulling it down), so the
# stage-3 drive needs probe_base_cmd_vel.py on the selected checkpoint.
say "reminder: score the base command with probe_base_cmd_vel.py -- the scan cannot see it"
