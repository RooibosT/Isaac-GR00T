#!/usr/bin/env bash
# One 19-dim ABS model for all three current task sets: stage 1+2, the flip, and stage 3.
#
# `merge_ikea_tasksets.py --sets stage1_2_new IKEA_fliptable_v31 IKEA_stage3_rc` plus `stage1_2_add_subtask` (make_alltask4_add.py)
# 439 eps / 597,537 frames / 580,416 H40 windows, nine instructions, indices 0-6 kept from
# stage1_2_new so a client built for it keeps working; 7 is `flip the table`, 8 is
# `move to the stage 3 location, and rotate the table base`. Shares of the windows:
# 74.2% / 20.1% / 5.6%. The pair to chain_alltask3_basevel.sh, which used stage1_2_new_tt05
# (the same set plus target_thread stage0.5) -- the val is byte-identical, so the two scans
# sit side by side and the difference is the 25,552 stage0.5 windows alone.
#
# **19-dim action** (`g1_dex1_ikea_absarm_basevel_config.py`): arms 7+7, grippers 1+1 and
# `base_cmd_vel` 3. stage3_rc is the one set where the robot drives, so dropping that block
# would train everything except the thing stage 3 was recorded for. Deployment must unpack
# 19 dims, not 16.
#
# `fix_alltask_basevel.py` has already done two things to that block, and the run refuses
# to start unless both are still in place:
#   * cleared a **stale** command in 31 stage1_2_new episodes that held
#     `+0.1000 / +0.1800 / 0` for every frame while the base measurably did not move
#     (|base_lin_vel| 0.0054 against 0.0043 idle). Trained, it would drive the base through
#     table assembly.
#   * patched the merged `stats.json` so `base_cmd_vel` keeps stage3_rc's own q01/q99.
#     Merging drops the real command's share from 8.8% of frames to 0.52%, which puts q01
#     and q99 both on 0; with percentile normalisation and clipping, commanded and idle
#     frames would both map to -1 and the drive would vanish silently.
#
# 110,000 steps = 10.3 epochs. Six runs put their best at 55-75% of the schedule and the
# 68-dim run was flat within +0-3.4% from 6.9 epochs on, so 12.9 epochs spent its last
# third for nothing; 10.3 keeps the useful band (6.5-9 epochs) at 63-87% of the run. In the 9-15 band every run in this project peaks inside.
#
# **4 GPUs here, unlike the A/B pair.** Those two had to replay one sample stream, because
# shards are dealt `rank * workers + worker` and only the same GPU count reproduces it.
# This run has no counterpart to match, so it takes the measured 1.43x: 0.340 s/step at
# micro 16 against 0.486 at micro 32 on two (workers 6 and 12 measured identical).
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
HUB=/root/02_hub/datasets
DS=$HUB/s12new_add_flip31_s3rc
CFG=$EX/g1_dex1_ikea_absarm_basevel_config.py
LOG=$ROOT/datasets/chain_alltask4_basevel.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

"$ROOT/.venv/bin/python" - "$DS" <<'PY' || { say "pre-flight FAILED"; exit 1; }
import json, sys
from pathlib import Path
ds = Path(sys.argv[1])
for split in ("train", "val"):
    d = ds.with_name(ds.name + f"_{split}")
    st = json.loads((d / "meta/stats.json").read_text())["action"]
    if abs(st["q01"][17] + 0.2) > 1e-6:
        raise SystemExit(f"{d.name}: base_cmd_vel q01[17] is {st['q01'][17]}, not -0.2 -- "
                         "run fix_alltask_basevel.py --step patch")
print("pre-flight ok: base_cmd_vel keeps stage3_rc's band in both splits")
PY
say "pre-flight ok"

say "launching 46-dim ABS, 19-dim action on the FOUR-set merge (110k steps = 10.3 epochs, 4 GPUs)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}" \
NUM_GPUS=4 \
DATALOADER_NUM_WORKERS=6 \
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_alltask4_basevel \
MAX_STEPS=110000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=10 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29697 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 >> "$LOG" 2>&1
say "chain complete (exit $?)"
# scan_ikea.py scores no part of base_cmd_vel (mae_arm reads *_arm keys, ee_mm is FK on the
# arm block, and the command is 1/19 of mse with two dead dims pulling it down), so the
# stage-3 drive needs probe_base_cmd_vel.py on the selected checkpoint.
say "reminder: score the base command with probe_base_cmd_vel.py -- the scan cannot see it"
