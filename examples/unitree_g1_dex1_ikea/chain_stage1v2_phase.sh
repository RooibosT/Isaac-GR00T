#!/usr/bin/env bash
# stage1_v2, 47-dim: the 46-dim ABSOLUTE config plus a subtask phase bit.
#
# The leg is turned four times before the base is aligned, and after two turns
# and after four the robot's state is the same. The count is not in the
# observation at all, so on the robot the policy keeps turning and never moves
# on. An external module counts the turns; this run puts its output in the state
# rather than splitting the instruction, so the policy stays single and there is
# no transition between two half-trained behaviours.
#
# `--state-dropout-keep-keys phase` matters. Dropping the one input that
# disambiguates two behaviours, on 20% of samples, while still demanding the
# right action teaches the model to hedge on exactly the signal it is given.
# Setting it also turns off the action head's embedding-level dropout, which
# cannot be selective -- see the config field's note. The dropped condition
# becomes "no state except phase" rather than "no state", and the effective
# hide-rate goes from 36% (two independent draws at 0.2) to 20%.
#
# Same train/val as the instruction-split dataset -- 158/217,430 and 9/10,604 --
# so the two approaches can be compared directly. Nine of the ten recordings the
# other stage1_v2 runs used survive here; v2 episode 118 is not in this export.
#
# ⚠️ The scan cannot judge this. Fed demonstration observations, a policy that
# would turn forever still scores well, and the transition is one moment per
# episode inside a phase worth 8.7% of frames. The scan is for checkpoint
# selection; the verdict is the robot.
set -uo pipefail
ROOT=/root/01_IKEA/Isaac-GR00T
EX=$ROOT/examples/unitree_g1_dex1_ikea
DS=/root/02_hub/datasets/IKEA_pick_leg_stage1_v2_phase
CFG=$EX/g1_dex1_ikea_absarm_phase_config.py
LOG=$ROOT/datasets/chain_stage1v2_phase.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

source "$ROOT/.venv/bin/activate"
for D in "${DS}_train" "${DS}_val"; do
    if [ ! -f "$D/meta/stats.json" ]; then
        say "stats for $D"
        python -m gr00t.data.stats --dataset-path "$D" \
            --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" \
            >> "$ROOT/datasets/stats_stage1v2_phase.log" 2>&1 || { say "stats FAILED for $D"; exit 1; }
    fi
done
say "stats ready"

# The phase only helps if it survives normalisation as a full-range signal.
python - "$DS" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
s = json.loads((Path(sys.argv[1] + "_train") / "meta/stats.json").read_text())["observation.state"]
i = json.loads((Path(sys.argv[1] + "_train") / "meta/modality.json").read_text())["state"]["phase"]["start"]
lo, hi = s["q01"][i], s["q99"][i]
print(f"phase column {i}: q01={lo} q99={hi} min={s['min'][i]} max={s['max'][i]}")
assert hi > lo, "phase q01 == q99; normalisation would divide by zero"
print(f"  0 -> {2 * (0 - lo) / (hi - lo) - 1:+.3f}   1 -> {2 * (1 - lo) / (hi - lo) - 1:+.3f} (clipped to [-1,1])")
PY

say "launching 47-dim ABS + phase on stage1_v2"
cd "$ROOT"
CONFIG=$CFG \
DATASET_ROOT=$DS \
EXP_SUFFIX=_stage1v2_absarm_phase \
MAX_STEPS=45000 SAVE_STEPS=2500 EVAL_STEPS=2500 \
SCAN_STRIDE=7 SCAN_GPUS="0 1" \
USE_WANDB=1 MASTER_PORT=29677 \
bash "$EX/run_finetune_ikea.sh" --use-ddp --ddp-comm-bf16 \
    --state-dropout-keep-keys phase >> "$LOG" 2>&1
say "phase chain complete ($?)"
