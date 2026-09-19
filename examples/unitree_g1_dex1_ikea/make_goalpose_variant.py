# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add a 9D "where this phase ends" pose to the action, as auxiliary supervision.

The robot places the leg at poses that are inside the demonstrated band and still
misses the hole, because the table moves between recordings and **nothing in the
state says where it is** -- only the cameras do. Measured on the stage1_v2 val
with the 46-dim ABS checkpoint, the policy already aims at *this* episode's hole
rather than the average one (11.6-14.1 mm against 55.8 mm for a mean-aimer at 45
to 75 frames out), so the lever is precision, not whether vision is used at all.

This is the loss that asks for that precision directly. For every frame the extra
block holds the right wrist pose at the **next right-gripper transition**: during
the carry and insert that is the pose the leg goes in at, during the pick it is
the grasp, and inside a rotation cycle the next transition is nearly where the
hand already is, so the target costs almost nothing there. The phase where the
signal matters is therefore the phase where the target is hard, without a
per-frame loss mask -- `action_mask` is built from shapes inside the processor
and carries nothing from the dataset, so masking would be a core change.

Why this is worth a run when torque, history and a gripper-derived phase were
not (sections 31, 34): those were all information the observation already
carried, and the model could ignore them at no cost. A goal pose cannot be
recovered from proprioception at all, so the gradient has nowhere to go but
through the vision path.

It is not the RAMEN-style EEF auxiliary (`make_eef_action_variant.py`). That one
predicts EEF = FK(joints) alongside the joints -- the same trajectory in another
frame, redundant by construction. This predicts a point in the future that only
the scene determines.

The target is read off the **measured** state, not the commanded action: while
the leg is pressed against the hole the command sits 20-48 mm below the wrist,
and the place to localise is where the leg physically goes.

ABSOLUTE and appended to the action only. A RELATIVE EEF block would need a
matching 9D reference block in the state (`_convert_to_relative_action` asserts
it); an absolute one is normalised per dimension and needs nothing.

Usage:
    python make_goalpose_variant.py [--src-prefix ...] [--dst-prefix ...]
"""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BLOCK = ("right_goal_9d", 9)
# The scan reads `*_eef_9d` as a wrist-pose action block; this name deliberately
# does not match, so the goal block stays out of its EEF metric.


def _helpers():
    spec = importlib.util.spec_from_file_location("eefvar", HERE / "make_eef_action_variant.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def transitions(grip: np.ndarray) -> list[int]:
    """Frames where the right gripper command changes state, hysteresis on its own range."""
    lo, hi = np.quantile(grip, 0.05), np.quantile(grip, 0.95)
    if hi - lo < 1e-3:
        return []
    closed, opened = lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo)
    state, out = None, []
    for i, v in enumerate(grip):
        new = "C" if v <= closed else ("O" if v >= opened else state)
        if new != state and new is not None:
            if state is not None:
                out.append(i)
            state = new
    return out


def convert_split(src: Path, dst: Path, fk, euler_to_rot6d) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    mod = json.loads((src / "meta/modality.json").read_text())
    a_width = max(v["end"] for v in mod["action"].values())
    s_blk = {k: (v["start"], v["end"]) for k, v in mod["state"].items()}
    a_blk = {k: (v["start"], v["end"]) for k, v in mod["action"].items()}

    for f in ("episodes.jsonl", "tasks.jsonl", "info.json"):
        shutil.copy2(src / "meta" / f, dst / "meta" / f)
    name, w = BLOCK
    mod["action"][name] = {"start": a_width, "end": a_width + w}
    (dst / "meta/modality.json").write_text(json.dumps(mod, indent=4))

    info = json.loads((dst / "meta/info.json").read_text())
    if "action" in info.get("features", {}):
        info["features"]["action"]["shape"] = [a_width + w]
        info["features"]["action"].pop("names", None)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))

    for vid in sorted((src / "videos").rglob("*.mp4")):
        out = dst / vid.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.hardlink_to(vid)

    rg_lo, _ = a_blk["right_gripper"]
    r_arm = s_blk["right_arm"]
    n_ep = n_row = 0
    lead, counts = [], []
    for pq in sorted(src.glob("data/chunk-*/episode_*.parquet")):
        d = pd.read_parquet(pq)
        S = np.stack(d["observation.state"]).astype(np.float64)
        A = np.stack(d["action"]).astype(np.float64)
        n = len(d)

        ev = transitions(A[:, rg_lo])
        counts.append(len(ev))
        # the pose of the right wrist at each transition, measured
        poses = {t: fk.wrist_pose("right", S[t, r_arm[0] : r_arm[1]]) for t in ev}
        end_pose = fk.wrist_pose("right", S[n - 1, r_arm[0] : r_arm[1]])

        goal = np.empty((n, 6))
        nxt = 0
        for t in range(n):
            while nxt < len(ev) and ev[nxt] <= t:
                nxt += 1
            goal[t] = poses[ev[nxt]] if nxt < len(ev) else end_pose
        block = np.concatenate([goal[:, :3], euler_to_rot6d(goal[:, 3:])], axis=1)

        here = np.array([fk.wrist_pose("right", S[t, r_arm[0] : r_arm[1]])[:3] for t in range(n)])
        lead.append(float(np.median(np.linalg.norm(goal[:, :3] - here, axis=1))))

        d["action"] = list(np.concatenate([A, block], axis=1).astype(np.float32))
        out = dst / pq.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(out, index=False)
        n_ep += 1
        n_row += len(d)

    print(f"  {dst.name}: {n_ep} ep / {n_row} frame, action {a_width}->{a_width + w}")
    print(f"    gripper transitions per episode: median {np.median(counts):.0f}")
    print(f"    distance from the wrist to its goal: median {np.median(lead) * 1000:.0f} mm")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-prefix", default="/root/02_hub/datasets/IKEA_pick_leg_stage1_v2")
    ap.add_argument("--dst-prefix", default="/root/02_hub/datasets/IKEA_pick_leg_stage1_v2_goal")
    a = ap.parse_args()
    h = _helpers()
    fk = h._load_fk()
    for split in ("_train", "_val"):
        convert_split(Path(a.src_prefix + split), Path(a.dst_prefix + split), fk, h._euler_to_rot6d)
    print("DONE")


if __name__ == "__main__":
    main()
