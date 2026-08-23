# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite the IKEA action vector so it lines up with the BCT model's output slots.

Warm-starting from the BCT fine-tune only transfers the action head if the two
runs agree on what each output slot *means*. BCT's config emits

    [ waist(3) | left_arm(7) | right_arm(7) | left_gripper(1) | right_gripper(1) ]  = 19

while the IKEA config emits the same list without the waist, so every arm slot is
shifted by three: BCT's waist weights would land on the left shoulder. This
builds a variant whose action block carries waist in front, restoring the
alignment.

`teleop_ikea.py` never commands the waist — it holds the startup pose (0.09-0.45
deg of drift within an episode) — so the waist "action" here is that row's
measured waist. That is the same kind of quantity BCT had: its waist action was
shown to be a locomotion-controller output, not a teleop command (corr 0.84-0.86
with the leg joints). Deployment ignores the waist output in both cases.

Everything else is untouched: `observation.state` is copied verbatim, videos are
hardlinked, and only meta/modality.json's action block changes. Stats must be
regenerated afterwards because the action width changes 19 -> 19 but its
*content* does not match the old stats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


# BCT's key order — the whole point of this variant
ACTION_BLOCKS = [
    ("waist", 3),
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_gripper", 1),
    ("right_gripper", 1),
]


def convert_split(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    mod = json.loads((src / "meta/modality.json").read_text())
    s_blk = {k: (v["start"], v["end"]) for k, v in mod["state"].items()}
    a_blk = {k: (v["start"], v["end"]) for k, v in mod["action"].items()}

    for f in ("episodes.jsonl", "tasks.jsonl", "info.json"):
        shutil.copy2(src / "meta" / f, dst / "meta" / f)

    new_action, start = {}, 0
    for name, width in ACTION_BLOCKS:
        new_action[name] = {"start": start, "end": start + width}
        start += width
    mod["action"] = new_action
    (dst / "meta/modality.json").write_text(json.dumps(mod, indent=4))

    info = json.loads((dst / "meta/info.json").read_text())
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [start],
        "names": [[f"{n}_{i}" if w > 1 else n for n, w in ACTION_BLOCKS for i in range(w)]],
    }
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))

    for vid in sorted((src / "videos").rglob("*.mp4")):
        out = dst / vid.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.hardlink_to(vid)

    n = 0
    for pq in sorted(src.glob("data/chunk-*/episode_*.parquet")):
        d = pd.read_parquet(pq)
        S = np.stack(d["observation.state"]).astype(np.float32)
        A = np.stack(d["action"]).astype(np.float32)
        cols = []
        for name, _ in ACTION_BLOCKS:
            if name == "waist":  # not commanded — take the measured pose
                lo, hi = s_blk["waist"]
                cols.append(S[:, lo:hi])
            else:
                lo, hi = a_blk[name]
                cols.append(A[:, lo:hi])
        d["action"] = list(np.concatenate(cols, axis=1).astype(np.float32))
        out = dst / pq.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(out, index=False)
        n += 1
    print(f"  {dst.name}: {n} episodes, action width {start}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-prefix", required=True, type=Path)
    ap.add_argument("--dst-prefix", required=True, type=Path)
    args = ap.parse_args()
    for split in ("train", "val"):
        convert_split(
            args.src_prefix.with_name(args.src_prefix.name + f"_{split}"),
            args.dst_prefix.with_name(args.dst_prefix.name + f"_{split}"),
        )
    print("DONE")


if __name__ == "__main__":
    main()
