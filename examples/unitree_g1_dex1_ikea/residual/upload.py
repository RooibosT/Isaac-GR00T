# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Push the correction heads, with the module needed to load them.

One repo holds the whole sweep rather than one repo per bound: the three differ
only in `clip`, which is the dial between a residual that nudges and a takeover
that replaces, and the point of shipping them together is that the robot picks.
`model.py` goes with them because a state_dict alone cannot be loaded.

Usage:
    python upload.py --repo RooibosT/... --heads /root/02_hub/residual_cache
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi
import torch


CARD = """---
license: apache-2.0
tags:
- robotics
- gr00t
---

# IKEA stage-1 — DAgger correction head on the deployed baseline

Not a policy. A small head that is **added to** the chunk of
`RooibosT/gr00t-n1.7-g1-dex1-ikea-stage1-absarm-subtask-30hz-h40`
(`checkpoint-37500`, the deploy baseline):

    a_deploy = a_base(o) + delta(o)

`delta = 0` is exactly that baseline, which two full retrains on the same
intervention data did not preserve: both scored 1-3% better on the offline scan
and worse on the robot, releasing the leg without pressing or adjusting.

## Training

`a_human - a_base(o)` on the human-teleop segments of
`RooibosT/ikea_stage1_subtask_intervention_full`, and exactly zero on the policy
segments and on `RooibosT/ikea_stage1_subtask_deploy_success`. Zero rather than
`a_policy - a_base`, because the recorded action on a policy segment is what the
deploy EXECUTED after RTC and the velocity clamp, and fitting it would teach the
head to reproduce clamping the deploy then applies again. `a_base` is the mean of
8 draws, since the baseline's head is flow matching and one draw's sampling noise
would enter every target as if it were a correction.

`hold` segments are excluded: the arm is parked on a fixed target while control
changes hands and the action is constant.

15,454 frames, 15.9% of them corrections, split by SOURCE EPISODE with 7 rescues
held out. 4.4M parameters; the last layer is zero-initialised, so an untrained
head is exactly the baseline.

## The three bounds

The corrections in this data need 0.59-0.90 rad on the rows the deploy executes,
and the gap GROWS with time since the takeover (0.589 rad in the first second,
0.895 by the fifth). The human is not adjusting the policy's plan, they are
driving a different trajectory. So `clip` is not a safety margin, it is the dial
between a residual that can only nudge and a takeover policy with a learned gate.

| file | held-out leave-alone \\|delta\\| p50 / p90 | correction cosine p50 / >0 | delivered on rows 6-22 |
| --- | --- | --- | --- |
{TABLE}

The gate holds at every bound: even with 0.8 rad of authority the head stays at
0.011 rad on held-out leave-alone frames, a tenth of the deploy's own per-tick
clamp.

## What is NOT established

The offline scan has called two DAgger runs an improvement and the robot
disagreed both times, so nothing here is evidence of a robot gain. The validation
above measures that the head is quiet where it should be and points where the
human pointed on rescues it has not seen. Which bound to run, and whether any of
them helps, is a question for the robot.

## Loading

```python
from model import ResidualHead          # this repo
import torch
blob = torch.load("head_clip0.8.pt", map_location="cpu")
head = ResidualHead(**blob["config"])
head.load_state_dict(blob["state_dict"])
```

Inputs are the baseline's own frozen backbone tokens, the 46-dim state as the
baseline receives it, and the baseline's chunk. Output is `(40, 16)` to add to
that chunk before the deploy's existing validate and clamp path.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--rows", default="", help="markdown rows for the results table")
    a = ap.parse_args()

    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    files = sorted(Path(a.heads).glob("head_clip*.pt"))
    if not files:
        raise SystemExit(f"no head_clip*.pt under {a.heads}")
    for f in files:
        blob = torch.load(f, map_location="cpu")
        print(
            f"{f.name}: clip {blob['config']['clip']}, "
            f"{sum(v.numel() for v in blob['state_dict'].values()) / 1e6:.1f}M params"
        )
        api.upload_file(
            path_or_fileobj=str(f), path_in_repo=f.name, repo_id=a.repo, repo_type="model"
        )
    for extra in ("model.py", "README.md" if False else "dataset.py", "train.py", "cache.py"):
        api.upload_file(
            path_or_fileobj=str(here / extra), path_in_repo=extra, repo_id=a.repo, repo_type="model"
        )
    card = CARD.replace("{TABLE}", a.rows or "| (see the training log) | | | |")
    api.upload_file(
        path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=a.repo, repo_type="model"
    )
    print(f"UPLOAD_DONE {a.repo}")


if __name__ == "__main__":
    main()
