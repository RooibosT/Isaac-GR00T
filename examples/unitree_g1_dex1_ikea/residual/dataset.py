# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble the cached shards into residual-training tensors.

`meta/segments.json` of the intervention export says who was driving during each
LeRobot episode, and segments never span a change of owner, so the filter is
exact. `hold` is dropped: the arm is parked on a fixed target while control
changes hands, its action is constant, and training on it teaches stopping.

The target is `a_human - a_base` on the human segments and exactly zero
everywhere else. Zero rather than `a_policy - a_base`, because the recorded action
on a policy segment is what the deploy EXECUTED -- after RTC and the velocity
clamp -- so fitting it would teach the residual to reproduce clamping that the
deploy then applies a second time. Nothing weights the two groups; their natural
sizes are the 17/83 split the README quotes.
"""

import json
from pathlib import Path

import numpy as np


HOLD = "hold"


def owners(segments_json: Path) -> dict[int, str]:
    """LeRobot episode index -> control owner."""
    blob = json.loads(Path(segments_json).read_text())
    return {int(s["episode_index"]): s["control_owner"] for s in blob["segments"]}


def source_of(segments_json: Path) -> dict[int, str]:
    """LeRobot episode index -> the raw episode it was cut from, for grouped splits."""
    blob = json.loads(Path(segments_json).read_text())
    return {int(s["episode_index"]): s["source_episode"] for s in blob["segments"]}


def load_shard(
    npz: Path, owner_of: dict[int, str] | None, tag: str, source_of_ep: dict[int, str] | None = None
) -> dict:
    d = np.load(npz)
    ep = d["ep"]
    if owner_of is not None:
        own = np.array([owner_of.get(int(e), "unknown") for e in ep])
        keep = own != HOLD
    else:
        own = np.full(len(ep), tag)
        keep = np.ones(len(ep), bool)
    src = (
        np.array([source_of_ep.get(int(e), f"{tag}{e}") for e in ep])
        if source_of_ep is not None
        else np.array([f"{tag}{e}" for e in ep])
    )
    return {
        "has_base": (
            d["has_base"][keep] if "has_base" in d.files else np.ones(int(keep.sum()), bool)
        ),
        "feats": d["feats"][keep],
        "state": d["state"][keep],
        "a_base": d["a_base"][keep],
        "a_gt": d["a_gt"][keep],
        "owner": own[keep],
        "group": src[keep],
        "shard": np.full(int(keep.sum()), tag),
    }


def concat(shards: list[dict]) -> dict:
    return {k: np.concatenate([s[k] for s in shards]) for k in shards[0]}


def build(cache_dir: Path, segments_json: Path) -> dict:
    cache_dir = Path(cache_dir)
    own = owners(segments_json)
    src = source_of(segments_json)
    parts = [load_shard(cache_dir / "intervention_full.npz", own, "full", src)]
    extra = cache_dir / "deploy_success.npz"
    if extra.exists():
        parts.append(load_shard(extra, None, "success"))
    data = concat(parts)
    corr = (data["owner"] == "human") & data["has_base"]
    delta = np.zeros_like(data["a_gt"])
    delta[corr] = data["a_gt"][corr] - data["a_base"][corr]
    data["delta"] = delta
    data["is_correction"] = corr
    return data


def report(data: dict) -> None:
    n = len(data["delta"])
    corr = data["is_correction"]
    print(
        f"{n} frames; correction {corr.sum()} ({100 * corr.mean():.1f}%), "
        f"leave-alone {n - corr.sum()}"
    )
    for name, m in (("correction", corr), ("leave-alone", ~corr)):
        if not m.any():
            continue
        mag = np.abs(data["delta"][m][:, :, :14]).max(axis=(1, 2))
        print(
            f"  {name:>12}: |delta| over the arm, per frame  "
            f"p50 {np.median(mag):.4f}  p90 {np.quantile(mag, 0.9):.4f} rad"
        )
    print(f"  groups: {len(set(data['group']))}")
