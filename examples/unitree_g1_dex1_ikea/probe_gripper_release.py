# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Does the policy ever decide to let go of the leg?

On the robot it inserts the leg and does not release, so it never reaches the
rotation. Two explanations look identical from a deployment trace: the model
never predicts the opening, or it predicts it and closed-loop state never
arrives at the frame where it would.

This separates them without a robot. It replays recorded observations from
frames leading up to a demonstrated release and asks only what the model itself
emits -- whether its own 40-step chunk contains the gripper opening, and how
far ahead it puts it.

The release is located as the last closed->open transition of the *right*
gripper in the back two thirds of an episode; the right hand is the one that
carries the leg into the hole here, and the transition is present in 98% of
training episodes, so it is a reliable anchor even without subtask labels.

Controls are frames where the gripper is closed and no release is due for three
seconds. A model that emits "open" everywhere would pass the release test for
the wrong reason.
"""

import argparse
from copy import deepcopy
import gc
import importlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import transformers.tokenization_utils_base as _tub


_tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(
    lambda cls, tokenizer, *args, **kwargs: tokenizer
)

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS  # noqa: E402
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.utils import parse_observation_gr00t  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402


OFFSETS = (-45, -30, -15, -5, 0)
CONTROL_GAP = 90  # frames a control must sit clear of any release


def find_release(grip: np.ndarray) -> int | None:
    """Last closed->open transition of the gripper in the back two thirds.

    Hysteresis on the episode's own range rather than a quantile comparison: the
    commanded gripper saturates at its open value, so the 85th percentile *is*
    the maximum and a strict `>` against it almost never fires.
    """
    lo, hi = np.quantile(grip, 0.05), np.quantile(grip, 0.95)
    if hi - lo < 1e-3:
        return None
    closed, opened = lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo)
    for i in range(len(grip) - 1, len(grip) // 3, -1):
        if grip[i] >= opened and (grip[max(0, i - 20) : i] <= closed).any():
            return i
    return None


def opens_at(traj: np.ndarray, thresh: float) -> int | None:
    """First step of a chunk at or above the open threshold, or None."""
    hits = np.where(traj >= thresh)[0]
    return int(hits[0]) if len(hits) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset-path", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--gripper-key", default="right_gripper")
    ap.add_argument("--controls-per-episode", type=int, default=6)
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sys.path.insert(0, str(a.config.parent))
    importlib.import_module(a.config.stem)

    tag = EmbodimentTag.resolve(a.embodiment_tag)
    modality = MODALITY_CONFIGS[tag.value]
    horizon = len(modality["action"].delta_indices)
    obs_configs = deepcopy(modality)
    obs_configs.pop("action")

    loader = LeRobotEpisodeLoader(dataset_path=str(a.dataset_path), modality_configs=modality)
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(a.checkpoint), device="cuda")
    policy.model.action_head.num_inference_timesteps = a.denoising_steps

    rng = np.random.default_rng(20260909)
    rows = []
    n_ep = len(loader) if a.max_episodes <= 0 else min(a.max_episodes, len(loader))
    for ep in range(n_ep):
        traj = loader[ep]
        gt = np.concatenate(
            [np.asarray(x, np.float32).ravel() for x in traj[f"action.{a.gripper_key}"]]
        )
        rel = find_release(gt)
        if rel is None:
            logging.info("episode %d/%d: no release found", ep + 1, n_ep)
            del traj
            continue
        # Threshold midway between the episode's own closed and open levels, so
        # a scale difference between action and state units cannot matter.
        thresh = float(np.quantile(gt, 0.05) + np.quantile(gt, 0.95)) / 2
        picks = [(rel + o, f"rel{o:+d}") for o in OFFSETS if 0 <= rel + o < len(gt) - horizon]
        far = np.array(
            [i for i in range(len(gt) - horizon) if abs(i - rel) > CONTROL_GAP and gt[i] < thresh]
        )
        if len(far):
            picks += [
                (int(t), "control")
                for t in rng.choice(far, size=min(a.controls_per_episode, len(far)), replace=False)
            ]
        logging.info(
            "episode %d/%d: release at %d/%d, %d queries", ep + 1, n_ep, rel, len(gt), len(picks)
        )

        for t, kind in picks:
            dp = extract_step_data(traj, t, obs_configs, tag)
            obs = {f"state.{k}": v for k, v in dp.states.items()}
            for k, v in dp.images.items():
                obs[f"video.{k}"] = np.array(v)
            for lk in modality["language"].modality_keys:
                obs[lk] = dp.text
            torch.manual_seed(20260909 + t)
            chunk, _ = policy.get_action(parse_observation_gr00t(obs, modality))
            pred = np.asarray(chunk[a.gripper_key])[0][:horizon].ravel()
            rows.append(
                {
                    "episode": ep,
                    "frame": t,
                    "kind": kind,
                    "release": rel,
                    "thresh": thresh,
                    "gt_open_step": opens_at(gt[t : t + horizon], thresh),
                    "pred_open_step": opens_at(pred, thresh),
                    "gt_max": float(gt[t : t + horizon].max()),
                    "pred_max": float(pred.max()),
                    "gt_start": float(gt[t]),
                    "pred_start": float(pred[0]),
                }
            )
        del traj
        gc.collect()

    a.output.write_text(json.dumps(rows, indent=1))
    print(f"\ncheckpoint: {a.checkpoint}")
    print(
        f"{'window':10} {'n':>4}  {'GT opens':>9}  {'pred opens':>11}  "
        f"{'GT step':>8}  {'pred step':>10}  {'pred max':>9}"
    )
    for kind in [f"rel{o:+d}" for o in OFFSETS] + ["control"]:
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        g = [r for r in sel if r["gt_open_step"] is not None]
        p = [r for r in sel if r["pred_open_step"] is not None]
        gs = np.median([r["gt_open_step"] for r in g]) if g else float("nan")
        ps = np.median([r["pred_open_step"] for r in p]) if p else float("nan")
        print(
            f"{kind:10} {len(sel):4}  {100 * len(g) / len(sel):7.0f}%  {100 * len(p) / len(sel):9.0f}%  "
            f"{gs:8.1f}  {ps:10.1f}  {np.median([r['pred_max'] for r in sel]):9.3f}"
        )


if __name__ == "__main__":
    main()
