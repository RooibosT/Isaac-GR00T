# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Did the policy learn the small in-hole corrections, or never see them?

On the robot the leg goes in badly and the small settling corrections that the
demonstrations contain do not appear. Two explanations that look identical from
a deployment trace -- which records what the server returned *after* RTC has
frozen and blended it -- are that the model never predicted the correction, and
that it predicted it and the execution path washed it out.

This separates them without a robot. It replays validation observations through
the checkpoint and asks only what the model itself emitted.

The correction is defined the way the training data shows it: while the left
hand holds the leg, low down at the base, the held wrist descends for a stretch
and then rises again. Across the 117 training episodes those reversals ascend a
median 9.8 mm, so this is a millimetre-scale motion inside a 1.33 s chunk, not a
retreat -- the retreat is absent from the data entirely.

At each reversal the model is given the real observation and its own 40-step
chunk is compared against the recorded one in wrist height. If the model turns
around where the demonstrator did, it learned the behaviour and the loss is
downstream. If it keeps descending, there is nothing for RTC to have suppressed.

Random held frames are scored the same way as a control, because a model that is
simply inaccurate in z would fail the reversal test for an unrelated reason.
"""

import argparse
from copy import deepcopy
import gc
import importlib
import json
import logging
import os
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


URL_LEROBOT = Path(
    os.environ.get("URL_LEROBOT") or Path(__file__).resolve().parents[3] / "url_lerobot"
)
sys.path.insert(0, str(URL_LEROBOT))
from url_groot_deploy.common.g1_kinematics import G1WristKinematics  # noqa: E402


LEFT_GRIPPER_STATE = 29
MIN_RUN = 3  # frames of one-way motion, at the 3-frame sampling used to find them


def reversal_points(state, kin, stride=3):
    """Frames where the held left wrist stops descending and starts rising."""
    idx = np.arange(0, len(state), stride)
    z = np.array([kin.both_wrist_poses(state[i, 15:29], np.zeros(3))[0][2] for i in idx]) * 1000
    held = state[idx, LEFT_GRIPPER_STATE] < 1.0
    low = z < np.percentile(z[held], 35) if held.any() else np.zeros(len(z), bool)

    dz = np.diff(z)
    sign = np.sign(np.where(np.abs(dz) < 0.5, 0, dz))
    runs, cur, start = [], 0, 0
    for k in range(len(sign)):
        s = sign[k] if held[k] else 0
        if s != cur:
            if cur != 0 and k - start >= MIN_RUN:
                runs.append((cur, start, k))
            cur, start = s, k
    turns = []
    for a, b in zip(runs, runs[1:]):
        if a[0] < 0 and b[0] > 0 and b[1] - a[2] <= 5 and low[a[2]]:
            turns.append(int(idx[a[2]]))
    return turns, held


def wrist_z(kin, arm_traj):
    return np.array([kin.both_wrist_poses(q, np.zeros(3))[0][2] for q in arm_traj]) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset-path", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--controls-per-episode", type=int, default=25)
    # The validation split holds only ~10 of these reversals, which is too few to
    # separate "predicts the turn" from "happens to be near zero". Pointing this
    # at the training split buys ~3 per episode, and the number it produces is an
    # upper bound rather than a generalisation estimate -- but a model that
    # cannot reproduce the correction on data it was fitted to did not learn it
    # at all, which is the question being asked.
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sys.path.insert(0, str(args.config.parent))
    importlib.import_module(args.config.stem)

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    modality = MODALITY_CONFIGS[tag.value]
    horizon = len(modality["action"].delta_indices)
    action_keys = modality["action"].modality_keys
    obs_configs = deepcopy(modality)
    obs_configs.pop("action")

    kin = G1WristKinematics(str(URL_LEROBOT / "xr_teleoperate"), waist_zero=True)
    loader = LeRobotEpisodeLoader(dataset_path=str(args.dataset_path), modality_configs=modality)

    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(args.checkpoint), device="cuda")
    policy.model.action_head.num_inference_timesteps = args.denoising_steps

    rng = np.random.default_rng(20260906)
    rows = []
    n_ep = len(loader) if args.max_episodes <= 0 else min(args.max_episodes, len(loader))
    for ep in range(n_ep):
        traj = loader[ep]
        state = np.vstack([np.asarray(s, dtype=np.float32) for s in traj["state.left_arm"]])
        full = np.zeros((len(state), 31), np.float32)
        for key, sl in (("left_arm", slice(15, 22)), ("right_arm", slice(22, 29))):
            full[:, sl] = np.vstack([np.asarray(s, np.float32) for s in traj[f"state.{key}"]])
        full[:, LEFT_GRIPPER_STATE] = np.concatenate(
            [np.asarray(s, np.float32).ravel() for s in traj["state.left_gripper"]]
        )
        gt_arm = np.concatenate(
            [
                np.vstack([np.asarray(a, np.float32) for a in traj[f"action.{k}"]])
                for k in action_keys
                if k.endswith("_arm")
            ],
            axis=-1,
        )

        turns, held = reversal_points(full, kin)
        held_frames = np.where(full[:, LEFT_GRIPPER_STATE] < 1.0)[0]
        held_frames = held_frames[held_frames < len(state) - horizon]
        controls = (
            rng.choice(
                held_frames, size=min(args.controls_per_episode, len(held_frames)), replace=False
            )
            if len(held_frames)
            else []
        )
        picks = [(int(t), "reversal") for t in turns if t < len(state) - horizon] + [
            (int(t), "control") for t in controls
        ]
        logging.info(
            "episode %d/%d: %d reversals, %d controls",
            ep + 1,
            len(loader),
            len(turns),
            len(controls),
        )

        for t, kind in picks:
            dp = extract_step_data(traj, t, obs_configs, tag)
            obs = {f"state.{k}": v for k, v in dp.states.items()}
            for k, v in dp.images.items():
                obs[f"video.{k}"] = np.array(v)
            for lk in modality["language"].modality_keys:
                obs[lk] = dp.text
            torch.manual_seed(20260906 + t)
            chunk, _ = policy.get_action(parse_observation_gr00t(obs, modality))
            pred_arm = np.concatenate(
                [np.asarray(chunk[k])[0] for k in action_keys if k.endswith("_arm")], axis=-1
            )[:horizon]
            zp = wrist_z(kin, pred_arm)
            zg = wrist_z(kin, gt_arm[t : t + horizon])
            # Endpoint displacement misses a correction that rises and settles
            # back inside the chunk, which is most of them: the recorded
            # reversals ascend a median 9.8 mm while their endpoint
            # displacement is about 2 mm. Peak rise above the starting height
            # is the quantity the behaviour actually has.
            rec = {"episode": ep, "frame": t, "kind": kind}
            for k in (8, 16, horizon):
                rec[f"gt_dz{k}"] = float(zg[k - 1] - zg[0])
                rec[f"pred_dz{k}"] = float(zp[k - 1] - zp[0])
                rec[f"gt_up{k}"] = float(zg[:k].max() - zg[0])
                rec[f"pred_up{k}"] = float(zp[:k].max() - zp[0])
            rows.append(rec)
        del traj
        gc.collect()

    args.output.write_text(json.dumps(rows, indent=1))

    def summarise(kind):
        r = [x for x in rows if x["kind"] == kind]
        if not r:
            return
        print(f"\n=== {kind}  (n={len(r)}) ===")
        for k in (8, 16, horizon):
            g = np.array([x[f"gt_up{k}"] for x in r])
            p = np.array([x[f"pred_up{k}"] for x in r])
            corr = np.corrcoef(g, p)[0, 1] if len(r) > 2 else float("nan")
            print(
                f"  peak rise in {k:2d} steps: gt median {np.median(g):6.2f} mm   "
                f"pred median {np.median(p):6.2f} mm   corr {corr:+.3f}   "
                f"MAE {np.abs(g - p).mean():5.2f} mm"
            )
        # As a detection problem, which is how it matters on the robot: the
        # demonstrator lifted by at least THRESH inside the chunk -- did the
        # model's own chunk lift too?
        for thresh in (3.0, 5.0):
            g = np.array([x[f"gt_up{horizon}"] for x in r])
            p = np.array([x[f"pred_up{horizon}"] for x in r])
            pos = g >= thresh
            if pos.any() and (~pos).any():
                print(
                    f"  >={thresh:.0f} mm rise: gt has it in {100 * pos.mean():5.1f}% of frames; "
                    f"model predicts one in {100 * (p[pos] >= thresh).mean():5.1f}% of those "
                    f"and {100 * (p[~pos] >= thresh).mean():5.1f}% of the rest"
                )

    print(f"\ncheckpoint: {args.checkpoint}")
    summarise("reversal")
    summarise("control")


if __name__ == "__main__":
    main()
