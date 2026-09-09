# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score a checkpoint on the insertion phase alone, not the whole episode.

The full-episode scan is dominated by transit -- reaching, lifting, carrying --
where proprioception predicts the next move well and nothing much is at stake.
The part that decides whether the leg goes in is short: the descent that starts
once the leg is over the base, and the contact that follows. A change that helps
there and hurts elsewhere is invisible in the aggregate, and a change that hurts
everywhere including there is a different verdict from one that hurts only the
transit.

An insertion attempt is a local minimum of the held wrist's height, low down.
Around each one:

    approach   the 1.5 s before it   -- the leg is being lowered onto the hole
    contact    the 1.5 s after it    -- pressing, settling, adjusting

Both are scored with the metrics the main scan uses, so the numbers sit beside
the recorded ones. `all` is every held frame, for reference.
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


LEFT_GRIPPER = 29
WIN = 45  # 1.5 s at 30 Hz, either side of the attempt


def insertion_attempts(z, held, low_pct=35, min_sep=45):
    """Local minima of the held wrist height, restricted to the low band."""
    if not held.any():
        return []
    thresh = np.percentile(z[held], low_pct)
    cand = [
        t
        for t in range(1, len(z) - 1)
        if held[t] and z[t] < thresh and z[t] <= z[t - 1] and z[t] <= z[t + 1]
    ]
    keep = []
    for t in cand:
        if not keep or t - keep[-1] >= min_sep:
            keep.append(t)
        elif z[t] < z[keep[-1]]:
            keep[-1] = t
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset-path", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-episodes", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sys.path.insert(0, str(args.config.parent))
    importlib.import_module(args.config.stem)

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    modality = MODALITY_CONFIGS[tag.value]
    horizon = len(modality["action"].delta_indices)
    action_keys = modality["action"].modality_keys
    obs_cfg = deepcopy(modality)
    obs_cfg.pop("action")

    kin = G1WristKinematics(str(URL_LEROBOT / "xr_teleoperate"), waist_zero=True)
    loader = LeRobotEpisodeLoader(dataset_path=str(args.dataset_path), modality_configs=modality)
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(args.checkpoint), device="cuda")
    policy.model.action_head.num_inference_timesteps = args.denoising_steps

    arm_keys = [k for k in action_keys if k.endswith("_arm")]
    grip_keys = [k for k in action_keys if "gripper" in k]
    n_ep = len(loader) if args.max_episodes <= 0 else min(args.max_episodes, len(loader))
    acc = {}
    zero = np.zeros(3)

    for ep in range(n_ep):
        traj = loader[ep]
        arm_state = np.concatenate(
            [
                np.vstack([np.asarray(s, np.float32) for s in traj[f"state.{k}"]])
                for k in ("left_arm", "right_arm")
            ],
            axis=-1,
        )
        grip = np.concatenate(
            [np.asarray(s, np.float32).ravel() for s in traj["state.left_gripper"]]
        )
        gt_arm = np.concatenate(
            [np.vstack([np.asarray(a, np.float32) for a in traj[f"action.{k}"]]) for k in arm_keys],
            axis=-1,
        )
        gt_grip = np.concatenate(
            [
                np.vstack([np.asarray(a, np.float32) for a in traj[f"action.{k}"]])
                for k in grip_keys
            ],
            axis=-1,
        )
        z = np.array([kin.both_wrist_poses(q, zero)[0][2] for q in arm_state]) * 1000
        held = grip < 1.0
        attempts = insertion_attempts(z, held)

        phase = np.full(len(z), "", dtype=object)
        for t in attempts:
            phase[max(0, t - WIN) : t] = "approach"
            phase[t : min(len(z), t + WIN)] = "contact"
        logging.info("episode %d/%d: %d insertion attempts", ep + 1, n_ep, len(attempts))

        for t in range(0, len(z) - horizon, args.stride):
            tags = ["all"] if held[t] else []
            if phase[t]:
                tags.append(phase[t])
            if not tags:
                continue
            dp = extract_step_data(traj, t, obs_cfg, tag)
            obs = {f"state.{k}": v for k, v in dp.states.items()}
            for k, v in dp.images.items():
                obs[f"video.{k}"] = np.array(v)
            for lk in modality["language"].modality_keys:
                obs[lk] = dp.text
            torch.manual_seed(20260907 + t)
            chunk, _ = policy.get_action(parse_observation_gr00t(obs, modality))
            pa = np.concatenate([np.asarray(chunk[k])[0] for k in arm_keys], axis=-1)[:horizon]
            pg = np.concatenate([np.asarray(chunk[k])[0] for k in grip_keys], axis=-1)[:horizon]
            ga, gg = gt_arm[t : t + horizon], gt_grip[t : t + horizon]
            arm_h = np.abs(pa - ga).mean(axis=1)
            ee_h = np.array(
                [
                    0.5
                    * sum(
                        np.linalg.norm(p[:3] - g[:3])
                        for p, g in zip(
                            kin.both_wrist_poses(pa[i], zero), kin.both_wrist_poses(ga[i], zero)
                        )
                    )
                    for i in range(horizon)
                ]
            )
            rec = {
                "arm": float(arm_h.mean()),
                "arm5": float(arm_h[:5].mean()),
                "arm8": float(arm_h[:8].mean()),
                "ee_mm": float(ee_h.mean() * 1000),
                "ee8_mm": float(ee_h[:8].mean() * 1000),
                "grip": float(np.abs(pg - gg).mean()),
            }
            for tg in tags:
                acc.setdefault(tg, []).append(rec)
        del traj
        gc.collect()

    out = {
        k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} | {"n": len(v)}
        for k, v in acc.items()
    }
    args.output.write_text(json.dumps(out, indent=1))
    print(f"\ncheckpoint: {args.checkpoint}")
    for k in ("all", "approach", "contact"):
        if k in out:
            o = out[k]
            print(
                f"  {k:9s} n={o['n']:5d}  arm {np.degrees(o['arm']):6.3f}  arm8 {np.degrees(o['arm8']):6.3f}  "
                f"EE8 {o['ee8_mm']:6.2f} mm  grip {o['grip']:.4f}"
            )


if __name__ == "__main__":
    main()
