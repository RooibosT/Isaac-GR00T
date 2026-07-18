# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open-loop MSE scan over checkpoints on the held-out val split.

For every checkpoint-* under --checkpoints-dir, runs full-chunk open-loop
prediction on every episode of --dataset-path and reports action MSE/MAE
(full chunk, first --first-k steps, arm/gripper split). Observations are
extracted once and shared across checkpoints; the diffusion seed is fixed
per window so checkpoints see identical noise.

Usage:
    python examples/unitree_g1_dex1/eval_val_mse.py \
        --checkpoints-dir /path/to/run/exp_name \
        --dataset-path demo_data/RooibosT/g1_pick_redblock_dex1_sim_merged_107demo_val \
        --output /path/to/run/val_mse_scan.json
"""

import argparse
from copy import deepcopy
import gc
import json
import logging
from pathlib import Path
import re
import time

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import transformers.tokenization_utils_base as _tub


# transformers' mistral-regex fixup calls the hub model_info API even when
# HF_HUB_OFFLINE=1 (and 429s under rate limiting). The backbone tokenizer here is
# Qwen3, never mistral, so skip the fixup and keep checkpoint loading fully local.
_tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(
    lambda cls, tokenizer, *args, **kwargs: tokenizer
)


def build_windows(loader, embodiment_tag, horizon):
    """Extract (episode, parsed_obs, gt_action[horizon, D]) for every full window."""
    obs_configs = deepcopy(loader.modality_configs)
    obs_configs.pop("action")
    action_keys = loader.modality_configs["action"].modality_keys
    windows = []
    for ep in range(len(loader)):
        traj = loader[ep]
        gt_all = np.concatenate(
            [
                np.vstack([np.asarray(a, dtype=np.float32) for a in traj[f"action.{k}"]])
                for k in action_keys
            ],
            axis=-1,
        )
        for t in range(0, len(traj) - horizon + 1, horizon):
            dp = extract_step_data(traj, t, obs_configs, embodiment_tag)
            obs = {f"state.{k}": v for k, v in dp.states.items()}
            for k, v in dp.images.items():
                obs[f"video.{k}"] = np.array(v)
            for lk in loader.modality_configs["language"].modality_keys:
                obs[lk] = dp.text
            parsed = parse_observation_gr00t(obs, loader.modality_configs)
            windows.append((ep, parsed, gt_all[t : t + horizon]))
        del traj
        gc.collect()
        logging.info("episode %d extracted (%d windows so far)", ep, len(windows))
    return windows


def eval_checkpoint(policy, windows, action_keys, horizon, first_k):
    n_eps = max(ep for ep, _, _ in windows) + 1
    sq_full, ab_full, sq_first, ab_first = [], [], [], []
    per_ep_sq = [[] for _ in range(n_eps)]
    arm_ab, grip_ab = [], []
    arm_idx = grip_idx = None

    for i, (ep, obs, gt) in enumerate(windows):
        torch.manual_seed(20260718 + i)
        chunk, _ = policy.get_action(deepcopy(obs))
        pred = np.concatenate([np.asarray(chunk[k])[0] for k in action_keys], axis=-1)[:horizon]
        if arm_idx is None:
            dims, start = {}, 0
            for k in action_keys:
                d = np.asarray(chunk[k]).shape[-1]
                dims[k] = list(range(start, start + d))
                start += d
            arm_idx = [d for k in action_keys if "arm" in k for d in dims[k]]
            grip_idx = [d for k in action_keys if "gripper" in k for d in dims[k]]
        err = pred - gt
        sq_full.append(np.mean(err**2))
        ab_full.append(np.mean(np.abs(err)))
        sq_first.append(np.mean(err[:first_k] ** 2))
        ab_first.append(np.mean(np.abs(err[:first_k])))
        arm_ab.append(np.mean(np.abs(err[:, arm_idx])))
        grip_ab.append(np.mean(np.abs(err[:, grip_idx])))
        per_ep_sq[ep].append(np.mean(err**2))

    return {
        "mse": float(np.mean(sq_full)),
        "mae": float(np.mean(ab_full)),
        f"mse_first{first_k}": float(np.mean(sq_first)),
        f"mae_first{first_k}": float(np.mean(ab_first)),
        "mae_arm": float(np.mean(arm_ab)),
        "mae_gripper": float(np.mean(grip_ab)),
        "per_episode_mse": [float(np.mean(s)) if s else None for s in per_ep_sq],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-dir", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--first-k", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tag = EmbodimentTag.resolve(args.embodiment_tag)

    ckpts = sorted(
        Path(args.checkpoints_dir).glob("checkpoint-*"),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1)),
    )
    assert ckpts, f"no checkpoints under {args.checkpoints_dir}"
    logging.info("found %d checkpoints: %s ... %s", len(ckpts), ckpts[0].name, ckpts[-1].name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(ckpts[0]), device=device)
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    modality = policy.get_modality_config()
    action_keys = modality["action"].modality_keys
    horizon = len(modality["action"].delta_indices)
    logging.info("action horizon = %d, keys = %s", horizon, action_keys)

    loader = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality)
    logging.info("val dataset: %d episodes", len(loader))
    windows = build_windows(loader, tag, horizon)
    logging.info("total windows: %d", len(windows))

    results = {}
    out_path = Path(args.output)
    for ci, ckpt in enumerate(ckpts):
        if ci > 0:
            del policy
            gc.collect()
            torch.cuda.empty_cache()
            policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(ckpt), device=device)
            policy.model.action_head.num_inference_timesteps = args.denoising_steps
        t0 = time.time()
        results[ckpt.name] = eval_checkpoint(policy, windows, action_keys, horizon, args.first_k)
        logging.info(
            "%s done in %.1fs: mse=%.6f mae=%.4f mae_first%d=%.4f arm=%.4f grip=%.4f",
            ckpt.name,
            time.time() - t0,
            results[ckpt.name]["mse"],
            results[ckpt.name]["mae"],
            args.first_k,
            results[ckpt.name][f"mae_first{args.first_k}"],
            results[ckpt.name]["mae_arm"],
            results[ckpt.name]["mae_gripper"],
        )
        out_path.write_text(json.dumps(results, indent=2))

    best = min(results, key=lambda k: results[k]["mse"])
    logging.info("BEST by full-chunk MSE: %s (%.6f)", best, results[best]["mse"])
    logging.info("results written to %s", out_path)


if __name__ == "__main__":
    main()
