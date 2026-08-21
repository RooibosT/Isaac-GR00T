# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open-loop checkpoint scan for the IKEA runs — the model-selection signal.

`eval_loss` is the flow-matching regression objective at random noise levels; it
rose after ~7.5k steps in all five BCT runs while open-loop action accuracy kept
improving, so it does not select checkpoints. This does.

Three things it reports that `eval_val_mse.py` does not:

* **A denser window stride.** That script strides by the horizon, which on this
  val split yields only 164 non-overlapping windows — too few to separate
  checkpoints that differ by 1-2%. Stride 10 gives 614. Windows overlap, so they
  are not independent samples, but every checkpoint is scored on the identical
  fixed set with a per-window fixed diffusion seed, which is what a *comparison*
  needs.
* **Per-task error.** `rotate leg to tighten` is 49.8% of training windows and
  `pick table leg` only 22.0%. If pick is the weak task, rebalancing has a
  reason; if not, throwing away 27% of a 60k-frame dataset does not.
* **Error in EE space, via FK.** Joint-space MAE weights every joint equally,
  but a shoulder degree moves the wrist ~5.5 mm and a wrist degree ~0.9 mm.
  What the task cares about is where the gripper ends up. This costs one FK per
  window and tells us whether training in EE space is worth a run at all.

Usage:
    python examples/unitree_g1_dex1_ikea/scan_ikea.py \
        --checkpoints-dir outputs/g1_dex1_ikea_relarm_3view_aug_b64 \
        --dataset-path datasets/carroll511/G1_Dex1_IKEA_table_30hz_val \
        --config examples/unitree_g1_dex1_ikea/g1_dex1_ikea_relarm_3view_aug_config.py \
        --output outputs/g1_dex1_ikea_relarm_3view_aug_b64/scan.json
"""

import argparse
from copy import deepcopy
import gc
import importlib
import json
import logging
from pathlib import Path
import re
import sys
import time

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


sys.path.insert(0, "/home/chan/IKEA/url_lerobot")
from url_groot_deploy.g1_kinematics import G1WristKinematics  # noqa: E402


XR_REPO = "/home/chan/IKEA/url_lerobot/xr_teleoperate"
FIRST_KS = (5, 8, 16)


def build_windows(loader, embodiment_tag, horizon, stride, tasks):
    """(episode, task, parsed_obs, gt[horizon, D]) every `stride` frames."""
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
        for t in range(0, len(traj) - horizon + 1, stride):
            dp = extract_step_data(traj, t, obs_configs, embodiment_tag)
            obs = {f"state.{k}": v for k, v in dp.states.items()}
            for k, v in dp.images.items():
                obs[f"video.{k}"] = np.array(v)
            for lk in loader.modality_configs["language"].modality_keys:
                obs[lk] = dp.text
            windows.append(
                (
                    ep,
                    tasks[ep],
                    parse_observation_gr00t(obs, loader.modality_configs),
                    gt_all[t : t + horizon],
                )
            )
        del traj
        gc.collect()
        logging.info("episode %d/%d extracted (%d windows)", ep + 1, len(loader), len(windows))
    return windows


def eval_checkpoint(policy, windows, action_keys, horizon, kin):
    idx = None
    acc = {}  # task -> list of dicts
    per_h_arm = np.zeros(horizon)
    per_h_ee = np.zeros(horizon)
    n = 0

    for i, (ep, task, obs, gt) in enumerate(windows):
        torch.manual_seed(20260718 + i)
        chunk, _ = policy.get_action(deepcopy(obs))
        pred = np.concatenate([np.asarray(chunk[k])[0] for k in action_keys], axis=-1)[:horizon]
        if idx is None:
            dims, s = {}, 0
            for k in action_keys:
                d = np.asarray(chunk[k]).shape[-1]
                dims[k] = list(range(s, s + d))
                s += d
            idx = {
                "arm": [d for k in action_keys if k.endswith("_arm") for d in dims[k]],
                "grip": [d for k in action_keys if "gripper" in k for d in dims[k]],
            }
        err = pred - gt
        arm_h = np.abs(err[:, idx["arm"]]).mean(axis=1)

        # EE space: FK both the prediction and the ground truth, compare the wrists
        zero = np.zeros(3)
        ee_h = np.zeros(horizon)
        for t in range(horizon):
            pl, pr = kin.both_wrist_poses(pred[t, idx["arm"]], zero)
            gl, gr = kin.both_wrist_poses(gt[t, idx["arm"]], zero)
            ee_h[t] = 0.5 * (np.linalg.norm(pl[:3] - gl[:3]) + np.linalg.norm(pr[:3] - gr[:3]))
        # how far the wrists actually travel over the chunk — the scale for ee_h
        g0l, g0r = kin.both_wrist_poses(gt[0, idx["arm"]], zero)
        gTl, gTr = kin.both_wrist_poses(gt[-1, idx["arm"]], zero)
        travel = 0.5 * (np.linalg.norm(gTl[:3] - g0l[:3]) + np.linalg.norm(gTr[:3] - g0r[:3]))

        rec = {
            "mse": float(np.mean(err**2)),
            "mae_arm": float(arm_h.mean()),
            "mae_grip": float(np.abs(err[:, idx["grip"]]).mean()),
            "ee_mm": float(ee_h.mean() * 1000),
            "ee_travel_mm": float(travel * 1000),
        }
        for k in FIRST_KS:
            rec[f"mae_arm_first{k}"] = float(arm_h[:k].mean())
            rec[f"ee_mm_first{k}"] = float(ee_h[:k].mean() * 1000)
        acc.setdefault(task, []).append(rec)
        acc.setdefault("__all__", []).append(rec)
        per_h_arm += arm_h
        per_h_ee += ee_h
        n += 1

    def agg(rs):
        return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} | {"n": len(rs)}

    out = {t: agg(rs) for t, rs in acc.items()}
    out["__per_horizon__"] = {
        "arm_deg": list(np.degrees(per_h_arm / n)),
        "ee_mm": list(per_h_ee / n * 1000),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-dir", required=True, type=Path)
    ap.add_argument("--dataset-path", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--denoising-steps", type=int, default=4)
    # Scanning one run's 10 checkpoints is ~2.5 h on one GPU; splitting the step
    # list across two GPUs halves that. Each shard writes its own --output, so
    # merge them afterwards.
    ap.add_argument("--steps", default="", help="comma-separated checkpoint steps; default all")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sys.path.insert(0, str(args.config.parent))
    importlib.import_module(args.config.stem)

    tag = EmbodimentTag.resolve("new_embodiment")
    modality = MODALITY_CONFIGS["new_embodiment"]
    horizon = len(modality["action"].delta_indices)
    action_keys = modality["action"].modality_keys

    tasks = [
        json.loads(line)["tasks"][0] for line in open(args.dataset_path / "meta/episodes.jsonl")
    ]
    loader = LeRobotEpisodeLoader(dataset_path=str(args.dataset_path), modality_configs=modality)
    windows = build_windows(loader, tag, horizon, args.stride, tasks)
    logging.info("windows: %d (stride %d)", len(windows), args.stride)

    kin = G1WristKinematics(XR_REPO, waist_zero=True)

    ckpts = sorted(
        (p for p in args.checkpoints_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1)),
    )
    if args.steps:
        want = {int(s) for s in args.steps.split(",")}
        ckpts = [p for p in ckpts if int(re.search(r"-(\d+)", p.name).group(1)) in want]
    logging.info("checkpoints: %s", [p.name for p in ckpts])

    results = {}
    if args.output.exists():
        results = json.loads(args.output.read_text())
    for ck in ckpts:
        if ck.name in results:
            logging.info("skip %s (already scanned)", ck.name)
            continue
        t0 = time.time()
        policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(ck), device="cuda")
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
        results[ck.name] = eval_checkpoint(policy, windows, action_keys, horizon, kin)
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        a = results[ck.name]["__all__"]
        logging.info(
            "%s  mse %.5f  arm %.3f deg  ee %.2f mm  grip %.4f  (%.1f min)",
            ck.name,
            a["mse"],
            np.degrees(a["mae_arm"]),
            a["ee_mm"],
            a["mae_grip"],
            (time.time() - t0) / 60,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1))

    best = min(results, key=lambda k: results[k]["__all__"]["mse"])
    logging.info("BEST by mse: %s", best)


if __name__ == "__main__":
    main()
