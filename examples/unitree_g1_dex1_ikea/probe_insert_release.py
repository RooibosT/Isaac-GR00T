# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Is the release at the end of insertion buried by re-planning?

On the robot the leg goes onto the hole and the policy then wriggles, pushes down
or stalls instead of letting go and moving on to the rotation. One reading is
that the release is in the chunk, but late -- past the ~8 rows the deploy
executes before it replans -- and every re-plan from a similar observation puts
it late again.

`probe_gripper_release.py` does not answer this, because it anchors on the
*last* closed->open transition of the right gripper. A stage1_v2 episode opens
that gripper six times -- pick, regrasp, the insertion release, then the four
rotation cycles -- so its anchor is the end of the fourth rotation. This probe
anchors on the second opening, the one that ends the insertion, and only uses
episodes with exactly six.

Demonstrated insertion, in the state: the leg rests ~25 mm above its seat for
~2 s while the wrist tilts upright, drops ~26 mm in the last ~15 frames, and the
gripper opens ~8 frames after the drop begins. `drop` below is the last frame
before the release where the wrist is still 15 mm above its seated height.

Three modes, each written to a pickle for `analyze_insert_release.py`:

  dense  every 3 frames from 120 before the release to 48 after, `--seeds`
         samples each
  swap   state from one frame, images from another, around the drop -- which
         input tells the model the leg has gone in
  rtc    chunks chained the way the deploy chains them (shift 8, freeze 7,
         overlap 12, ramp 5): along the demonstration, and with the observation
         held fixed at a pre-drop frame, as a stand-in for a leg that cannot
         drop. RTC is reproduced through the action head's own inpainting branch.

All observations are recorded ones. Nothing here measures the robot's own
states; the `rtc` hold is an approximation of a stall, not a recording of one.
"""

import argparse
from copy import deepcopy
import gc
import importlib
import logging
import os
from pathlib import Path
import pickle
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
from gr00t.data.types import MessageType  # noqa: E402
from gr00t.data.utils import parse_observation_gr00t  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype  # noqa: E402


URL_LEROBOT = Path(
    os.environ.get("URL_LEROBOT") or Path(__file__).resolve().parents[3] / "url_lerobot"
)
sys.path.insert(0, str(URL_LEROBOT))
from url_groot_deploy.common.g1_kinematics import G1WristKinematics  # noqa: E402


# The deploy's chaining, from url_groot_deploy/client/run.py: a request goes out
# 8 ticks after the previous one, GR00T answers in ~5 ticks, the freeze covers
# the entry step plus 2, and the default overlap / ramp are 12 / 5.0.
SHIFT, FROZEN, OVERLAP, RAMP = 8, 7, 12, 5.0
DENSE = range(-120, 49, 3)
SWAP = (-36, -24, -15, -9, -3)  # relative to the release; gripper still closed
CHAIN_START, CHAIN_END = -64, 32
HOLDS = (-24, -15, -9)  # frames the observation is frozen at, relative to drop
HOLD_REPLANS = 12


def gripper_events(g: np.ndarray):
    lo, hi = np.quantile(g, 0.05), np.quantile(g, 0.95)
    closed, opened = lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo)
    state, ev = None, []
    for i, v in enumerate(g):
        new = "C" if v <= closed else ("O" if v >= opened else state)
        if new != state and new is not None:
            ev.append((i, new))
            state = new
    return ev, float((lo + hi) / 2)


class Probe:
    def __init__(self, policy: Gr00tPolicy, modality, tag: EmbodimentTag):
        self.policy = policy
        self.modality = modality
        self.tag = tag
        self.keys = list(modality["action"].modality_keys)
        self.horizon = len(modality["action"].delta_indices)
        self.sap = policy.processor.state_action_processor
        self.width = policy.model.action_head.action_dim
        self.obs_cfg = deepcopy(modality)
        self.obs_cfg.pop("action")

    def observation(self, traj, t_state: int, t_img: int | None = None):
        dp_s = extract_step_data(traj, t_state, self.obs_cfg, self.tag)
        dp_i = (
            dp_s
            if t_img in (None, t_state)
            else extract_step_data(traj, t_img, self.obs_cfg, self.tag)
        )
        flat = {f"state.{k}": v for k, v in dp_s.states.items()}
        for k, v in dp_i.images.items():
            flat[f"video.{k}"] = np.array(v)
        for lk in self.modality["language"].modality_keys:
            flat[lk] = dp_s.text
        return parse_observation_gr00t(flat, self.modality)

    def infer(self, parsed: list, seed: int, previous: list | None = None):
        """One forward for a batch of observations; `previous` turns on RTC.

        `previous[b]` is the absolute chunk that batch element b would be seeded
        from. It is shifted in absolute units and only then re-encoded against
        the new observation's state, as the deploy's rtc_overlay does, so a
        RELATIVE checkpoint gets its seed anchored to the right state.
        """
        torch.manual_seed(seed)
        batched = {"video": {}, "state": {}, "language": {}}
        for m in ("video", "state"):
            for k in parsed[0][m]:
                batched[m][k] = np.concatenate([p[m][k] for p in parsed], axis=0)
        for k in parsed[0]["language"]:
            batched["language"][k] = [p["language"][k][0] for p in parsed]

        pol = self.policy
        states, processed = [], []
        for obs in pol._unbatch_observation(batched):
            step = pol._to_vla_step_data(obs)
            states.append(step.states)
            processed.append(
                pol.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            )
        collated = _rec_to_dtype(pol.collate_fn(processed), dtype=torch.bfloat16)

        options = None
        if previous is not None:
            inputs = collated["inputs"]
            assert "action" not in inputs, "processor already emits an action tensor"
            seed_rows = torch.zeros((len(parsed), self.horizon, self.width))
            for b, prev in enumerate(previous):
                aligned = {}
                for k in self.keys:
                    tail = prev[k][SHIFT:]
                    tail = np.concatenate(
                        [tail, np.repeat(tail[-1:], len(prev[k]) - len(tail), axis=0)]
                    )
                    aligned[k] = tail.astype(np.float32)
                current = {k: np.asarray(v, np.float32) for k, v in states[b].items()}
                enc = self.sap.apply_action(aligned, self.tag.value, current)
                flat = np.concatenate([enc[k] for k in self.keys], axis=-1)
                # The head copies rows [H - overlap, H) of this tensor into the
                # first `overlap` rows of its initial noise.
                seed_rows[b, self.horizon - OVERLAP :, : flat.shape[-1]] = torch.from_numpy(
                    flat[:OVERLAP]
                )
            inputs["action"] = seed_rows.to(torch.bfloat16)
            options = {
                "action_horizon": self.horizon,
                "rtc_overlap_steps": OVERLAP,
                "rtc_frozen_steps": FROZEN,
                "rtc_ramp_rate": RAMP,
            }

        with torch.inference_mode():
            out = pol.model.get_action(**collated, options=options)
        normalized = out["action_pred"].float().cpu().numpy()
        batch_states = {
            k: np.stack([s[k] for s in states], axis=0)
            for k in self.modality["state"].modality_keys
        }
        decoded = pol.processor.decode_action(normalized, self.tag, batch_states)
        return [
            {k: np.asarray(decoded[k][b], np.float32)[: self.horizon] for k in self.keys}
            for b in range(len(parsed))
        ]

    def sample(self, traj, n, t_state, t_img=None, previous=None, seed=0):
        """`n` samples at one observation, or one per chain when RTC-seeded."""
        obs = self.observation(traj, t_state, t_img)
        batch = [obs] * (n if previous is None else len(previous))
        return self.infer(batch, seed, previous)


def stack(chunks: list, keys: list) -> np.ndarray:
    """list of per-sample dicts -> (S, H, D) in action-key order."""
    return np.stack([np.concatenate([c[k] for k in keys], axis=-1) for c in chunks])


def main() -> None:
    global OVERLAP, RAMP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset-path", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--modes", default="dense,swap,rtc")
    # dex1:real_g1 runs 20 / 2.0 (profiles/dex1_real_g1.env) to soften the seam
    # its gripper column tripped the hand-step gate on.
    ap.add_argument("--rtc-overlap", type=int, default=OVERLAP)
    ap.add_argument("--rtc-ramp", type=float, default=RAMP)
    ap.add_argument("--episodes", default="", help="comma-separated loader indices; default all")
    a = ap.parse_args()
    modes = set(a.modes.split(","))
    OVERLAP, RAMP = a.rtc_overlap, a.rtc_ramp

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sys.path.insert(0, str(a.config.parent))
    importlib.import_module(a.config.stem)

    tag = EmbodimentTag.resolve(a.embodiment_tag)
    modality = MODALITY_CONFIGS[tag.value]
    loader = LeRobotEpisodeLoader(dataset_path=str(a.dataset_path), modality_configs=modality)
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(a.checkpoint), device="cuda")
    policy.model.action_head.num_inference_timesteps = a.denoising_steps
    probe = Probe(policy, modality, tag)
    keys, H, S = probe.keys, probe.horizon, a.seeds
    kin = G1WristKinematics(str(URL_LEROBOT / "xr_teleoperate"), waist_zero=True)

    episodes = [int(e) for e in a.episodes.split(",")] if a.episodes else range(len(loader))
    out = {
        "keys": keys,
        "horizon": H,
        "checkpoint": str(a.checkpoint),
        "rtc": {"shift": SHIFT, "frozen": FROZEN, "overlap": OVERLAP, "ramp": RAMP},
        "episodes": [],
    }
    for ep in episodes:
        traj = loader[ep]
        gt = np.concatenate(
            [np.vstack([np.asarray(x, np.float32) for x in traj[f"action.{k}"]]) for k in keys],
            axis=-1,
        )
        widths = [np.asarray(traj[f"action.{k}"].iloc[0]).size for k in keys]
        col = int(np.sum(widths[: keys.index("right_gripper")]))
        ev, thresh = gripper_events(gt[:, col])
        ops = [i for i, s in ev if s == "O" and i > 0]
        if len(ops) != 6:
            logging.info("episode %d: %d openings, skipped", ep, len(ops))
            continue
        release = ops[1]
        arm_state = np.concatenate(
            [
                np.vstack([np.asarray(x, np.float64) for x in traj["state.left_arm"]]),
                np.vstack([np.asarray(x, np.float64) for x in traj["state.right_arm"]]),
            ],
            axis=-1,
        )
        z = np.array(
            [kin.both_wrist_poses(arm_state[t], np.zeros(3))[1][2] for t in range(len(gt))]
        )
        above = np.where(z[:release] > z[release] + 0.015)[0]
        drop = int(above[-1]) if len(above) else release
        rec = {
            "episode": ep,
            "release": release,
            "drop": drop,
            "openings": ops,
            "thresh": thresh,
            "T": len(gt),
            "z": z,
            "gt": gt,
            "arm_state": arm_state,
        }
        logging.info("episode %d: release %d, drop %d, T %d", ep, release, drop, len(gt))

        if "dense" in modes:
            rec["dense"] = []
            for off in DENSE:
                t = release + off
                if not 0 <= t < len(gt) - H:
                    continue
                chunks = probe.sample(traj, S, t, seed=20260911 + t)
                rec["dense"].append({"t": t, "pred": stack(chunks, keys)})

        if "swap" in modes:
            rec["swap"] = []
            for os_ in SWAP:
                for oi in SWAP:
                    ts, ti = release + os_, release + oi
                    chunks = probe.sample(traj, S, ts, ti, seed=20260912 + 97 * ts + ti)
                    rec["swap"].append({"t_state": ts, "t_img": ti, "pred": stack(chunks, keys)})

        if "rtc" in modes:
            chains = []
            # Along the demonstration: plain sampling and RTC-chained, side by side.
            prev = None
            for k, t in enumerate(range(release + CHAIN_START, release + CHAIN_END, SHIFT)):
                if not 0 <= t < len(gt) - H:
                    prev = None
                    continue
                plain = probe.sample(traj, S, t, seed=20260913 + t)
                seeded = (
                    probe.sample(traj, S, t, previous=prev, seed=20260914 + t)
                    if prev is not None
                    else plain
                )
                chains.append(
                    {
                        "kind": "follow",
                        "k": k,
                        "t": t,
                        "plain": stack(plain, keys),
                        "rtc": stack(seeded, keys),
                        "seeded": prev is not None,
                    }
                )
                prev = seeded
            # Held at a pre-drop frame: the scene and the arm stop changing.
            for off in HOLDS:
                t = drop + off
                if not 0 <= t < len(gt) - H:
                    continue
                prev = None
                for k in range(HOLD_REPLANS):
                    chunks = probe.sample(traj, S, t, previous=prev, seed=20260915 + 131 * t + k)
                    chains.append(
                        {
                            "kind": f"hold{off:+d}",
                            "k": k,
                            "t": t,
                            "rtc": stack(chunks, keys),
                            "seeded": prev is not None,
                        }
                    )
                    prev = chunks
            rec["rtc"] = chains

        out["episodes"].append(rec)
        del traj
        gc.collect()
        a.output.write_bytes(pickle.dumps(out))

    a.output.write_bytes(pickle.dumps(out))
    logging.info("wrote %s (%d episodes)", a.output, len(out["episodes"]))


if __name__ == "__main__":
    main()
