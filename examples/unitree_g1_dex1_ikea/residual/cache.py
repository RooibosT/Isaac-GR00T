# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache what the residual head needs, once.

The baseline's backbone is frozen, so one observation always maps to one feature
tensor; and `a_base` has to be recomputed anyway because the policy was not
running while the human drove. Both are expensive and neither ever changes, so
they are written to disk here and every later script reads `.npz` files.

`a_base` is the mean of `--samples` draws. The head is flow matching: one draw
carries its own sampling noise, and that noise would land in every residual
target as if it were a correction the human made.

It is computed only where the residual target needs it -- the segments a human
drove. Elsewhere the target is exactly zero by definition rather than
`a_policy - a_base`: the recorded action there is what the deploy EXECUTED, after
RTC and the velocity clamp, so fitting it would teach the residual to reproduce
clamping that the deploy then applies again. Skipping those frames also removes
most of the cost, since the eight draws dominate it.

Frames are taken every `--stride` of each episode. Video decoding dominates the
cost and it is per-episode, so a stride costs almost nothing to widen.

Usage:
    python cache.py --dataset-path ... --out ... [--stride 2] [--samples 8]
"""

import argparse
import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


EX = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EX))
sys.path.insert(0, "/root/01_IKEA/url_lerobot")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--samples", type=int, default=8, help="draws averaged into a_base")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--segments",
        default=None,
        help="segments.json; without it every episode is treated as a correction",
    )
    ap.add_argument(
        "--base-owner",
        default="human",
        help="only episodes with this control_owner get a_base computed",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("pir", EX / "probe_insert_release.py")
    pir = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pir)
    importlib.import_module(Path(a.config).stem)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    tag = EmbodimentTag.resolve("new_embodiment")
    modality = MODALITY_CONFIGS[tag.value]
    loader = LeRobotEpisodeLoader(dataset_path=a.dataset_path, modality_configs=modality)
    pol = Gr00tPolicy(embodiment_tag=tag, model_path=a.checkpoint, device="cuda")
    probe = pir.Probe(pol, modality, tag)
    H, keys = probe.horizon, probe.keys

    owner_of = {}
    if a.segments:
        import json

        owner_of = {
            int(x["episode_index"]): x["control_owner"]
            for x in json.loads(Path(a.segments).read_text())["segments"]
        }

    feats, states, base, gt, ep_idx, frame_idx, has_base = [], [], [], [], [], [], []
    skipped = []
    n_eps = len(loader) if a.episodes == 0 else min(a.episodes, len(loader))
    for ep in range(n_eps):
        # 16 of intervention_full's 299 clips came out of the cut with no video
        # stream -- the source mp4s differ in length per camera, so a segment can
        # land past the end of one of them. `check_clip_lengths` catches it at
        # conversion; here the decoder simply raises, and the episode is dropped.
        try:
            traj = loader[ep]
        except Exception as exc:  # noqa: BLE001 - any decoder failure means no usable video
            skipped.append(ep)
            print(f"episode {ep}: unreadable, skipped ({type(exc).__name__})", flush=True)
            continue
        truth = np.concatenate(
            [np.vstack([np.asarray(x, np.float32) for x in traj[f"action.{k}"]]) for k in keys], -1
        )
        n = len(truth)
        for t in range(0, n - H, a.stride):
            parsed = probe.observation(traj, t)
            step = pol._to_vla_step_data(
                next(
                    iter(
                        pol._unbatch_observation(
                            {m: dict(parsed[m]) for m in ("video", "state", "language")}
                        )
                    )
                )
            )
            proc = pol.processor([{"type": pir.MessageType.EPISODE_STEP.value, "content": step}])
            col = pir._rec_to_dtype(pol.collate_fn([proc]), dtype=torch.bfloat16)
            with torch.no_grad():
                bi, ai = pol.model.prepare_input(col["inputs"])
                out = pol.model.backbone(bi)
            feats.append(out["backbone_features"][0].to(torch.float16).cpu().numpy())
            states.append(ai["state"][0].float().cpu().numpy().ravel())
            # "none" is for shards that are all leave-alone, such as
            # deploy_success: their target is zero, so a_base is never read.
            want = a.base_owner != "none" and ((not owner_of) or owner_of.get(ep) == a.base_owner)
            if want:
                draws = pir.stack(probe.sample(traj, a.samples, t, seed=911 + t), keys)
                base.append(draws.mean(0).astype(np.float32))
            else:
                base.append(np.zeros((H, truth.shape[-1]), np.float32))
            has_base.append(want)
            gt.append(truth[t : t + H].astype(np.float32))
            ep_idx.append(ep)
            frame_idx.append(t)
        del traj
        if ep % 10 == 0:
            print(f"episode {ep}/{n_eps}  frames so far {len(feats)}", flush=True)

    if skipped:
        print(f"skipped {len(skipped)} unreadable episodes: {skipped}")
    tok = min(f.shape[0] for f in feats)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        a.out,
        feats=np.stack([f[:tok] for f in feats]),
        state=np.stack(states).astype(np.float32),
        a_base=np.stack(base),
        a_gt=np.stack(gt),
        has_base=np.array(has_base),
        ep=np.array(ep_idx),
        frame=np.array(frame_idx),
        skipped=np.array(skipped, dtype=np.int64),
    )
    print(f"{len(feats)} frames, {tok} tokens -> {a.out}\nDONE")


if __name__ == "__main__":
    main()
