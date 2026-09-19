# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fit the correction head on the cached shards.

The split is by SOURCE EPISODE, not by frame. Frames inside one rescue are nearly
identical to each other, so a frame-level split would report a score that says
nothing about a rescue the head has not seen.

`--clip` is the interesting knob and it is swept in one process, because loading
the 22 GB cache costs more than the fits do. It is not a safety bound so much as
the dial between two different things. The corrections this data contains need
0.59-0.90 rad on the rows the deploy executes, and the gap GROWS with time since
the takeover (0.589 rad in the first second, 0.895 by the fifth) -- the human is
not adjusting the policy's plan, they are driving a different trajectory, and the
further they go the further the policy's own prediction from that state diverges.
So a small clip makes a true residual that can only nudge, and a large one makes a
takeover policy with a learned gate.

Two numbers decide whether a run may reach the robot, and neither is a training
loss. On held-out leave-alone frames the correction has to stay near zero, because
one that fires during normal operation degrades a baseline that is already the
best policy here -- and the larger the clip, the more a false positive costs. On
held-out corrections it has to point the way the human actually moved. A head that
scores well on the first and at chance on the second has learned to output zero,
which the zero-initialised last layer already does for free.

Usage:
    python train.py --cache-dir ... --segments ... --out ... [--clip 0.1,0.3,0.8]
"""

import argparse
from pathlib import Path

import dataset as ds
import numpy as np
import torch
from torch import nn

from model import ResidualHead


def cosine(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = pred.reshape(len(pred), -1)
    t = true.reshape(len(true), -1)
    return (p * t).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", type=int, default=7, help="rescues held out")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--clip", default="0.1", help="comma-separated bounds to sweep")
    ap.add_argument("--correction-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    data = ds.build(Path(a.cache_dir), Path(a.segments))
    ds.report(data)

    rng = np.random.default_rng(a.seed)
    rescues = np.array(sorted({g for g, c in zip(data["group"], data["is_correction"]) if c}))
    held = set(rng.permutation(rescues)[: a.holdout].tolist())
    te = np.array([g in held for g in data["group"]])
    tr = ~te
    print(
        f"\n{len(set(data['group']))} groups, {len(rescues)} contain corrections; "
        f"holding out {len(held)}"
    )
    print(
        f"train {tr.sum()} frames ({data['is_correction'][tr].sum()} corrections), "
        f"test {te.sum()} ({data['is_correction'][te].sum()} corrections)"
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def T(x):
        return torch.from_numpy(np.ascontiguousarray(x))

    feats, state = T(data["feats"]), T(data["state"])
    a_base, delta = T(data["a_base"]), T(data["delta"])
    corr = T(data["is_correction"].astype(np.float32))
    idx_tr, idx_te = np.where(tr)[0], np.where(te)[0]
    steps = a.epochs * max(1, len(idx_tr) // a.batch)

    def run(idx, train, net, opt, sched):
        net.train(train)
        tot = n = 0.0
        preds = []
        for i in range(0, len(idx), a.batch):
            j = idx[i : i + a.batch]
            f = feats[j].to(dev, torch.float32)
            s, ab, y = state[j].to(dev), a_base[j].to(dev), delta[j].to(dev)
            w = 1.0 + (a.correction_weight - 1.0) * corr[j].to(dev)
            with torch.set_grad_enabled(train):
                p = net(f, s, ab)
                loss = (((p - y) ** 2).mean(dim=(1, 2)) * w).mean()
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
            else:
                preds.append(p.detach().cpu().numpy())
            tot += float(loss.detach()) * len(j)
            n += len(j)
        return tot / max(n, 1), (np.concatenate(preds) if preds else None)

    for clip in [float(x) for x in str(a.clip).split(",")]:
        print(f"\n=== clip {clip:g} rad ===", flush=True)
        net = ResidualHead(
            feats.shape[-1],
            state.shape[-1],
            delta.shape[1],
            delta.shape[2],
            width=a.width,
            clip=clip,
        ).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for ep in range(a.epochs):
            rng.shuffle(idx_tr)
            tl, _ = run(idx_tr, True, net, opt, sched)
            if ep % 10 == 9 or ep == a.epochs - 1:
                _, p = run(idx_te, False, net, opt, sched)
                c = data["is_correction"][idx_te]
                quiet = np.abs(p[~c]).max(axis=(1, 2))
                cs = (
                    cosine(p[c][:, :, :14], data["delta"][idx_te][c][:, :, :14])
                    if c.any()
                    else np.array([np.nan])
                )
                got = np.median(np.abs(p[c][:, 6:22, :14]).max(axis=(1, 2)))
                want = np.median(np.abs(data["delta"][idx_te][c][:, 6:22, :14]).max(axis=(1, 2)))
                print(
                    f"epoch {ep + 1:3d}  train {tl:.5f}  |  leave-alone |delta| p50 "
                    f"{np.median(quiet):.4f} p90 {np.quantile(quiet, 0.9):.4f}  |  cosine p50 "
                    f"{np.median(cs):+.3f} >0 {100 * np.mean(cs > 0):.0f}%  |  exec magnitude "
                    f"{got:.3f} of {want:.3f} rad",
                    flush=True,
                )
        out = Path(a.out)
        out = out.with_name(f"{out.stem}_clip{clip:g}{out.suffix}")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": net.state_dict(),
                "config": {
                    "token_dim": feats.shape[-1],
                    "state_dim": state.shape[-1],
                    "horizon": delta.shape[1],
                    "action_dim": delta.shape[2],
                    "width": a.width,
                    "clip": clip,
                },
                "holdout": sorted(held),
            },
            out,
        )
        print(f"saved {out}")


if __name__ == "__main__":
    main()
