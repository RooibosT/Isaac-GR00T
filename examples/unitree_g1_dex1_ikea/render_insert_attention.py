"""Read attn.npz from probe_insert_attention.py with the fixed position bias taken out.

Raw attention maps from the smoke run lit the same border and corner tokens in every
frame -- a position bias, not a response to content. So each view's map is first
normalised to sum to 1, the mean over every frame and episode is taken as that bias,
and each frame is shown as its ratio to it: a cell at 2x draws twice the attention
that token usually gets. Only a ratio map can say "this frame looks HERE".

Usage: python render_insert_attention.py <out-dir of probe_insert_attention.py>
"""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


VIEWS = ["cam_left_high", "cam_left_wrist", "cam_right_wrist"]
TW, TH = 352, 256


def heat(x):
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def overlay(img, x, alpha):
    base = Image.fromarray(img).resize((TW, TH), Image.BILINEAR)
    cells = Image.fromarray(heat(x)).resize((TW, TH), Image.NEAREST)
    return Image.blend(base, cells, alpha)


def label(im, text):
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, TW, 16], fill=(0, 0, 0))
    dr.text((4, 2), text, fill=(255, 255, 255))
    return im


def main() -> None:
    d = Path(sys.argv[1])
    z = np.load(d / "attn.npz")
    ep, off, images, grids = z["ep"], z["off"], z["images"], z["grids"]
    n, nv, gh, gw = grids.shape
    ntok = gh * gw
    share = grids.sum(axis=(2, 3)) / grids.sum(axis=(1, 2, 3))[:, None]
    within = grids / grids.sum(axis=(2, 3), keepdims=True)
    bias = within.mean(0)
    ratio = within / bias[None]

    peak = within.max(axis=(2, 3)) * ntok
    entropy = -(within * np.log(within)).sum(axis=(2, 3)) / np.log(ntok)
    corr = np.array(
        [
            [np.corrcoef(within[i, v].ravel(), bias[v].ravel())[0, 1] for v in range(nv)]
            for i in range(n)
        ]
    )
    rmax = ratio.max(axis=(2, 3))

    lines = [f"{n} frames / {len(set(ep.tolist()))} episodes, {gh}x{gw} tokens per view", ""]
    lines.append(
        f"{'view':>16} {'share':>7} {'peak/mean':>10} {'entropy':>8} {'corr w/ bias':>13} {'max ratio':>10}"
    )
    for v in range(nv):
        lines.append(
            f"{VIEWS[v]:>16} {share[:, v].mean() * 100:6.1f}% {np.median(peak[:, v]):10.2f} "
            f"{np.median(entropy[:, v]):8.3f} {np.median(corr[:, v]):13.2f} {np.median(rmax[:, v]):10.2f}"
        )
    lines += ["", "bias: top 5 tokens per view (row, col, x uniform)"]
    for v in range(nv):
        top = np.argsort(bias[v].ravel())[::-1][:5]
        lines.append(
            f"{VIEWS[v]:>16} "
            + "  ".join(f"({t // gw},{t % gw}) x{bias[v].ravel()[t] * ntok:.2f}" for t in top)
        )
    lines += ["", "by offset (mean over episodes): view share % | right-wrist max ratio"]
    for o in sorted(set(off.tolist())):
        m = off == o
        lines.append(
            f"{o:>6} "
            + " ".join(f"{share[m, v].mean() * 100:5.1f}" for v in range(nv))
            + f" | {rmax[m, 2].mean():.2f}"
        )
    text = "\n".join(lines)
    print(text)
    (d / "stats.txt").write_text(text + "\n")

    first = {v: images[0, v] for v in range(nv)}
    sheet = Image.new("RGB", (TW * nv, TH))
    for v in range(nv):
        t = overlay(first[v], bias[v] / bias[v].max(), 0.5)
        sheet.paste(
            label(t, f"{VIEWS[v]} BIAS (mean of {n})  peak x{bias[v].max() * ntok:.2f}"),
            (v * TW, 0),
        )
    sheet.save(d / "bias.png")

    for e in sorted(set(ep.tolist())):
        idx = [i for i in range(n) if ep[i] == e]
        idx.sort(key=lambda i: off[i])
        sh = Image.new("RGB", (TW * nv * 2, TH * len(idx)))
        for r, i in enumerate(idx):
            for v in range(nv):
                raw = Image.fromarray(images[i, v]).resize((TW, TH), Image.BILINEAR)
                sh.paste(
                    label(raw, f"{VIEWS[v]} t{off[i]:+d}  share {share[i, v] * 100:.0f}%"),
                    (v * 2 * TW, r * TH),
                )
                x = np.clip(
                    np.log2(ratio[i, v]), 0, 1
                )  # 0 = at or below its usual weight, 1 = 2x or more
                ov = label(overlay(images[i, v], x, 0.5), f"vs bias: max x{rmax[i, v]:.2f}")
                sh.paste(ov, ((v * 2 + 1) * TW, r * TH))
        sh.save(d / f"ratio_ep{e:03d}.png")
    (d / "stats.json").write_text(
        json.dumps(
            {
                "share": share.tolist(),
                "peak": peak.tolist(),
                "entropy": entropy.tolist(),
                "corr_bias": corr.tolist(),
                "max_ratio": rmax.tolist(),
                "ep": ep.tolist(),
                "off": off.tolist(),
            }
        )
    )
    print(f"wrote bias.png, ratio_ep*.png, stats.txt to {d}")


if __name__ == "__main__":
    main()
