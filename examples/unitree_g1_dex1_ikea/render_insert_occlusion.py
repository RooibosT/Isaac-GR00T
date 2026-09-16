"""Read the occ_s*.npz shards of probe_insert_occlusion.py.

Usage: python render_insert_occlusion.py <occlusion out-dir> [<attention out-dir with attn.npz>]
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


VIEWS = ["cam_left_high", "cam_left_wrist", "cam_right_wrist"]
TW, TH = 352, 256
ROW_NAMES = ("row12", "row39")


def heat(x):
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def label(im, text):
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, TW, 16], fill=(0, 0, 0))
    dr.text((4, 2), text, fill=(255, 255, 255))
    return im


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main() -> None:
    d = Path(sys.argv[1])
    parts = [np.load(p) for p in sorted(d.glob("occ_s*.npz"))]
    z = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    ep, off = z["ep"], z["off"]
    cells, views, allv, noise = (
        z["cells"],
        z["views"],
        z["allv"],
        z["noise"],
    )  # (N,3,8,11,R) (N,3,R) (N,R) (N,R)
    n = len(ep)
    lines = [
        f"{n} frames / {len(set(ep.tolist()))} episodes; displacement of the FK right wrist, mm, mean of seeds",
        "",
    ]
    for r, rn in enumerate(ROW_NAMES):
        lines.append(f"== {rn}: median over episodes")
        lines.append(
            f"{'off':>5} {'noise':>6} {'all views':>9} | "
            + " | ".join(f"{v[:15]:>15} view / cellmax / cellsum" for v in VIEWS)
        )
        for o in sorted(set(off.tolist())):
            m = off == o
            lines.append(
                f"{o:>5} {np.median(noise[m, r]):6.1f} {np.median(allv[m, r]):9.1f} | "
                + " | ".join(
                    f"{np.median(views[m, v, r]):15.1f} / {np.median(cells[m, v, ..., r].max(axis=(1, 2))):6.1f} / "
                    f"{np.median(cells[m, v, ..., r].sum(axis=(1, 2))):6.1f}"
                    for v in range(3)
                )
            )
        top_view = cells[..., r].reshape(n, 3, -1).max(-1).argmax(-1)
        lines.append(
            "view holding the single most sensitive cell: "
            + ", ".join(f"{VIEWS[v]} {np.mean(top_view == v) * 100:.0f}%" for v in range(3))
        )
        above = (cells[..., r] > noise[:, r][:, None, None, None]).reshape(n, 3, -1).mean(-1)
        lines.append(
            "cells whose effect exceeds that frame's noise floor: "
            + ", ".join(f"{VIEWS[v]} {np.mean(above[:, v]) * 100:.0f}%" for v in range(3))
        )
        lines.append("")

    if len(sys.argv) > 2:
        za = np.load(Path(sys.argv[2]) / "attn.npz")
        idx = {(int(e), int(o)): i for i, (e, o) in enumerate(zip(za["ep"], za["off"]))}
        grids = za["grids"]
        within = grids / grids.sum(axis=(2, 3), keepdims=True)
        ratio = within / within.mean(0)[None]
        rho_raw, rho_ratio = [[], [], []], [[], [], []]
        for i in range(n):
            j = idx.get((int(ep[i]), int(off[i])))
            if j is None:
                continue
            for v in range(3):
                s = cells[i, v, ..., 1].ravel()
                rho_raw[v].append(spearman(within[j, v].ravel(), s))
                rho_ratio[v].append(spearman(ratio[j, v].ravel(), s))
        lines.append("attention vs row39 sensitivity, Spearman per frame, median:")
        for v in range(3):
            lines.append(
                f"{VIEWS[v]:>16}: raw attention {np.median(rho_raw[v]):+.2f}   attention/bias {np.median(rho_ratio[v]):+.2f}"
            )
    text = "\n".join(lines)
    print(text)
    (d / "stats.txt").write_text(text + "\n")

    images = z["images"]
    for e in sorted(set(ep.tolist())):
        ids = sorted([i for i in range(n) if ep[i] == e], key=lambda i: off[i])
        sh = Image.new("RGB", (TW * 6, TH * len(ids)))
        for r_i, i in enumerate(ids):
            s = cells[i, ..., 1]
            scale = max(s.max(), 1e-6)
            for v in range(3):
                raw = Image.fromarray(images[i, v]).resize((TW, TH), Image.BILINEAR)
                sh.paste(
                    label(raw, f"{VIEWS[v]} t{off[i]:+d}  whole view {views[i, v, 1]:.1f} mm"),
                    (v * 2 * TW, r_i * TH),
                )
                cellimg = Image.fromarray(heat(s[v] / scale)).resize((TW, TH), Image.NEAREST)
                ov = Image.blend(raw, cellimg, 0.5)
                sh.paste(
                    label(
                        ov,
                        f"cell max {s[v].max():.1f} mm (frame max {scale:.1f}, noise {noise[i, 1]:.1f})",
                    ),
                    ((v * 2 + 1) * TW, r_i * TH),
                )
        sh.save(d / f"occ_ep{e:03d}.png")
    print(f"wrote occ_ep*.png and stats.txt to {d}")


if __name__ == "__main__":
    main()
