"""Same kernel read-out as the 27 mm baseline, one row per zoom setting.

`base` must reproduce the earlier numbers -- wrist 26.9 mm, hole 28.7 -- before any
other row means anything. The wrist is the positive control: it is plainly visible, so
if a crop sharpened the features at all, the wrist read-out moves first.

Decided before the numbers: promote a setting only if the wrist drops to about 20 mm or
less. A 3-11% move is what raising the input to 384 or adding a fourth view already did,
and that was not enough to promote either.

Usage: python fit_zoom_crop.py <out-dir of probe_zoom_crop.py> [setting ...]
"""

from pathlib import Path
import sys
import time

import numpy as np


ALPHAS = np.logspace(-6, 4, 21)


def center(K, tr, te):
    Ktr = K[np.ix_(tr, tr)]
    Kte = K[np.ix_(te, tr)]
    m = Ktr.mean(0)
    mm = Ktr.mean()
    return (
        Ktr - m[None, :] - Ktr.mean(1)[:, None] + mm,
        Kte - m[None, :] - Kte.mean(1)[:, None] + mm,
    )


def krr_oof(K, T, F, G, fold):
    P = np.zeros_like(T)
    for f in range(10):
        te = np.where(F == f)[0]
        tr = np.where(F != f)[0]
        A, B = center(K, tr, te)
        ym = T[tr].mean(0)
        Yt = T[tr] - ym
        inner = np.array([fold[g] % 5 for g in G[tr]])
        best, ba = np.inf, ALPHAS[0]
        for al in ALPHAS:
            err = 0.0
            for g in range(5):
                i2 = np.where(inner != g)[0]
                j2 = np.where(inner == g)[0]
                Aa = A[np.ix_(i2, i2)]
                c = np.linalg.solve(Aa + al * np.trace(Aa) / len(i2) * np.eye(len(i2)), Yt[i2])
                err += np.linalg.norm(A[np.ix_(j2, i2)] @ c - Yt[j2], axis=1).sum()
            if err < best:
                best, ba = err, al
        c = np.linalg.solve(A + ba * np.trace(A) / len(tr) * np.eye(len(tr)), Yt)
        P[te] = B @ c + ym
    return np.linalg.norm(P - T, axis=1)


def main() -> None:
    d = Path(sys.argv[1])
    metas = [np.load(p) for p in sorted(d.glob("meta_s*.npz"))]
    Y = np.concatenate([m["Y"] for m in metas])
    CUR = np.concatenate([m["CUR"] for m in metas])
    G = np.concatenate([m["G"] for m in metas])
    n = len(Y)
    rs = np.random.default_rng(0)
    eps = np.array(sorted(set(G.tolist())))
    perm = rs.permutation(eps)
    fold = {e: i % 10 for i, e in enumerate(perm)}
    F = np.array([fold[g] for g in G])
    chance = np.zeros(n)
    for f in range(10):
        te, tr = F == f, F != f
        chance[te] = np.linalg.norm(CUR[te] - CUR[tr].mean(0), axis=1)

    names = sys.argv[2:] or sorted({p.name.split("_s")[0] for p in d.glob("*_s*.npy")})
    print(
        f"{n} samples / {len(eps)} episodes; wrist chance {np.median(chance):.1f} mm\n", flush=True
    )
    print(
        f"{'setting':>14} {'tokens':>7} {'wrist (control)':>16} {'hole':>8}   (mm, median out-of-fold)"
    )
    rows = []
    for name in names:
        parts = [np.load(p) for p in sorted(d.glob(f"{name}_s*.npy"))]
        if not parts:
            continue
        k = min(x.shape[1] for x in parts)
        X = np.concatenate([x[:, :k] for x in parts]).reshape(n, -1).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True).mean()
        t0 = time.time()
        K = (X @ X.T).astype(np.float64)
        del X, parts
        w = float(np.median(krr_oof(K, CUR, F, G, fold)))
        h = float(np.median(krr_oof(K, Y, F, G, fold)))
        rows.append((name, k, w, h))
        print(f"{name:>14} {k:>7} {w:16.1f} {h:8.1f}   ({time.time() - t0:.0f}s)", flush=True)

    if rows and rows[0][0] == "base":
        bw, bh = rows[0][2], rows[0][3]
        print(f"\nvs base ({bw:.1f} / {bh:.1f} mm)")
        for name, k, w, h in rows[1:]:
            verdict = "promote" if w <= 20.0 else "not enough"
            print(
                f"{name:>14} wrist {(w - bw) / bw * 100:+6.1f}%   hole {(h - bh) / bh * 100:+6.1f}%   -> {verdict}"
            )
    (d / "stats.txt").write_text(
        "\n".join(f"{r[0]} tokens={r[1]} wrist={r[2]:.1f} hole={r[3]:.1f}" for r in rows) + "\n"
    )


if __name__ == "__main__":
    main()
