"""Where is the ~27 mm wall? The kernel read-out of fit_res.py, at every depth
probe_vit_vs_llm.py saved.

llm_L16_all is the previous probe's feature and must reproduce it (wrist 26.9 /
hole 28.7) before any other row is read.

Reading fixed before the numbers: the wrist is the positive control, because it is
plainly visible. If the ViT rows locate it clearly better than llm_L16_all (~20 mm
or less), the LLM is losing detail the ViT has. If they also sit near 27 mm, the
wall is at or before the ViT output, which is what tune_visual can move.
"""

from pathlib import Path
import sys
import time

import numpy as np


D = Path(sys.argv[1])
ALPHAS = np.logspace(-6, 4, 21)
FEATURES = [
    "llm_L16_all",
    "llm_L16_img",
    "llm_L12_img",
    "llm_L8_img",
    "llm_L4_img",
    "vit_merged",
    "vit_b24",
    "vit_b12",
]


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


metas = [np.load(p) for p in sorted(D.glob("meta_s*.npz"))]
Y = np.concatenate([m["Y"] for m in metas])
CUR = np.concatenate([m["CUR"] for m in metas])
G = np.concatenate([m["G"] for m in metas])
N = len(Y)
rs = np.random.default_rng(0)
eps = np.array(sorted(set(G)))
perm = rs.permutation(eps)
fold = {e: i % 10 for i, e in enumerate(perm)}
F = np.array([fold[g] for g in G])
chance = np.zeros(N)
for f in range(10):
    te, tr = F == f, F != f
    chance[te] = np.linalg.norm(CUR[te] - CUR[tr].mean(0), axis=1)
print(f"{N} samples / {len(eps)} episodes; wrist chance {np.median(chance):.1f} mm", flush=True)

rows = []
for name in FEATURES:
    parts = [np.load(p) for p in sorted(D.glob(f"{name}_s*.npy"))]
    n = min(x.shape[1] for x in parts)
    X = np.concatenate([x[:, :n] for x in parts])
    shape = X.shape[1:]
    X = X.reshape(N, -1).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True).mean()
    t0 = time.time()
    K = (X @ X.T).astype(np.float64)
    del X, parts
    w = np.median(krr_oof(K, CUR, F, G, fold))
    h = np.median(krr_oof(K, Y, F, G, fold))
    rows.append((name, shape, w, h))
    print(
        f"{name:>12} {str(shape):>14}  wrist {w:5.1f}  hole {h:5.1f}  ({time.time() - t0:.0f}s)",
        flush=True,
    )

print(
    f"\n{'feature':>12} {'tokens x dim':>14} {'wrist (control)':>16} {'hole':>7}   (mm, median OOF)"
)
for name, shape, w, h in rows:
    print(f"{name:>12} {str(shape):>14} {w:16.1f} {h:7.1f}")
