# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read the pickles `probe_insert_release.py` writes.

Everything is judged against the rows the robot actually runs. A chunk is
installed at its observation's timestamp, GR00T answers in ~5 ticks and the next
request goes out at row 8, so rows [5, 13) of each chunk reach the arm. An
opening at row >= 13 is one the robot never executes unless a later chunk brings
it forward.

  open   first row where the right gripper crosses the episode's midpoint
  press  right-wrist height of the commanded rows [5, 13) minus the observed
         wrist, in mm. The leg does not drop because the command drops: the
         demonstrator holds the command ~30 mm below the arm while the leg sits
         on the hole edge, and the arm catches up when it aligns. `clip` is the
         same press after the controller's lead clamp, which holds every
         published command within 0.04 rad (arm_velocity_limit_rad_s 10 x 1/250
         s) of the measured arm, scaled by the worst of the 14 joints
         (xr_teleoperate robot_arm.clip_arm_q_target). Teleop ran at 0.12.
  gates  the deploy's chunk checks (validate_action_chunk): first arm row
         within 0.45 rad of the observed arm, no arm step over 0.20 rad, no
         gripper step over 0.80. A failure drops the whole chunk and holds.
"""

import argparse
import os
from pathlib import Path
import pickle
import sys

import numpy as np


URL_LEROBOT = Path(
    os.environ.get("URL_LEROBOT") or Path(__file__).resolve().parents[3] / "url_lerobot"
)
sys.path.insert(0, str(URL_LEROBOT))
from url_groot_deploy.common.g1_kinematics import G1WristKinematics  # noqa: E402


EXEC_FROM, EXEC_TO = 5, 13
FIRST_ERR, ARM_STEP, HAND_STEP = 0.45, 0.20, 0.80
DEPLOY_LEAD = 0.04

KIN = G1WristKinematics(str(URL_LEROBOT / "xr_teleoperate"), waist_zero=True)


def right_z(arm: np.ndarray) -> np.ndarray:
    """(..., 14) joint rows -> right wrist z in mm."""
    flat = arm.reshape(-1, 14).astype(np.float64)
    z = np.array([KIN.both_wrist_poses(q, np.zeros(3))[1][2] for q in flat])
    return z.reshape(arm.shape[:-1]) * 1000.0


def clip(cmd: np.ndarray, meas: np.ndarray, lead: float) -> np.ndarray:
    """clip_arm_q_target: scale the whole 14-joint delta by its worst joint."""
    d = cmd - meas
    scale = np.maximum(np.abs(d).max(-1, keepdims=True) / lead, 1.0)
    return meas + d / scale


def first(mask: np.ndarray) -> np.ndarray:
    """(..., H) bool -> first True index, or H when there is none."""
    h = mask.shape[-1]
    return np.where(mask.any(-1), mask.argmax(-1), h)


def opens(chunk: np.ndarray, thresh: float) -> np.ndarray:
    return first(chunk[..., 15] >= thresh)


def press(chunk: np.ndarray, meas: np.ndarray):
    """(S, H, 16) -> median raw and deploy-clipped press over rows [5, 13), mm."""
    rows = chunk[:, EXEC_FROM:EXEC_TO, :14]
    z0 = right_z(meas[None])[0]
    raw = np.median(right_z(rows) - z0, axis=1)
    clipped = np.median(right_z(clip(rows, meas, DEPLOY_LEAD)) - z0, axis=1)
    return raw, clipped


def gates(chunk: np.ndarray, arm_now: np.ndarray) -> np.ndarray:
    first_err = np.abs(chunk[:, 0, :14] - arm_now[None]).max(-1) > FIRST_ERR
    arm_step = np.abs(np.diff(chunk[..., :14], axis=1)).max((1, 2)) > ARM_STEP
    hand_step = np.abs(np.diff(chunk[..., 14:16], axis=1)).max((1, 2)) > HAND_STEP
    return np.stack([first_err, arm_step, hand_step], -1)


def med(a, h):
    return float(np.median(a[a < h])) if (a < h).any() else np.nan


def fmt(v, w=6):
    return (
        " " * (w - 1) + "-"
        if v is None or (isinstance(v, float) and np.isnan(v))
        else f"{v:{w}.1f}"
    )


def dense_report(eps, H):
    print("\n== dense: plain sampling along the demonstration, by frames to the drop ==")
    print("'run' = opening at a row < 13, i.e. inside what this chunk hands the robot.")
    print("If the robot stopped progressing at that frame, every replan would draw")
    print("from this row again, independently.")
    print(
        f"{'to drop':>9} {'n':>4} | {'GT open':>7} {'pred open':>9} {'P(run) GT/pred %':>17} "
        f"{'no open %':>9} | {'press GT':>8} {'pred':>6} {'pred clip':>9} | {'reject %':>8}"
    )
    by = {}
    for e in eps:
        for q in e["dense"]:
            t = q["t"]
            meas = e["arm_state"][t]
            gt = e["gt"][None, t : t + H]
            key = int(np.floor((t - e["drop"]) / 6) * 6)
            by.setdefault(key, []).append(
                (
                    opens(q["pred"], e["thresh"]),
                    opens(gt, e["thresh"])[0],
                    press(q["pred"], meas),
                    press(gt, meas)[0][0],
                    gates(q["pred"], meas),
                )
            )
    for key in sorted(by):
        if not -66 <= key <= 18:
            continue
        v = by[key]
        po = np.concatenate([x[0] for x in v])
        go = np.array([x[1] for x in v])
        pr = np.concatenate([x[2][0] for x in v])
        pc = np.concatenate([x[2][1] for x in v])
        gp = np.array([x[3] for x in v])
        g = np.concatenate([x[4] for x in v])
        print(
            f"{f'{key:+d}..{key + 5:+d}':>9} {len(v):4d} | {fmt(med(go, H), 7)} {fmt(med(po, H), 9)} "
            f"{np.mean(go < EXEC_TO) * 100:9.0f}/{np.mean(po < EXEC_TO) * 100:3.0f}    "
            f"{np.mean(po >= H) * 100:9.0f} | {np.median(gp):8.1f} {np.median(pr):6.1f} {np.median(pc):9.1f} | "
            f"{np.mean(g.any(-1)) * 100:7.1f}%"
        )


def swap_report(eps, H):
    print("\n== swap: state from one frame, images from another (frames to release) ==")
    offs = sorted({q["t_state"] - e["release"] for e in eps for q in e["swap"]})
    drop_at = int(np.median([e["drop"] - e["release"] for e in eps]))
    print(f"(the drop starts ~{-drop_at} frames before the release)")
    cells = {}
    for e in eps:
        for q in e["swap"]:
            k = (q["t_state"] - e["release"], q["t_img"] - e["release"])
            cells.setdefault(k, []).append(opens(q["pred"], e["thresh"]))
    for name, f in (
        ("P(open run) %", lambda v: np.mean(v < EXEC_TO) * 100),
        ("median open row", lambda v: med(v, H)),
    ):
        print(f"\n{name:20s} images ->  " + " ".join(f"{o:+6d}" for o in offs))
        for os_ in offs:
            row = [f(np.concatenate(cells[(os_, oi)])) for oi in offs]
            print(
                f"{'state ' + format(os_, '+d'):>20s}              " + " ".join(fmt(c) for c in row)
            )


def rtc_report(eps, H):
    print("\n== rtc follow: chained along the demonstration, plain vs RTC ==")
    print("rows [5,13) of each chunk stitched; delay = first commanded open frame minus")
    print("the demonstrated one. Rejections by frames to the release.")
    for kind in ("plain", "rtc"):
        delays, rej = [], {}
        for e in eps:
            chain = [c for c in e["rtc"] if c["kind"] == "follow"]
            g = e["gt"][:, 15]
            demo = next(i for i in range(e["release"] - 30, len(g)) if g[i] >= e["thresh"])
            for s in range(chain[0][kind].shape[0]):
                cmd = None
                for c in chain:
                    hit = np.where(c[kind][s, EXEC_FROM:EXEC_TO, 15] >= e["thresh"])[0]
                    if len(hit):
                        cmd = c["t"] + EXEC_FROM + int(hit[0])
                        break
                delays.append(np.nan if cmd is None else cmd - demo)
            for c in chain:
                rej.setdefault(c["t"] - e["release"], []).append(
                    gates(c[kind], e["arm_state"][c["t"]])
                )
        d = np.array(delays, float)
        allr = np.concatenate([np.concatenate(v) for v in rej.values()])
        print(
            f"\n  {kind}: open commanded in {np.mean(~np.isnan(d)) * 100:.0f}% of chains; delay "
            f"median {np.nanmedian(d):+.0f}, p10 {np.nanpercentile(d, 10):+.0f}, p90 "
            f"{np.nanpercentile(d, 90):+.0f} frames; early (< -8) {np.mean(d < -8) * 100:.0f}%; "
            f"rejected {np.mean(allr.any(-1)) * 100:.1f}% "
            f"(first {allr[:, 0].mean() * 100:.1f} / arm {allr[:, 1].mean() * 100:.1f} / "
            f"hand {allr[:, 2].mean() * 100:.1f})"
        )
        print("    to release: " + " ".join(f"{k:+5d}" for k in sorted(rej)))
        print(
            "    rejected %: "
            + " ".join(f"{np.mean(np.concatenate(rej[k]).any(-1)) * 100:5.0f}" for k in sorted(rej))
        )

    print("\n== rtc hold: observation frozen at a pre-drop frame, RTC-chained replans ==")
    print("a stand-in for a leg that stays on the hole edge. cum open = share of chains")
    print("whose executed rows have commanded the release by that replan. The frozen")
    print("gripper state makes rejections after an opening an artifact of the hold.")
    kinds = sorted(
        {c["kind"] for e in eps for c in e["rtc"] if c["kind"].startswith("hold")},
        key=lambda k: int(k[4:]),
    )
    for kind in kinds:
        per_k = {}
        for e in eps:
            chain = [c for c in e["rtc"] if c["kind"] == kind]
            if not chain:
                continue
            t = chain[0]["t"]
            meas = e["arm_state"][t]
            opened = np.zeros(chain[0]["rtc"].shape[0], bool)
            for c in chain:
                ch = c["rtc"]
                opened |= (ch[:, EXEC_FROM:EXEC_TO, 15] >= e["thresh"]).any(-1)
                per_k.setdefault(c["k"], []).append(
                    (opened.copy(), opens(ch, e["thresh"]), press(ch, meas), gates(ch, meas))
                )
        gt_rel = int(np.median([e["release"] - (e["drop"] + int(kind[4:])) for e in eps]))
        print(f"\n  {kind}  (the demonstrated release comes ~{gt_rel} frames after this frame)")
        print(
            f"  {'replan':>6} {'cum open %':>10} {'open row':>8} {'no open %':>9} "
            f"{'press':>6} {'clip':>6} {'rejected % (first/arm/hand)':>28}"
        )
        for k in sorted(per_k):
            v = per_k[k]
            o = np.concatenate([x[0] for x in v])
            po = np.concatenate([x[1] for x in v])
            pr = np.concatenate([x[2][0] for x in v])
            pc = np.concatenate([x[2][1] for x in v])
            rj = np.concatenate([x[3] for x in v])
            print(
                f"  {k:6d} {o.mean() * 100:10.0f} {fmt(med(po, H), 8)} {np.mean(po >= H) * 100:9.0f} "
                f"{np.median(pr):6.1f} {np.median(pc):6.1f} {rj.any(-1).mean() * 100:9.1f} "
                f"({rj[:, 0].mean() * 100:.0f}/{rj[:, 1].mean() * 100:.0f}/{rj[:, 2].mean() * 100:.0f})"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pickles", nargs="+", type=Path)
    a = ap.parse_args()
    for p in a.pickles:
        d = pickle.loads(p.read_bytes())
        eps, H = d["episodes"], d["horizon"]
        print(f"\n######## {d['checkpoint']}  ({len(eps)} episodes)")
        if eps and "dense" in eps[0]:
            dense_report(eps, H)
        if eps and "swap" in eps[0]:
            swap_report(eps, H)
        if eps and "rtc" in eps[0]:
            rtc_report(eps, H)


if __name__ == "__main__":
    main()
