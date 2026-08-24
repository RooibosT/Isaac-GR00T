# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cut carroll511/lightwheel_lerobot (sim, LeRobot v3.0) into v2.1 clips GR00T can mix in.

Unlike the IKEA converter this is not a pure re-layout. The source is 114 very
long episodes (median 226 s) that walk to the table, carry parts around and
assemble; the part we can use is the subset where the robot is *standing still
and manipulating*, because the real recordings have no locomotion at all and the
policy we deploy never commands the base.

So each source episode is split into clips that satisfy all of:

* the frame's task is one of ``--tasks`` (default: the two the real dataset is
  weakest on, ``insert table leg to table base`` and ``rotate leg to tighten``),
* the commanded base velocity is below ``--still-eps`` on both the linear and the
  yaw channel, and
* the run is at least ``--min-frames`` long, which has to exceed the action
  horizon or the clip yields no training window at all.

Measured on the full dataset, that keeps 86.9% of those two tasks' frames --
the base command is genuinely idle for most of the assembly, the walking is
concentrated in ``move to table`` and ``move table base``.

Two things are deliberately *not* changed, because the mixture loader gives each
dataset its own embodiment tag and therefore its own state/action projection:

* **The 33-dim state and 23-dim action are copied verbatim.** No attempt is made
  to reshape them into the real robot's 46-dim state or 16-dim joint action.
  They mean different things (the sim action is end-effector pose plus base
  command, in the competition's `decoupled` layout) and forcing them into one
  vector would be a silent lie about what the numbers are.
* **20 fps is left alone.** The horizon is counted in frames, so a 40-step chunk
  is 2.0 s here against 1.33 s on the real robot. That is a real difference to
  keep in mind when reading results, but resampling video and actions to 30 fps
  would invent frames that were never rendered.

Video is re-encoded rather than stream-copied: cutting on a non-keyframe is what
makes a clip start one frame away from its state row, and `-ss` before `-i` plus
`-frames:v` is frame-exact. `--verify` decodes sampled frames out of the cut
clips and compares them against the same frames of the source, which is the only
real check that the arithmetic above is right.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


CAMERAS = (
    "observation.images.first_person_camera_rgb",
    "observation.images.left_hand_camera_rgb",
    "observation.images.right_hand_camera_rgb",
)
CHUNK_SIZE = 1000
DEFAULT_TASKS = ("insert table leg to table base", "rotate leg to tighten")


def ffmpeg_bin(name: str) -> str:
    """torchcodec's FFmpeg is not on PATH in this environment; prefer the env's."""
    local = Path.home() / "micromamba/envs/ffmpeg7/bin" / name
    return str(local) if local.exists() else name


def find_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs at least `min_len` long."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def cut_clip(src: Path, first_frame: int, n_frames: int, dst: Path, fps: int, crf: int) -> None:
    """Frame-exact cut of `n_frames` starting at `first_frame`.

    Seeking a quarter frame early and letting `-frames:v` count forward keeps the
    first decoded frame the intended one even when the seek lands mid-GOP.
    """
    seek = max(0.0, (first_frame - 0.25) / fps)
    cmd = [
        ffmpeg_bin("ffmpeg"),
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{seek:.6f}",
        "-i",
        str(src),
        "-frames:v",
        str(n_frames),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "2",
        "-keyint_min",
        "2",
        "-sc_threshold",
        "0",
        "-an",
        "-fps_mode",
        "passthrough",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def decode_frames(path: Path, indices: list[int]) -> np.ndarray:
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(str(path), seek_mode="exact")
    return np.stack([dec[i].permute(1, 2, 0).numpy() for i in indices])


def build_clips(src: Path, tasks: tuple[str, ...], still_eps: float, min_frames: int):
    """(episode_index, start, stop, task) for every run worth keeping."""
    eps = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")
    task_tbl = pd.read_parquet(src / "meta/tasks.parquet").reset_index()
    name_of = {int(r.task_index): r.task for r in task_tbl.itertuples()}
    wanted = {i for i, n in name_of.items() if n in tasks}
    if not wanted:
        raise SystemExit(f"none of {tasks} exist in this dataset: {sorted(name_of.values())}")

    frame = pd.concat(
        [pd.read_parquet(p) for p in sorted((src / "data").glob("chunk-*/*.parquet"))],
        ignore_index=True,
    )
    action = np.stack(frame["action"]).astype(np.float32)
    # action[16:19] is [base_linear_velocity_x, _y, base_angular_velocity_z]
    still = (np.hypot(action[:, 16], action[:, 17]) < still_eps) & (
        np.abs(action[:, 18]) < still_eps
    )
    task_index = frame["task_index"].to_numpy()

    clips = []
    for row in eps.itertuples():
        lo, hi = int(row.dataset_from_index), int(row.dataset_to_index)
        keep = still[lo:hi] & np.isin(task_index[lo:hi], list(wanted))
        for a, b in find_runs(keep, min_frames):
            seg = task_index[lo + a : lo + b]
            # a run can straddle a task boundary; label it by what it mostly is
            label = name_of[int(np.bincount(seg).argmax())]
            clips.append((int(row.episode_index), a, b, label))
    return frame, eps, clips


def write_dataset(
    src: Path,
    out: Path,
    frame: pd.DataFrame,
    eps: pd.DataFrame,
    clips: list,
    fps: int,
    crf: int,
    workers: int,
    verify: int,
) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    (out / "data/chunk-000").mkdir(parents=True)
    for cam in CAMERAS:
        (out / f"videos/chunk-000/{cam}").mkdir(parents=True)

    src_info = json.loads((src / "meta/info.json").read_text())
    ep_lo = {int(r.episode_index): int(r.dataset_from_index) for r in eps.itertuples()}

    tasks_seen: dict[str, int] = {}
    episodes_meta = []
    jobs = []
    total = 0
    for new_idx, (ep, a, b, label) in enumerate(clips):
        n = b - a
        lo = ep_lo[ep] + a
        d = frame.iloc[lo : lo + n].copy().reset_index(drop=True)
        ti = tasks_seen.setdefault(label, len(tasks_seen))
        d["frame_index"] = np.arange(n, dtype=np.int64)
        d["timestamp"] = (np.arange(n, dtype=np.float32) / fps).astype(np.float32)
        d["episode_index"] = np.int64(new_idx)
        d["index"] = np.arange(total, total + n, dtype=np.int64)
        d["task_index"] = np.int64(ti)
        d = d[[c for c in d.columns if not c.startswith("observation.images.")]]
        d.to_parquet(out / f"data/chunk-000/episode_{new_idx:06d}.parquet", index=False)

        for cam in CAMERAS:
            s = src / f"videos/{cam}/chunk-000/file-{ep:03d}.mp4"
            t = out / f"videos/chunk-000/{cam}/episode_{new_idx:06d}.mp4"
            jobs.append((s, a, n, t))
        episodes_meta.append({"episode_index": new_idx, "tasks": [label], "length": n})
        total += n

    print(f"cutting {len(jobs)} clips with {workers} workers ...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(cut_clip, s, a, n, t, fps, crf): t for s, a, n, t in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            f.result()
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)

    with open(out / "meta/episodes.jsonl", "w") as fh:
        for m in episodes_meta:
            fh.write(json.dumps(m) + "\n")
    with open(out / "meta/tasks.jsonl", "w") as fh:
        for name, i in sorted(tasks_seen.items(), key=lambda kv: kv[1]):
            fh.write(json.dumps({"task_index": i, "task": name}) + "\n")

    info = {
        "codebase_version": "v2.1",
        "robot_type": src_info.get("robot_type", "Unitree_G1_Dex1_DecoupledWBC"),
        "total_episodes": len(episodes_meta),
        "total_frames": total,
        "total_tasks": len(tasks_seen),
        "total_videos": len(episodes_meta) * len(CAMERAS),
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {},
    }
    for key, feat in src_info["features"].items():
        if feat.get("dtype") == "video":
            feat = dict(feat)
            feat["info"] = dict(feat.get("info", {}))
            feat["info"].update({"video.codec": "h264", "video.pix_fmt": "yuv420p"})
        info["features"][key] = feat
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))

    modality = json.loads((src / "meta/modality.json").read_text())
    (out / "meta/modality.json").write_text(json.dumps(modality, indent=4))

    print(f"wrote {len(episodes_meta)} clips / {total} frames -> {out}")
    if verify:
        verify_cuts(src, out, clips, verify)


def verify_cuts(src: Path, out: Path, clips: list, n_check: int) -> None:
    """Decode the same frames from clip and source; they must be identical."""
    rng = np.random.default_rng(0)
    picks = rng.choice(len(clips), size=min(n_check, len(clips)), replace=False)
    cam = CAMERAS[0]
    worst = 0.0
    for i in picks:
        ep, a, b, _ = clips[int(i)]
        n = b - a
        idx = sorted({0, n // 2, n - 1})
        got = decode_frames(out / f"videos/chunk-000/{cam}/episode_{int(i):06d}.mp4", idx)
        want = decode_frames(
            src / f"videos/{cam}/chunk-000/file-{ep:03d}.mp4", [a + j for j in idx]
        )
        err = float(np.abs(got.astype(np.int16) - want.astype(np.int16)).mean())
        worst = max(worst, err)
    # both sides are lossy h264/AV1 decodes of the same pixels, so a small
    # residual is expected; a whole-frame offset shows up as tens of levels.
    print(f"verify: max mean |diff| over {len(picks)} clips = {worst:.2f} levels")
    if worst > 12.0:
        sys.exit("verify FAILED: clips do not line up with the source")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    ap.add_argument(
        "--still-eps",
        type=float,
        default=0.01,
        help="max |base velocity| (m/s and rad/s) still counted as standing",
    )
    ap.add_argument(
        "--min-frames",
        type=int,
        default=60,
        help="drop runs shorter than this; must exceed the action horizon",
    )
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--verify", type=int, default=12, help="clips to spot-check; 0 disables")
    args = ap.parse_args()

    fps = int(json.loads((args.src / "meta/info.json").read_text())["fps"])
    frame, eps, clips = build_clips(args.src, tuple(args.tasks), args.still_eps, args.min_frames)
    kept = sum(b - a for _, a, b, _ in clips)
    print(
        f"{len(clips)} clips / {kept} frames ({kept / fps / 60:.1f} min) "
        f"from {len(eps)} source episodes"
    )
    write_dataset(args.src, args.out, frame, eps, clips, fps, args.crf, args.workers, args.verify)


if __name__ == "__main__":
    main()
