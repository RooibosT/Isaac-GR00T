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

"""Convert carroll511/IKEA_table_assembly (LeRobot v3.0) to the v2.1 layout GR00T reads.

v3.0 packs every episode into one parquet and a handful of very long mp4s;
`LeRobotEpisodeLoader` wants one parquet and one mp4 per episode per camera plus
`meta/episodes.jsonl`. This is a pure re-layout — no frame is resampled, dropped
or reordered, and the source's per-episode `timestamp`/`frame_index` columns
already follow the v2.1 convention, so they are copied through untouched.

Two things are not straight copies:

* **Video is re-encoded, not stream-copied.** The source is AV1 with GOP 2, so a
  copy would land on the preceding keyframe for any episode starting at an odd
  frame — one frame of silent state/action misalignment on ~half the episodes.
  Cutting with input `-ss` plus `-frames:v` is frame-exact, and because the GOP
  is 2 the seek costs about one frame, so the whole job is roughly one decode
  pass per camera. Output is h264 GOP 2 to match the BCT datasets the training
  throughput numbers were measured on.
* **`modality.json` gains the `video` and `annotation` blocks** GR00T needs; the
  source has only `state`/`action`. The `state`/`action` blocks are copied
  verbatim, including `base_cmd_vel` — it is constant zero in this recording and
  must be left out of the *training config*, but the dataset should still
  describe what was recorded.

`--verify` decodes sampled frames out of the cut clips and compares them against
the same frames decoded from the source, which is the only real proof that the
seek arithmetic is right.
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


CAMERAS = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist")
CHUNK_SIZE = 1000


def video_key(cam: str) -> str:
    return f"observation.images.{cam}"


def cut_clip(src: Path, first_frame: int, n_frames: int, dst: Path, fps: float, crf: int) -> None:
    """Write frames [first_frame, first_frame + n_frames) of `src` to `dst`.

    Seeks a quarter of a frame early so the requested frame is unambiguously the
    first one at or after the seek point regardless of PTS rounding; `-frames:v`
    then bounds the tail.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    seek = max(0.0, (first_frame - 0.25) / fps)
    cmd = [
        "ffmpeg",
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
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)


def build_meta(src: Path, out: Path, eps: pd.DataFrame, episode_ids: list[int]) -> dict:
    """Write meta/ for the subset `episode_ids`, renumbered 0..N-1 in file order."""
    (out / "meta").mkdir(parents=True, exist_ok=True)
    info3 = json.loads((src / "meta/info.json").read_text())
    mod3 = json.loads((src / "meta/modality.json").read_text())
    tasks = pd.read_parquet(src / "meta/tasks.parquet")
    # tasks.parquet carries the task string as the frame index
    task_names = {int(v): str(k) for k, v in tasks["task_index"].items()}

    with (out / "meta/tasks.jsonl").open("w") as fh:
        for idx in sorted(task_names):
            fh.write(json.dumps({"task_index": idx, "task": task_names[idx]}) + "\n")

    used = set()
    with (out / "meta/episodes.jsonl").open("w") as fh:
        for new_i, old_i in enumerate(episode_ids):
            row = eps.iloc[old_i]
            task_list = [str(t) for t in row["tasks"]]
            used.update(task_list)
            fh.write(
                json.dumps(
                    {
                        "episode_index": new_i,
                        "tasks": task_list,
                        "length": int(row["length"]),
                    }
                )
                + "\n"
            )

    total_frames = int(eps.iloc[episode_ids]["length"].sum())
    n_eps = len(episode_ids)
    info = {
        "codebase_version": "v2.1",
        "robot_type": info3.get("robot_type", "Unitree_G1_Dex1_IKEA"),
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": len(task_names),
        "total_videos": n_eps * len(CAMERAS),
        "total_chunks": (n_eps + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": info3["fps"],
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": dict(info3["features"]),
    }
    for cam in CAMERAS:
        feat = info["features"].get(video_key(cam))
        if feat is not None:
            feat = json.loads(json.dumps(feat))
            feat["info"] = {**feat.get("info", {}), "video.codec": "h264"}
            info["features"][video_key(cam)] = feat
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))

    modality = {
        "state": mod3["state"],
        "action": mod3["action"],
        "video": {cam: {"original_key": video_key(cam)} for cam in CAMERAS},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    if "_ikea" in mod3:
        modality["_ikea"] = mod3["_ikea"]
    (out / "meta/modality.json").write_text(json.dumps(modality, indent=4))
    return info


def convert(
    src: Path,
    out: Path,
    episode_ids: list[int],
    *,
    jobs: int,
    crf: int,
    link_from: Path | None = None,
) -> None:
    eps = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")
    info3 = json.loads((src / "meta/info.json").read_text())
    fps = float(info3["fps"])

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    info = build_meta(src, out, eps, episode_ids)
    print(f"  meta written: {info['total_episodes']} eps / {info['total_frames']} frames")

    # ---- data ----
    data = pd.read_parquet(src / "data/chunk-000/file-000.parquet")
    groups = {int(k): v for k, v in data.groupby("episode_index")}
    for new_i, old_i in enumerate(episode_ids):
        sub = groups[old_i]
        dst = out / f"data/chunk-{new_i // CHUNK_SIZE:03d}/episode_{new_i:06d}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        sub.reset_index(drop=True).to_parquet(dst, index=False)
    print(f"  data: {len(episode_ids)} episode parquets written")

    # ---- video ----
    tasks: list[tuple] = []
    for new_i, old_i in enumerate(episode_ids):
        row = eps.iloc[old_i]
        for cam in CAMERAS:
            dst = (
                out
                / f"videos/chunk-{new_i // CHUNK_SIZE:03d}/{video_key(cam)}/episode_{new_i:06d}.mp4"
            )
            if link_from is not None:
                srcclip = (
                    link_from
                    / f"videos/chunk-{old_i // CHUNK_SIZE:03d}/{video_key(cam)}/episode_{old_i:06d}.mp4"
                )
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.hardlink_to(srcclip)
                continue
            fi = int(row[f"videos/{video_key(cam)}/file_index"])
            t0 = float(row[f"videos/{video_key(cam)}/from_timestamp"])
            f0 = int(round(t0 * fps))
            srcvid = src / f"videos/{video_key(cam)}/chunk-000/file-{fi:03d}.mp4"
            tasks.append((srcvid, f0, int(row["length"]), dst, fps, crf))

    if tasks:
        done = 0
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(cut_clip, *t): t for t in tasks}
            for fut in as_completed(futs):
                fut.result()
                done += 1
                if done % 200 == 0 or done == len(tasks):
                    print(f"  video: {done}/{len(tasks)} clips", flush=True)
    else:
        print(f"  video: {len(episode_ids) * len(CAMERAS)} clips hardlinked")


def verify(src: Path, out: Path, episode_ids: list[int], n_samples: int, fps: float) -> None:
    """Decode sampled frames from the cut clips and from the source; compare pixels."""
    from torchcodec.decoders import VideoDecoder

    eps = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")
    rng = np.random.default_rng(0)
    picks = rng.choice(len(episode_ids), size=min(n_samples, len(episode_ids)), replace=False)
    worst = 0.0
    n_checked = 0
    # exact-mode indexing of the multi-thousand-frame sources is expensive; build once
    sources: dict[tuple[str, int], object] = {}
    for new_i in sorted(picks):
        old_i = episode_ids[new_i]
        row = eps.iloc[old_i]
        L = int(row["length"])
        for cam in CAMERAS:
            fi = int(row[f"videos/{video_key(cam)}/file_index"])
            f0 = int(round(float(row[f"videos/{video_key(cam)}/from_timestamp"]) * fps))
            if (cam, fi) not in sources:
                sources[(cam, fi)] = VideoDecoder(
                    str(src / f"videos/{video_key(cam)}/chunk-000/file-{fi:03d}.mp4"),
                    seek_mode="exact",
                )
            sd = sources[(cam, fi)]
            cd = VideoDecoder(
                str(
                    out
                    / f"videos/chunk-{new_i // CHUNK_SIZE:03d}/{video_key(cam)}/episode_{new_i:06d}.mp4"
                ),
                seek_mode="exact",
            )
            if cd.metadata.num_frames != L:
                raise SystemExit(
                    f"FAIL ep{new_i} {cam}: clip has {cd.metadata.num_frames} frames, expected {L}"
                )
            for j in (0, L // 2, L - 1):
                a = sd.get_frame_at(f0 + j).data.float()
                b = cd.get_frame_at(j).data.float()
                err = (a - b).abs().mean().item()
                worst = max(worst, err)
                n_checked += 1
            del cd
    print(f"  verified {n_checked} frames across {len(picks)} episodes x {len(CAMERAS)} cams")
    print(f"  worst mean|Δpixel| between source frame and cut frame: {worst:.3f} / 255")
    if worst > 6.0:
        raise SystemExit("FAIL: frames do not match — the cut is misaligned, not just re-encoded")
    print("  OK: every sampled clip is frame-aligned with the source")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument(
        "--out", required=True, type=Path, help="output prefix; _train/_val are appended"
    )
    ap.add_argument(
        "--val-sessions", default="", help="comma-separated session ids for the val split"
    )
    ap.add_argument("--sessions-file", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--verify-samples", type=int, default=12)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    eps = pd.read_parquet(args.src / "meta/episodes/chunk-000/file-000.parquet")
    fps = float(json.loads((args.src / "meta/info.json").read_text())["fps"])
    n = len(eps)

    val_eps: list[int] = []
    if args.val_sessions:
        sessions = json.loads(args.sessions_file.read_text())
        want = {int(s) for s in args.val_sessions.split(",")}
        for sid in sorted(want):
            a, b = sessions[sid]
            val_eps.extend(range(a, b))
    val_set = set(val_eps)
    train_eps = [i for i in range(n) if i not in val_set]

    full = args.out
    print(f"[full] {full}  ({n} episodes)")
    convert(args.src, full, list(range(n)), jobs=args.jobs, crf=args.crf)
    if not args.no_verify:
        verify(args.src, full, list(range(n)), args.verify_samples, fps)

    if val_eps:
        for name, ids in (("train", train_eps), ("val", val_eps)):
            dst = full.with_name(full.name + f"_{name}")
            print(
                f"[{name}] {dst}  ({len(ids)} episodes, {int(eps.iloc[ids]['length'].sum())} frames)"
            )
            convert(args.src, dst, ids, jobs=args.jobs, crf=args.crf, link_from=full)

    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
