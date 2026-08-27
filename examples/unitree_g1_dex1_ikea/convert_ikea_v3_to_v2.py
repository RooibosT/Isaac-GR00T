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

`--merge-tasks` fuses runs of consecutive episodes back into one. The source is
two continuous recordings -- file-000 holds 42,943 frames and its episodes sum to
exactly 42,943 over `[0, 42943)`, file-001 likewise -- so an episode boundary is a
label cut, not a splice, and re-joining `pick` to `insert` recovers a stretch of
motion that is otherwise unreachable: with `allow_padding=False` the sampler only
starts a window where the whole horizon fits, so the last `horizon - 1` frames of
every episode are never a window start and the grasp-to-lift transition has no
supervision at all. Merging is still gated per boundary on state continuity
(`--merge-jump-tol`), because a cut made to mark a failed attempt can carry a
sensor discontinuity the frame count cannot see.
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


def resolve_file_index(eps: pd.DataFrame, cam: str, n_files: int) -> np.ndarray:
    """Which physical mp4 each episode is in, derived rather than believed.

    `videos/<key>/file_index` is not trustworthy. The IKEA_pickuptheleg export
    labels episodes 0-177 of both high cameras with file_index 1, when the first
    91 of them are physically in file-000 -- each pair of files was collapsed onto
    the pair's last index. The wrist cameras, which have two files instead of
    four, came out right, so the damage is silent and partial.

    Timestamps cannot lie the same way: they are positions inside one file, so a
    reset to 0 is a new file. Numbering the resets recovers the mapping, and the
    count of resets has to match the count of mp4s on disk or this raises --
    `verify` then decodes frames through the mapping as the real check.
    """
    vk = video_key(cam)
    ft = eps[f"videos/{vk}/from_timestamp"].to_numpy(dtype=float)
    starts = np.flatnonzero(ft == 0.0)
    if len(starts) != n_files:
        raise SystemExit(
            f"{cam}: {len(starts)} timestamp resets but {n_files} mp4 files -- "
            "cannot tell which episode is in which file"
        )
    return np.searchsorted(starts, np.arange(len(ft)), side="right") - 1


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


def build_groups(
    src: Path,
    eps: pd.DataFrame,
    episode_ids: list[int],
    merge_tasks: set[str],
    tol: float,
) -> list[list[int]]:
    """Group `episode_ids` into output episodes, fusing contiguous merge-task runs.

    A run is fused only where the source frames really do abut -- same video file
    and `to_timestamp == from_timestamp` on every camera -- and where the state
    does not step further across the cut than it ever steps inside an episode.
    That second test is the one that matters here: nothing in the frame counts
    can distinguish a cut made mid-take from a cut made because the attempt
    failed and had to be restarted, but a restart shows up as a joint or gripper
    reading that moves further in one frame than the robot ever moves.
    """
    if not merge_tasks:
        return [[i] for i in episode_ids]

    fps = float(json.loads((src / "meta/info.json").read_text())["fps"])
    data = pd.read_parquet(src / "data/chunk-000/file-000.parquet")
    state = np.stack([np.asarray(x, dtype=np.float32) for x in data["observation.state"]])
    lo = eps["dataset_from_index"].to_numpy()
    hi = eps["dataset_to_index"].to_numpy()

    # the largest one-frame move of each dim seen strictly inside an episode
    within = np.zeros(state.shape[1], dtype=np.float32)
    for a, b in zip(lo, hi):
        if b - a > 1:
            within = np.maximum(within, np.abs(np.diff(state[a:b], axis=0)).max(0))

    file_of = {
        cam: resolve_file_index(
            eps, cam, len(list((src / f"videos/{video_key(cam)}/chunk-000").glob("file-*.mp4")))
        )
        for cam in CAMERAS
    }

    def abuts(k: int, nxt: int) -> bool:
        if hi[k] != lo[nxt]:
            return False
        for cam in CAMERAS:
            vk = video_key(cam)
            if file_of[cam][k] != file_of[cam][nxt]:
                return False
            gap = abs(
                float(eps[f"videos/{vk}/to_timestamp"].iloc[k])
                - float(eps[f"videos/{vk}/from_timestamp"].iloc[nxt])
            )
            if gap > 0.5 / fps:
                return False
        return True

    def continuous(k: int, nxt: int) -> tuple[bool, float, int]:
        jump = np.abs(state[hi[k] - 1] - state[lo[nxt]])
        ratio = jump / np.maximum(within, 1e-6)
        d = int(np.argmax(ratio))
        return bool(ratio[d] <= tol), float(ratio[d]), d

    task_of = {i: str(eps["tasks"].iloc[i][0]) for i in episode_ids}
    order = list(episode_ids)
    groups: list[list[int]] = []
    refused: list[tuple[int, int, float, int]] = []
    i = 0
    while i < len(order):
        k = order[i]
        if task_of[k] not in merge_tasks:
            groups.append([k])
            i += 1
            continue
        grp = [k]
        while i + 1 < len(order):
            nxt = order[i + 1]
            if task_of[nxt] not in merge_tasks or nxt != order[i] + 1 or not abuts(order[i], nxt):
                break
            ok, ratio, d = continuous(order[i], nxt)
            if not ok:
                refused.append((order[i], nxt, ratio, d))
                break
            grp.append(nxt)
            i += 1
        groups.append(grp)
        i += 1

    sizes = {}
    for g in groups:
        if len(g) > 1:
            sizes[len(g)] = sizes.get(len(g), 0) + 1
    merged = sum(sizes.values())
    print(f"  merge: {len(order)} source episodes -> {len(groups)} output episodes")
    print(f"         {merged} fused groups, sizes {dict(sorted(sizes.items()))}")
    if tol < 0:
        print(f"         label only: {len(refused)} joins available, none taken")
    else:
        for a, b, ratio, d in refused:
            print(
                f"         refused ep{a}->ep{b}: state dim {d} steps "
                f"{ratio:.1f}x its within-episode max"
            )
        if not refused:
            print("         no boundary refused (every cut is continuous in state)")
    return groups


def build_meta(
    src: Path,
    out: Path,
    eps: pd.DataFrame,
    groups: list[list[int]],
    merge_tasks: set[str],
    merge_label: str,
) -> tuple[dict, dict[int, int]]:
    """Write meta/ for `groups`, one output episode per group, renumbered 0..N-1.

    Returns the info dict and the source-to-output `task_index` remap that the
    parquet writer has to apply: fusing two tasks into one retires their indices.
    """
    (out / "meta").mkdir(parents=True, exist_ok=True)
    info3 = json.loads((src / "meta/info.json").read_text())
    mod3 = json.loads((src / "meta/modality.json").read_text())
    tasks = pd.read_parquet(src / "meta/tasks.parquet")
    # tasks.parquet carries the task string as the frame index
    task_names = {int(v): str(k) for k, v in tasks["task_index"].items()}

    def label_of(old_i: int) -> str:
        t = str(eps["tasks"].iloc[old_i][0])
        return merge_label if t in merge_tasks else t

    if merge_tasks:
        kept = sorted({label_of(g[0]) for g in groups})
        new_index = {name: i for i, name in enumerate(kept)}
        remap = {
            old: new_index[merge_label if name in merge_tasks else name]
            for old, name in task_names.items()
            if (merge_label if name in merge_tasks else name) in new_index
        }
        task_names = {i: name for name, i in new_index.items()}
    else:
        remap = {i: i for i in task_names}

    with (out / "meta/tasks.jsonl").open("w") as fh:
        for idx in sorted(task_names):
            fh.write(json.dumps({"task_index": idx, "task": task_names[idx]}) + "\n")

    with (out / "meta/episodes.jsonl").open("w") as fh:
        for new_i, grp in enumerate(groups):
            fh.write(
                json.dumps(
                    {
                        "episode_index": new_i,
                        "tasks": [label_of(grp[0])],
                        "length": int(sum(int(eps["length"].iloc[g]) for g in grp)),
                    }
                )
                + "\n"
            )

    flat = [g for grp in groups for g in grp]
    total_frames = int(eps.iloc[flat]["length"].sum())
    n_eps = len(groups)
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
    return info, remap


def convert(
    src: Path,
    out: Path,
    groups: list[list[int]],
    *,
    jobs: int,
    crf: int,
    merge_tasks: set[str] = frozenset(),
    merge_label: str = "",
    link_index: dict[tuple[int, ...], int] | None = None,
    link_from: Path | None = None,
) -> None:
    eps = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")
    info3 = json.loads((src / "meta/info.json").read_text())
    fps = float(info3["fps"])

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    info, remap = build_meta(src, out, eps, groups, merge_tasks, merge_label)
    print(f"  meta written: {info['total_episodes']} eps / {info['total_frames']} frames")

    # ---- data ----
    data = pd.read_parquet(src / "data/chunk-000/file-000.parquet")
    by_ep = {int(k): v for k, v in data.groupby("episode_index")}
    for new_i, grp in enumerate(groups):
        sub = pd.concat([by_ep[g] for g in grp], ignore_index=True)
        if len(grp) > 1:
            # frame_index and timestamp restart at every source episode, so a fused
            # episode has to be renumbered or the loader sees the clock jump back
            n = len(sub)
            sub["frame_index"] = np.arange(n, dtype=sub["frame_index"].dtype)
            sub["timestamp"] = (np.arange(n) / fps).astype(sub["timestamp"].dtype)
        sub["episode_index"] = np.full(len(sub), new_i, dtype=sub["episode_index"].dtype)
        sub["task_index"] = sub["task_index"].map(remap).astype(sub["task_index"].dtype)
        dst = out / f"data/chunk-{new_i // CHUNK_SIZE:03d}/episode_{new_i:06d}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(dst, index=False)
    print(f"  data: {len(groups)} episode parquets written")

    # ---- video ----
    file_of = {
        cam: resolve_file_index(
            eps, cam, len(list((src / f"videos/{video_key(cam)}/chunk-000").glob("file-*.mp4")))
        )
        for cam in CAMERAS
    }
    tasks: list[tuple] = []
    for new_i, grp in enumerate(groups):
        head = eps.iloc[grp[0]]
        length = int(sum(int(eps["length"].iloc[g]) for g in grp))
        for cam in CAMERAS:
            dst = (
                out
                / f"videos/chunk-{new_i // CHUNK_SIZE:03d}/{video_key(cam)}/episode_{new_i:06d}.mp4"
            )
            if link_from is not None:
                src_i = link_index[tuple(grp)]
                srcclip = (
                    link_from
                    / f"videos/chunk-{src_i // CHUNK_SIZE:03d}/{video_key(cam)}/episode_{src_i:06d}.mp4"
                )
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.hardlink_to(srcclip)
                continue
            fi = int(file_of[cam][grp[0]])
            t0 = float(head[f"videos/{video_key(cam)}/from_timestamp"])
            f0 = int(round(t0 * fps))
            srcvid = src / f"videos/{video_key(cam)}/chunk-000/file-{fi:03d}.mp4"
            tasks.append((srcvid, f0, length, dst, fps, crf))

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
        print(f"  video: {len(groups) * len(CAMERAS)} clips hardlinked")


def verify(src: Path, out: Path, groups: list[list[int]], n_samples: int, fps: float) -> None:
    """Decode sampled frames from the cut clips and from the source; compare pixels.

    For a fused group this is also the check that the join is real: the sampled
    offsets are taken across the whole merged length, so a clip that silently
    stopped at the first source episode fails on the frame count alone.
    """
    from torchcodec.decoders import VideoDecoder

    eps = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")
    file_of = {
        cam: resolve_file_index(
            eps, cam, len(list((src / f"videos/{video_key(cam)}/chunk-000").glob("file-*.mp4")))
        )
        for cam in CAMERAS
    }
    rng = np.random.default_rng(0)
    picks = rng.choice(len(groups), size=min(n_samples, len(groups)), replace=False)
    worst = 0.0
    n_checked = 0
    # exact-mode indexing of the multi-thousand-frame sources is expensive; build once
    sources: dict[tuple[str, int], object] = {}
    for new_i in sorted(picks):
        grp = groups[new_i]
        row = eps.iloc[grp[0]]
        L = int(sum(int(eps["length"].iloc[g]) for g in grp))
        for cam in CAMERAS:
            fi = int(file_of[cam][grp[0]])
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
            seams = []
            acc = 0
            for g in grp[:-1]:
                acc += int(eps["length"].iloc[g])
                seams += [acc - 1, acc]
            for j in sorted({0, L // 2, L - 1, *seams}):
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
    ap.add_argument(
        "--merge-tasks",
        default="",
        help="comma-separated task strings to fuse across contiguous episodes",
    )
    ap.add_argument("--merge-label", default="", help="instruction the fused episodes are given")
    ap.add_argument(
        "--link-clips-from",
        type=Path,
        default=None,
        help="reuse the mp4s of an existing conversion instead of re-encoding. Only "
        "valid when that conversion has the same episode grouping -- which is the "
        "case for a re-export that adds state columns and leaves the video alone.",
    )
    ap.add_argument(
        "--merge-label-only",
        action="store_true",
        help="apply --merge-label without fusing any episode: isolates what the "
        "instruction alone buys from what the recovered boundary windows buy",
    )
    ap.add_argument(
        "--merge-jump-tol",
        type=float,
        default=1.0,
        help="refuse a join whose state step exceeds this many times the largest "
        "one-frame step that dim ever makes inside an episode (1.0 = physically possible)",
    )
    args = ap.parse_args()

    merge_tasks = {t.strip() for t in args.merge_tasks.split(",") if t.strip()}
    if merge_tasks and not args.merge_label:
        ap.error("--merge-tasks needs --merge-label")

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

    mkw = dict(merge_tasks=merge_tasks, merge_label=args.merge_label)

    full = args.out
    print(f"[full] {full}  ({n} episodes)")
    full_groups = build_groups(
        args.src,
        eps,
        list(range(n)),
        merge_tasks,
        -1.0 if args.merge_label_only else args.merge_jump_tol,
    )
    reuse = {}
    if args.link_clips_from is not None:
        have = sum(1 for _ in open(args.link_clips_from / "meta/episodes.jsonl"))
        if have != len(full_groups):
            raise SystemExit(
                f"--link-clips-from has {have} episodes, this conversion makes "
                f"{len(full_groups)}: the groupings differ, so the clips do not correspond"
            )
        reuse = dict(
            link_from=args.link_clips_from,
            link_index={tuple(g): i for i, g in enumerate(full_groups)},
        )
    convert(args.src, full, full_groups, jobs=args.jobs, crf=args.crf, **mkw, **reuse)
    if not args.no_verify and not reuse:
        verify(args.src, full, full_groups, args.verify_samples, fps)

    if val_eps:
        # the split must not cut a group in half, so it is taken on whole groups
        link_index = {tuple(g): i for i, g in enumerate(full_groups)}
        for name, keep in (
            ("train", lambda g: g[0] not in val_set),
            ("val", lambda g: g[0] in val_set),
        ):
            sel = [g for g in full_groups if keep(g)]
            mixed = [g for g in full_groups if any((e in val_set) != (g[0] in val_set) for e in g)]
            if mixed:
                raise SystemExit(f"group straddles the train/val split: {mixed}")
            dst = full.with_name(full.name + f"_{name}")
            frames = int(sum(int(eps["length"].iloc[e]) for g in sel for e in g))
            print(f"[{name}] {dst}  ({len(sel)} episodes, {frames} frames)")
            convert(
                args.src,
                dst,
                sel,
                jobs=args.jobs,
                crf=args.crf,
                link_index=link_index,
                link_from=full,
                **mkw,
            )

    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
