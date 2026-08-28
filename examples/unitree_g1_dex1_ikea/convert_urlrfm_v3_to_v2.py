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

"""Convert the URL-RFM subtask recordings (LeRobot v3.0) to the v2.1 layout.

Sources are `URL-RFM/IKEA_rotatetable{1,2,3}` and `URL-RFM/IKEA_fliptable`: one
aggregated parquet plus four long mp4s each, where `LeRobotEpisodeLoader` wants
one parquet and one mp4 per episode per camera plus `meta/episodes.jsonl`.

The cutting is lifted from `convert_ikea_v3_to_v2.py` and is frame-exact for the
same reason: input `-ss` plus `-frames:v`, never a stream copy, because the
source is GOP-2 and a copy would land on the preceding keyframe for any episode
starting on an odd frame.

What is specific to these sources:

* **Two stages, so the expensive half runs once.** Stage 1 cuts each source into
  its own v2.1 dataset at full width. Stage 2 builds the merged training set by
  rewriting parquets and hardlinking the already-cut clips. Video cutting is
  ~40 min for this much footage; parquet rewriting is seconds. Anything that
  changes only labels or column widths therefore costs seconds, not a re-cut.
* **State 117 -> 86 and action 33 -> 19 on merge.** These recordings carry joint
  torque (state 86:117) and feedforward torque (action 19:33) that the older
  `G1_Dex1_IKEA_table_30hz_v2` does not. A parquet column is fixed width, so a
  single merged dataset has to agree; the older set cannot gain torque except by
  padding zeros, which would be a lie the model would happily learn. Torque is
  dropped from the merged copy and kept in the stage-1 datasets, so a future
  torque ablation re-runs stage 2 only.
* **Task strings are assigned per source, not read from the source.** All three
  rotate sets are labelled `rotate table base` upstream even though set 2 starts
  from the other table orientation, and whether that distinction should reach the
  model is exactly what the label variants exist to test. `unified` and `split`
  differ in one thing only -- what set 2 is called -- so that comparison has a
  single variable; `renamed` differs from `unified` only in the shared name, so
  that one prices the wording by itself. Sets 1 and 3 are never separated: they differ by leg count, which
  measures smaller than the orientation difference and which vision should absorb.

  The wording is not free choice. Measured against this model's frozen text stack
  (cosine distance between the instruction's text tokens, with a trivial reword
  such as adding "it" landing at 0.0095 as the floor):

  - `... to portrait` / `... to landscape`              0.0056  -- below the floor
  - every other direction adverb tried                 0.014-0.026
  - `stand ... upright` / `lay ... sideways`            0.0650  -- wrong motion
  - **`turn the tabletop square` / `spin the crossbars around`  0.1056**

  Direction words barely move this encoder; only a different verb does. The two
  chosen strings sit where the existing task names sit from each other
  (0.1157-0.1442) and at least 0.0914 from all of them. They are deliberately not
  descriptive of direction -- the string only has to be a distinguishable
  condition, and every phrasing that did describe the direction was too close to
  its partner to be one.

  `rotate table base` is avoided as the shared name for the same reason: it sits
  0.056 from the existing `rotate leg to tighten`, closer than the pair that
  already interferes.
* **The sources carry sporadic undecodable frames.** `rotatetable1/cam_left_high`
  frame 8753 and `rotatetable2/cam_right_high` frame 946 both fail to decode from
  the aggregated mp4, in training and unused views alike. Re-encoding conceals
  them: the clip that contains one comes out the right length and decodes end to
  end, because ffmpeg carries on past a bad packet. So verification skips source
  frames it cannot read rather than aborting, and reports how many it skipped --
  the clips are what training consumes and they are checked directly.
  The same flakiness shows up as a lone frame that matches the source one index
  over while the frames either side of it match exactly, so alignment is judged
  per clip rather than per frame: a mis-seeked cut is shifted uniformly and
  cannot be right at both ends and wrong in the middle.
* **Val is held out per source by episode.** Each source is a single recording
  session, so the session-level split used for the older set (see EXPERIMENTS.md
  section 1) is not available here and episode-level is the best on offer.

ffmpeg 7 is needed on PATH, not just its libs on LD_LIBRARY_PATH -- the training
launcher only exports the latter, so a shell set up for training still fails here
with FileNotFoundError: 'ffmpeg'.

    export PATH="$HOME/micromamba/envs/ffmpeg7/bin:$PATH"

Usage:
    # stage 1 -- cut each source once (slow)
    python examples/unitree_g1_dex1_ikea/convert_urlrfm_v3_to_v2.py stage1 \
        --src-root datasets/URL-RFM --out-root datasets/URL-RFM/v2

    # stage 2 -- merge with the existing IKEA set, one directory per label variant (fast)
    python examples/unitree_g1_dex1_ikea/convert_urlrfm_v3_to_v2.py stage2 \
        --stage1-root datasets/URL-RFM/v2 \
        --ikea datasets/carroll511/G1_Dex1_IKEA_table_30hz_v2 \
        --out datasets/carroll511/G1_Dex1_IKEA_all_30hz --variant unified
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

# mean |dpixel| above which a sampled frame counts as not matching the source.
# CRF 18 re-encoding lands around 1.2-2.3 on this footage; a one-frame slip on
# moving arms lands above 40.
PIXEL_TOL = 6.0

# state/action width the merged set is cut back to -- everything the older
# G1_Dex1_IKEA_table_30hz_v2 carries, and nothing past it. `stage2 --state-dim`
# raises the state cut to 117 for the torque re-export, whose extra 31 columns
# sit past 86; the action stays at 19 because the two `arm_tauff` blocks past it
# are a different action space, not extra observation.
MERGED_STATE_DIM = 86
MERGED_ACTION_DIM = 19

# source -> ({variant: task string}, held-out episode count)
#
#   unified   one string for all three rotate sets
#   split     set 2 gets its own string; the only difference from `unified`
#   renamed   as `unified`, under the natural name, to price the rename by itself
#   twohand   rotatetable1 is replaced by its re-shoot and sets 1 and 3 are dropped
#   twohand1  as `twohand`, but the two surviving sets share one string
#
# The last two exist because `IKEA_rotatetable1_v2` is not more of the same motion.
# The older three sets are all one-handed -- the right arm's joints move 0.05-0.23
# rad and its wrist never leaves an 8-17 cm box -- while the re-shoot moves the
# right arm 0.35-1.24 rad per joint and its wrist 9-11 cm on every axis, at a
# higher median torque, without ever closing the right gripper. It braces the
# table with an open hand while the left turns it, and finishes 13% sooner.
#
# So the re-shoot cannot be *added* to `IKEA_rotatetable1` under one string: that
# is two answers to one picture, and the model averages them. It replaces it, and
# `IKEA_rotatetable3` goes too -- same direction, still one-handed, and the pair
# that fails on hardware. `twohand1` is the control that prices the instruction
# split now that the two surviving sets really do differ in technique, which was
# not true when section 18 measured the split and found it unnecessary.
#
# A `None` label drops the source from that variant.
VARIANTS = ("unified", "split", "renamed", "twohand", "twohand1")
SOURCES = {
    "IKEA_rotatetable1": (
        {
            "unified": "turn the tabletop square",
            "split": "turn the tabletop square",
            "renamed": "rotate table base",
            "twohand": None,
            "twohand1": None,
        },
        4,
    ),
    "IKEA_rotatetable2": (
        {
            "unified": "turn the tabletop square",
            "split": "spin the crossbars around",
            "renamed": "rotate table base",
            "twohand": "spin the crossbars around",
            "twohand1": "turn the tabletop square",
        },
        4,
    ),
    "IKEA_rotatetable3": (
        {
            "unified": "turn the tabletop square",
            "split": "turn the tabletop square",
            "renamed": "rotate table base",
            "twohand": None,
            "twohand1": None,
        },
        4,
    ),
    "IKEA_fliptable": (
        {
            "unified": "flip table",
            "split": "flip table",
            "renamed": "flip table",
            "twohand": "flip table",
            "twohand1": "flip table",
        },
        5,
    ),
    # last on purpose: the holdout draws come off one rng in this order, so
    # inserting a source anywhere earlier would silently reshuffle which episodes
    # the already-built variants held out
    "IKEA_rotatetable1_v2": (
        {
            "unified": None,
            "split": None,
            "renamed": None,
            "twohand": "turn the tabletop square",
            "twohand1": "turn the tabletop square",
        },
        4,
    ),
}


def video_key(cam: str) -> str:
    return f"observation.images.{cam}"


def cut_clip(src: Path, first_frame: int, n_frames: int, dst: Path, fps: float, crf: int) -> None:
    """Write frames [first_frame, first_frame + n_frames) of `src` to `dst`.

    Seeks a quarter frame early so the requested frame is unambiguously the first
    at or after the seek point regardless of PTS rounding; `-frames:v` bounds the
    tail.
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


# --------------------------------------------------------------------------- #
# stage 1: one v3.0 source -> one v2.1 dataset, full width
# --------------------------------------------------------------------------- #
def stage1_one(src: Path, out: Path, jobs: int, crf: int) -> None:
    eps = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")
    info3 = json.loads((src / "meta/info.json").read_text())
    mod3 = json.loads((src / "meta/modality.json").read_text())
    fps = float(info3["fps"])
    n = len(eps)

    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    tasks = pd.read_parquet(src / "meta/tasks.parquet")
    task_names = {int(v): str(k) for k, v in tasks["task_index"].items()}
    with (out / "meta/tasks.jsonl").open("w") as fh:
        for idx in sorted(task_names):
            fh.write(json.dumps({"task_index": idx, "task": task_names[idx]}) + "\n")
    with (out / "meta/episodes.jsonl").open("w") as fh:
        for i in range(n):
            fh.write(
                json.dumps(
                    {
                        "episode_index": i,
                        "tasks": [str(t) for t in eps.iloc[i]["tasks"]],
                        "length": int(eps.iloc[i]["length"]),
                    }
                )
                + "\n"
            )

    info = {
        "codebase_version": "v2.1",
        "robot_type": info3.get("robot_type", "Unitree_G1_Dex1_IKEA"),
        "total_episodes": n,
        "total_frames": int(eps["length"].sum()),
        "total_tasks": len(task_names),
        "total_videos": n * len(CAMERAS),
        "total_chunks": (n + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": info3["fps"],
        "splits": {"train": f"0:{n}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": json.loads(json.dumps(info3["features"])),
    }
    for cam in CAMERAS:
        if video_key(cam) in info["features"]:
            info["features"][video_key(cam)].setdefault("info", {})["video.codec"] = "h264"
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    (out / "meta/modality.json").write_text(
        json.dumps(
            {
                "state": mod3["state"],
                "action": mod3["action"],
                "video": {cam: {"original_key": video_key(cam)} for cam in CAMERAS},
                "annotation": {"human.task_description": {"original_key": "task_index"}},
            },
            indent=4,
        )
    )

    data = pd.read_parquet(src / "data/chunk-000/file-000.parquet")
    for i, (_, sub) in enumerate(data.groupby("episode_index")):
        dst = out / f"data/chunk-{i // CHUNK_SIZE:03d}/episode_{i:06d}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        sub.reset_index(drop=True).to_parquet(dst, index=False)
    print(f"  data: {n} episode parquets", flush=True)

    jobs_list = []
    for i in range(n):
        row = eps.iloc[i]
        for cam in CAMERAS:
            t0 = float(row[f"videos/{video_key(cam)}/from_timestamp"])
            jobs_list.append(
                (
                    src / f"videos/{video_key(cam)}/chunk-000/file-000.mp4",
                    int(round(t0 * fps)),
                    int(row["length"]),
                    out
                    / f"videos/chunk-{i // CHUNK_SIZE:03d}/{video_key(cam)}/episode_{i:06d}.mp4",
                    fps,
                    crf,
                )
            )
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = [pool.submit(cut_clip, *t) for t in jobs_list]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 50 == 0 or done == len(jobs_list):
                print(f"  video: {done}/{len(jobs_list)} clips", flush=True)


def check_clip_lengths(out: Path) -> None:
    """Frame-count check on every produced clip. Metadata only, so it is cheap.

    A clip that is short by even one frame silently misaligns state and action for
    the rest of that episode, and the pixel check below only samples a few.
    """
    from torchcodec.decoders import VideoDecoder

    eps = [json.loads(line) for line in open(out / "meta/episodes.jsonl")]
    bad = []
    for e in eps:
        i, L = e["episode_index"], e["length"]
        for cam in CAMERAS:
            clip = out / f"videos/chunk-{i // CHUNK_SIZE:03d}/{video_key(cam)}/episode_{i:06d}.mp4"
            n = VideoDecoder(str(clip), seek_mode="approximate").metadata.num_frames
            if n != L:
                bad.append((i, cam, n, L))
    if bad:
        for b in bad[:10]:
            print(f"    ep{b[0]} {b[1]}: {b[2]} frames, expected {b[3]}")
        raise SystemExit(f"FAIL: {len(bad)} clips have the wrong length")
    print(f"  all {len(eps) * len(CAMERAS)} clips have the expected frame count")


def verify(src: Path, out: Path, n_samples: int) -> None:
    """Decode sampled frames from cut clips and the source and compare pixels."""
    from torchcodec.decoders import VideoDecoder

    eps = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")
    fps = float(json.loads((src / "meta/info.json").read_text())["fps"])
    rng = np.random.default_rng(0)
    picks = rng.choice(len(eps), size=min(n_samples, len(eps)), replace=False)
    checked, skipped, artefacts = 0, 0, 0
    misaligned: list[tuple[int, str, list[float]]] = []
    sources: dict[str, object] = {}
    for i in sorted(picks):
        row = eps.iloc[i]
        L = int(row["length"])
        for cam in CAMERAS:
            f0 = int(round(float(row[f"videos/{video_key(cam)}/from_timestamp"]) * fps))
            if cam not in sources:
                sources[cam] = VideoDecoder(
                    str(src / f"videos/{video_key(cam)}/chunk-000/file-000.mp4"), seek_mode="exact"
                )
            cd = VideoDecoder(
                str(
                    out / f"videos/chunk-{i // CHUNK_SIZE:03d}/{video_key(cam)}/episode_{i:06d}.mp4"
                ),
                seek_mode="exact",
            )
            if cd.metadata.num_frames != L:
                raise SystemExit(
                    f"FAIL ep{i} {cam}: clip has {cd.metadata.num_frames} frames, expected {L}"
                )
            errs = []
            for j in (0, L // 2, L - 1):
                try:
                    a = sources[cam].get_frame_at(f0 + j).data.float()
                except RuntimeError:
                    # bad packet in the source; the clip is re-encoded from it and
                    # is fine, there is just nothing to compare against here
                    skipped += 1
                    continue
                b = cd.get_frame_at(j).data.float()
                errs.append((a - b).abs().mean().item())
                checked += 1
            del cd
            bad = [e for e in errs if e > PIXEL_TOL]
            if errs and len(bad) == len(errs):
                misaligned.append((i, cam, errs))
            elif bad:
                artefacts += 1
    print(f"  verified {checked} frames across {len(picks)} eps x {len(CAMERAS)} cams")
    if skipped:
        print(f"    {skipped} source frames skipped (undecodable in the source)")
    if artefacts:
        print(f"    {artefacts} clips had one frame disagree while its neighbours matched")
        print("      -> source decoder flakiness, not a shifted cut")
    if misaligned:
        for i, cam, errs in misaligned[:10]:
            print(f"    ep{i} {cam}: every sampled frame differs ({[round(e, 1) for e in errs]})")
        raise SystemExit(f"FAIL: {len(misaligned)} clips are uniformly misaligned")
    if checked == 0:
        raise SystemExit("FAIL: no frame could be compared")
    print("  OK: no clip is uniformly shifted against the source")


# --------------------------------------------------------------------------- #
# stage 2: merge stage-1 outputs with the existing IKEA set
# --------------------------------------------------------------------------- #
def read_v21(path: Path) -> tuple[list[dict], dict, dict]:
    eps = [json.loads(line) for line in open(path / "meta/episodes.jsonl")]
    info = json.loads((path / "meta/info.json").read_text())
    mod = json.loads((path / "meta/modality.json").read_text())
    return eps, info, mod


def stage2(
    stage1_root: Path,
    ikea: Path,
    out: Path,
    variant: str,
    seed: int,
    drop_ikea: bool = False,
    state_dim: int = MERGED_STATE_DIM,
    action_dim: int = MERGED_ACTION_DIM,
) -> None:
    rng = np.random.default_rng(seed)

    # (source_path, episode_index_in_source, task_string, is_val)
    plan: list[tuple[Path, int, str, bool]] = []
    # `ikea` is always read, even when its episodes are dropped: its meta is the
    # template that fixes the state/action layout, so every variant -- including
    # the new-tasks-only one -- agrees on what each column means.
    for split in ("train", "val"):
        src = ikea.with_name(ikea.name + f"_{split}")
        eps, _, _ = read_v21(src)
        if drop_ikea:
            continue
        for e in eps:
            plan.append((src, e["episode_index"], e["tasks"][0], split == "val"))

    for name, (labels, n_val) in SOURCES.items():
        src = stage1_root / name
        if not src.exists():
            if labels[variant] is not None:
                raise SystemExit(f"{name} has no stage1 output at {src}")
            continue
        eps, _, _ = read_v21(src)
        # deterministic per-source holdout; each source is one session, so this is
        # episode-level rather than the session-level split used for the older set.
        # The draw happens even for a source this variant drops, so adding a variant
        # cannot shift which episodes the older variants held out.
        val = set(rng.choice(len(eps), size=n_val, replace=False).tolist())
        label = labels[variant]
        if label is None:
            continue
        for e in eps:
            plan.append((src, e["episode_index"], label, e["episode_index"] in val))

    for split in ("train", "val"):
        want = [p for p in plan if p[3] == (split == "val")]
        dst = out.with_name(out.name + f"_{split}")
        write_merged(want, dst, ikea, state_dim, action_dim)
        print(f"[{split}] {dst.name}: {len(want)} episodes")


def write_merged(
    plan: list[tuple[Path, int, str, bool]],
    dst: Path,
    ikea: Path,
    state_dim: int = MERGED_STATE_DIM,
    action_dim: int = MERGED_ACTION_DIM,
) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    labels = sorted({p[2] for p in plan})
    task_index = {t: i for i, t in enumerate(labels)}
    with (dst / "meta/tasks.jsonl").open("w") as fh:
        for t, i in task_index.items():
            fh.write(json.dumps({"task_index": i, "task": t}) + "\n")

    total = 0
    with (dst / "meta/episodes.jsonl").open("w") as feh:
        for new_i, (src, old_i, label, _) in enumerate(plan):
            d = pd.read_parquet(
                src / f"data/chunk-{old_i // CHUNK_SIZE:03d}/episode_{old_i:06d}.parquet"
            )
            S = np.stack(d["observation.state"]).astype(np.float32)[:, :state_dim]
            A = np.stack(d["action"]).astype(np.float32)[:, :action_dim]
            out_df = pd.DataFrame(
                {
                    "observation.state": list(S),
                    "action": list(A),
                    "timestamp": d["timestamp"].to_numpy(),
                    "frame_index": d["frame_index"].to_numpy(),
                    "episode_index": np.full(len(d), new_i, dtype=np.int64),
                    "index": np.arange(total, total + len(d), dtype=np.int64),
                    "task_index": np.full(len(d), task_index[label], dtype=np.int64),
                }
            )
            pq = dst / f"data/chunk-{new_i // CHUNK_SIZE:03d}/episode_{new_i:06d}.parquet"
            pq.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_parquet(pq, index=False)
            for cam in CAMERAS:
                s = (
                    src
                    / f"videos/chunk-{old_i // CHUNK_SIZE:03d}/{video_key(cam)}/episode_{old_i:06d}.mp4"
                )
                t = (
                    dst
                    / f"videos/chunk-{new_i // CHUNK_SIZE:03d}/{video_key(cam)}/episode_{new_i:06d}.mp4"
                )
                t.parent.mkdir(parents=True, exist_ok=True)
                t.hardlink_to(s)
            feh.write(
                json.dumps({"episode_index": new_i, "tasks": [label], "length": int(len(d))}) + "\n"
            )
            total += len(d)

    _, info_i, mod_i = read_v21(ikea.with_name(ikea.name + "_train"))
    n = len(plan)
    info = dict(info_i)
    info.update(
        {
            "total_episodes": n,
            "total_frames": total,
            "total_tasks": len(labels),
            "total_videos": n * len(CAMERAS),
            "total_chunks": (n + CHUNK_SIZE - 1) // CHUNK_SIZE,
            "splits": {"train": f"0:{n}"},
        }
    )
    feats = json.loads(json.dumps(info_i["features"]))
    feats["observation.state"]["shape"] = [state_dim]
    feats["action"]["shape"] = [action_dim]
    info["features"] = feats
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))

    mod = json.loads(json.dumps(mod_i))
    mod["state"] = {k: v for k, v in mod["state"].items() if v["end"] <= state_dim}
    mod["action"] = {k: v for k, v in mod["action"].items() if v["end"] <= action_dim}
    (dst / "meta/modality.json").write_text(json.dumps(mod, indent=4))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("stage1")
    s1.add_argument("--src-root", required=True, type=Path)
    s1.add_argument("--out-root", required=True, type=Path)
    s1.add_argument("--jobs", type=int, default=24)
    s1.add_argument("--crf", type=int, default=18)
    s1.add_argument("--verify-samples", type=int, default=8)
    s1.add_argument("--no-verify", action="store_true")
    s1.add_argument(
        "--skip-existing",
        action="store_true",
        help="leave sources that already have a complete output directory alone",
    )

    s2 = sub.add_parser("stage2")
    s2.add_argument("--stage1-root", required=True, type=Path)
    s2.add_argument("--ikea", required=True, type=Path, help="prefix; _train/_val appended")
    s2.add_argument("--out", required=True, type=Path, help="prefix; _train/_val appended")
    s2.add_argument("--variant", choices=VARIANTS, required=True)
    s2.add_argument(
        "--drop-ikea",
        action="store_true",
        help="keep only the new sources; the older set is still read for its layout",
    )
    s2.add_argument("--seed", type=int, default=20260826)
    s2.add_argument("--state-dim", type=int, default=MERGED_STATE_DIM)
    s2.add_argument("--action-dim", type=int, default=MERGED_ACTION_DIM)
    args = ap.parse_args()

    if args.cmd == "stage1":
        for name in SOURCES:
            src, out = args.src_root / name, args.out_root / name
            if args.skip_existing and (out / "meta/episodes.jsonl").exists():
                print(f"[stage1] {name}: already converted, skipping", flush=True)
                continue
            print(f"[stage1] {name}", flush=True)
            stage1_one(src, out, args.jobs, args.crf)
            if not args.no_verify:
                check_clip_lengths(out)
                verify(src, out, args.verify_samples)
    else:
        stage2(
            args.stage1_root,
            args.ikea,
            args.out,
            args.variant,
            args.seed,
            args.drop_ikea,
            args.state_dim,
            args.action_dim,
        )
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
