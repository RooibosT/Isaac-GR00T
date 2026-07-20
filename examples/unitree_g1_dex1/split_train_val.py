#!/usr/bin/env python
"""Split a converted LeRobot v2.1 dataset into train/val subsets.

Holds out every Nth episode (default 10 -> 107demo becomes train 97 / val 10),
renumbers episode/global indices contiguously, hardlinks videos (falls back to
copy across filesystems), and rewrites meta. Statistics are NOT computed here —
run gr00t/data/stats.py on the train split afterwards (the finetune runner
does this automatically).

Usage:
    python examples/unitree_g1_dex1/split_train_val.py \
        --src demo_data/RooibosT/g1_pick_redblock_dex1_sim_merged_107demo \
        [--val-every 10]

Outputs: <src>_train and <src>_val
"""

import argparse
import json
import os
import shutil

import pandas as pd


def build_split(
    src: str, split: str, old_ids: list[int], episodes: dict, cams: list[str], chunk_size: int
) -> None:
    dst = f"{src}_{split}"
    os.makedirs(f"{dst}/meta", exist_ok=True)

    records = []
    global_idx = 0
    for new_idx, old_idx in enumerate(old_ids):
        old_chunk = f"chunk-{old_idx // chunk_size:03d}"
        new_chunk = f"chunk-{new_idx // chunk_size:03d}"
        os.makedirs(f"{dst}/data/{new_chunk}", exist_ok=True)
        df = pd.read_parquet(f"{src}/data/{old_chunk}/episode_{old_idx:06d}.parquet")
        n = len(df)
        df["episode_index"] = new_idx
        df["index"] = range(global_idx, global_idx + n)
        global_idx += n
        df.to_parquet(f"{dst}/data/{new_chunk}/episode_{new_idx:06d}.parquet")

        for cam in cams:
            vdir = f"{dst}/videos/{new_chunk}/observation.images.{cam}"
            os.makedirs(vdir, exist_ok=True)
            vsrc = f"{src}/videos/{old_chunk}/observation.images.{cam}/episode_{old_idx:06d}.mp4"
            vdst = f"{vdir}/episode_{new_idx:06d}.mp4"
            if os.path.exists(vdst):
                os.remove(vdst)
            try:
                os.link(vsrc, vdst)
            except OSError:
                shutil.copy(vsrc, vdst)

        records.append({"episode_index": new_idx, "tasks": episodes[old_idx]["tasks"], "length": n})

    with open(f"{dst}/meta/episodes.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    shutil.copy(f"{src}/meta/tasks.jsonl", f"{dst}/meta/tasks.jsonl")
    shutil.copy(f"{src}/meta/modality.json", f"{dst}/meta/modality.json")

    info = json.load(open(f"{src}/meta/info.json"))
    info["total_episodes"] = len(records)
    info["total_frames"] = global_idx
    info["total_videos"] = len(records) * len(cams)
    info["splits"] = {"train": f"0:{len(records)}"}
    json.dump(info, open(f"{dst}/meta/info.json", "w"), indent=2)
    print(f"  {split}: {len(records)} eps, {global_idx} frames -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Path to the v2.1 dataset root")
    parser.add_argument("--val-every", type=int, default=10, help="Hold out every Nth episode")
    args = parser.parse_args()
    src = args.src.rstrip("/")

    info = json.load(open(f"{src}/meta/info.json"))
    cams = [
        key.removeprefix("observation.images.")
        for key in info["features"]
        if key.startswith("observation.images.")
    ]
    print("cameras:", cams)

    episodes = {
        json.loads(line)["episode_index"]: json.loads(line)
        for line in open(f"{src}/meta/episodes.jsonl")
    }
    ids = sorted(episodes.keys())
    val_ids = set(ids[:: args.val_every])
    train_ids = [i for i in ids if i not in val_ids]
    print(f"train={len(train_ids)} val={len(val_ids)}")

    chunk_size = info.get("chunks_size", 1000)
    build_split(src, "train", train_ids, episodes, cams, chunk_size)
    build_split(src, "val", sorted(val_ids), episodes, cams, chunk_size)
    print("SPLIT DONE")


if __name__ == "__main__":
    main()
