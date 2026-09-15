"""Does the policy aim at this episode's hole, or at the average hole?

The table moves between recordings, and nothing in the state says where it is --
only the cameras do. So the distance between where a chunk is heading and where
this episode's hole actually is, compared against the same distance for a policy
that always aims at the mean hole, measures how much the vision path is worth.

The hole is read off the demonstration: at the insertion release the leg is in
it, so the right wrist pose at that frame stands in for the hole, per episode.
Queries sit before contact, where the arm has not yet been guided by touching
anything.
"""

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


EX = Path("/root/01_IKEA/Isaac-GR00T/examples/unitree_g1_dex1_ikea")
sys.path.insert(0, str(EX))
sys.path.insert(0, "/root/01_IKEA/url_lerobot")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--offsets", default="-75,-60,-45,-30")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("pir", EX / "probe_insert_release.py")
    pir = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pir)
    importlib.import_module(Path(a.config).stem)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from url_groot_deploy.common.g1_kinematics import G1WristKinematics

    kin = G1WristKinematics("/root/01_IKEA/url_lerobot/xr_teleoperate", waist_zero=True)

    def xyz(q):
        return np.asarray(kin.both_wrist_poses(np.asarray(q, float), np.zeros(3))[1][:3]) * 1000.0

    tag = EmbodimentTag.resolve("new_embodiment")
    modality = MODALITY_CONFIGS[tag.value]
    loader = LeRobotEpisodeLoader(dataset_path=a.dataset_path, modality_configs=modality)
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=a.checkpoint, device="cuda")
    policy.model.action_head.num_inference_timesteps = 4
    probe = pir.Probe(policy, modality, tag)
    H, keys = probe.horizon, probe.keys
    offs = [int(o) for o in a.offsets.split(",")]

    eps = []
    for ep in range(len(loader)):
        traj = loader[ep]
        gt = np.concatenate(
            [np.vstack([np.asarray(x, np.float32) for x in traj[f"action.{k}"]]) for k in keys], -1
        )
        widths = [np.asarray(traj[f"action.{k}"].iloc[0]).size for k in keys]
        col = int(np.sum(widths[: keys.index("right_gripper")]))
        ev, _ = pir.gripper_events(gt[:, col])
        ops = [i for i, s in ev if s == "O" and i > 0]
        if len(ops) != 6:
            del traj
            continue
        rel = ops[1]
        eps.append({"ep": ep, "rel": rel, "hole": xyz(gt[rel, :14]), "traj": None})
        eps[-1]["queries"] = []
        for o in offs:
            t = rel + o
            if 0 <= t < len(gt) - H:
                pred = pir.stack(probe.sample(traj, a.seeds, t, seed=31337 + t), keys)
                # where the chunk is heading: the wrist at its last row
                aim = np.array([xyz(p[-1, :14]) for p in pred])
                q = {"off": o, "aim": aim.tolist(), "gt_end": xyz(gt[t + H - 1, :14]).tolist()}
                # a run with the goal-pose auxiliary states its estimate outright:
                # the block's first three columns are the goal position, in metres
                if "right_goal_9d" in keys:
                    col = int(
                        sum(
                            np.asarray(traj[f"action.{k}"].iloc[0]).size
                            for k in keys[: keys.index("right_goal_9d")]
                        )
                    )
                    q["goal_head"] = (pred[:, 0, col : col + 3] * 1000.0).tolist()
                eps[-1]["queries"].append(q)
        del traj
        print(f"episode {ep}: release {rel}, hole {np.round(eps[-1]['hole'], 1)}", flush=True)

    holes = np.array([e["hole"] for e in eps])
    mean_hole = holes.mean(0)
    print(
        f"\n{len(eps)} episodes; hole spread across them (mm): "
        f"x {holes[:, 0].std():.1f}  y {holes[:, 1].std():.1f}  z {holes[:, 2].std():.1f}, "
        f"mean pairwise distance {np.mean([np.linalg.norm(h1 - h2) for h1 in holes for h2 in holes]):.1f}"
    )
    print(
        f"\n{'offset':>8} {'aim vs this hole':>18} {'aim vs mean hole':>18} {'mean-hole baseline':>20}"
    )
    out = {"holes": holes.tolist(), "rows": []}
    for o in offs:
        d_self, d_mean, d_base, used = [], [], [], []
        for e in eps:
            q = next((q for q in e["queries"] if q["off"] == o), None)
            if q is None:
                continue
            aim = np.array(q["aim"])
            d_self.append(np.median(np.linalg.norm(aim - e["hole"], axis=-1)))
            d_mean.append(np.median(np.linalg.norm(aim - mean_hole, axis=-1)))
            d_base.append(np.linalg.norm(mean_hole - e["hole"]))
            used.append(int(e["ep"]))
        head = [
            np.median(np.linalg.norm(np.array(q["goal_head"]) - e["hole"], axis=-1))
            for e in eps
            for q in e["queries"]
            if q["off"] == o and "goal_head" in q
        ]
        extra = f"   goal head: {np.median(head):6.1f} mm" if head else ""
        print(
            f"{o:>8} {np.median(d_self):18.1f} {np.median(d_mean):18.1f} {np.median(d_base):20.1f}{extra}"
        )
        out["rows"].append(
            {
                "off": o,
                "aim_vs_hole": float(np.median(d_self)),
                "aim_vs_mean": float(np.median(d_mean)),
                "mean_baseline": float(np.median(d_base)),
                "per_ep": [float(x) for x in d_self],
                "per_ep_head": [float(x) for x in head],
                "eps": used,
            }
        )
    Path(a.output).write_text(json.dumps(out, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
