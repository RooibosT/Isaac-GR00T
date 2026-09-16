"""Does zooming into the insertion area sharpen the frozen features?

The hole read-out saturates near 27 mm at every depth of the frozen backbone, and the
grid the action head sees is coarse: one token covers about 55 x 57 px of the 640 x 480
frame, because the pipeline resizes the short edge to 256 first. Cropping the region the
insertion happens in and letting the same pipeline resize THAT to 256 spends the token
budget on fewer millimetres per token -- 640 -> 256 is 0.53x, a 60% crop -> 256 is 0.67x.

It cannot add optical detail; it only gives back some of what the downscale throws away.

Settings share one pass over the episodes, because decoding the video dominates. Each
saves the same feature the 27 mm number came from: the backbone's last hidden state over
all tokens. Targets are also unchanged -- hole = FK(right wrist) of the commanded arm at
the insertion release, wrist = FK at the queried frame, the positive control.
"""

import argparse
import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


EX = Path("/root/01_IKEA/Isaac-GR00T/examples/unitree_g1_dex1_ikea")
sys.path.insert(0, str(EX))
sys.path.insert(0, "/root/01_IKEA/url_lerobot")

V3 = ["cam_left_high", "cam_left_wrist", "cam_right_wrist"]
FOURTH = "cam_right_high"  # dataset view used as the slot for an extra, derived view

# name -> (views, {view: (x0, x1, y0, y1) crop in fractions}, view whose pixels fill FOURTH)
# every crop keeps 4:3, so a replaced view still costs 88 tokens.
SETTINGS = {
    "base": (V3, {}, None),
    "rw_bl2": (V3, {"cam_right_wrist": (0.00, 0.70, 0.30, 1.00)}, None),
    "rw_b60": (V3, {"cam_right_wrist": (0.20, 0.80, 0.40, 1.00)}, None),
    "rw_bl": (V3, {"cam_right_wrist": (0.00, 0.60, 0.40, 1.00)}, None),
    "rw_bl2_extra": (V3 + [FOURTH], {}, ("cam_right_wrist", (0.00, 0.70, 0.30, 1.00))),
}


def crop(img, box):
    h, w = img.shape[:2]
    x0, x1, y0, y1 = box
    return img[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--offsets", default="-90,-75,-60,-45,-30")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    names = a.settings.split(",")

    spec = importlib.util.spec_from_file_location("pir", EX / "probe_insert_release.py")
    pir = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pir)
    importlib.import_module(Path(a.config).stem)
    from copy import deepcopy

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from url_groot_deploy.common.g1_kinematics import G1WristKinematics

    kin = G1WristKinematics("/root/01_IKEA/url_lerobot/xr_teleoperate", waist_zero=True)

    def xyz(q):
        return np.asarray(kin.both_wrist_poses(np.asarray(q, float), np.zeros(3))[1][:3]) * 1000.0

    tag = EmbodimentTag.resolve("new_embodiment")
    modality = deepcopy(MODALITY_CONFIGS[tag.value])
    modality["video"].modality_keys = list(V3 + [FOURTH])  # load all four, subset per setting
    loader = LeRobotEpisodeLoader(dataset_path=a.dataset_path, modality_configs=modality)
    pol = Gr00tPolicy(embodiment_tag=tag, model_path=a.checkpoint, device="cuda")
    probe = pir.Probe(pol, modality, tag)
    proc = pol.processor

    offs = [int(o) for o in a.offsets.split(",")]
    feats = {n: [] for n in names}
    Y, CUR, G, OFF = [], [], [], []
    n_eps = len(loader) if a.episodes == 0 else min(a.episodes, len(loader))
    for ep in range(n_eps):
        if ep % sn != si:
            continue
        traj = loader[ep]
        keys = probe.keys
        gt = np.concatenate(
            [np.vstack([np.asarray(x, np.float32) for x in traj[f"action.{k}"]]) for k in keys], -1
        )
        widths = [np.asarray(traj[f"action.{k}"].iloc[0]).size for k in keys]
        gcol = int(np.sum(widths[: keys.index("right_gripper")]))
        ev, _ = pir.gripper_events(gt[:, gcol])
        ops = [i for i, s in ev if s == "O" and i > 0]
        if len(ops) < 2:
            del traj
            continue
        rel = ops[1]
        for o in offs:
            t = rel + o
            if t < 0:
                continue
            parsed = probe.observation(traj, t)
            raw = {v: np.asarray(parsed["video"][v])[0, -1] for v in V3 + [FOURTH]}
            for name in names:
                views, crops, extra = SETTINGS[name]
                imgs = {v: (crop(raw[v], crops[v]) if v in crops else raw[v]) for v in views}
                if extra is not None:
                    src, box = extra
                    imgs[FOURTH] = crop(raw[src], box)
                proc.modality_configs[tag.value]["video"].modality_keys = list(views)
                sub = {
                    "video": {v: imgs[v][None, None] for v in views},
                    "state": parsed["state"],
                    "language": parsed["language"],
                }
                step = pol._to_vla_step_data(next(iter(pol._unbatch_observation(sub))))
                p = proc([{"type": pir.MessageType.EPISODE_STEP.value, "content": step}])
                col = pir._rec_to_dtype(pol.collate_fn([p]), dtype=torch.bfloat16)
                with torch.no_grad():
                    bi, _ = pol.model.prepare_input(col["inputs"])
                    out = pol.model.backbone(bi)
                feats[name].append(out["backbone_features"][0].to(torch.float16).cpu().numpy())
            Y.append(xyz(gt[rel, :14]))
            CUR.append(xyz(gt[t, :14]))
            G.append(ep)
            OFF.append(o)
        del traj
        if ep % 10 == 0:
            print(
                f"shard {si}: episode {ep}/{n_eps} rel {rel} samples {len(Y)} tokens "
                + " ".join(f"{n}:{feats[n][-1].shape[0]}" for n in names),
                flush=True,
            )

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"meta_s{si}.npz",
        Y=np.stack(Y),
        CUR=np.stack(CUR),
        G=np.array(G),
        O=np.array(OFF),
    )
    for n in names:
        k = min(x.shape[0] for x in feats[n])
        np.save(out_dir / f"{n}_s{si}.npy", np.stack([x[:k] for x in feats[n]]))
        print(f"{n}: {len(feats[n])} samples, {k} tokens", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
