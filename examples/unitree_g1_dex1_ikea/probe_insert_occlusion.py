"""Which pixels does the insertion aim actually depend on?

Attention says where the action head looks; it does not say what moves the action.
This greys out one token cell at a time -- one cell of the 8 x 11 grid each view
becomes after the Qwen processor, about 31 x 32 px of the 340 x 256 eval crop -- and
measures how far the predicted right wrist moves (FK, mm) against the same frame
unoccluded. The diffusion noise is held identical across the batch, so the occlusion
is the only difference. Whole-view and all-view occlusions give the scale, and the
spread between two noise seeds of the unoccluded frame gives the floor.

Rows read: 12, the last of the rows [5, 13) the deployment executes, and 39, where
the chunk is heading.
"""

import argparse
import contextlib
import copy
import importlib
import importlib.util
from pathlib import Path
import sys

import albumentations as A
import numpy as np
from PIL import Image
import torch


EX = Path("/root/01_IKEA/Isaac-GR00T/examples/unitree_g1_dex1_ikea")
sys.path.insert(0, str(EX))
sys.path.insert(0, "/root/01_IKEA/url_lerobot")

VIEWS = ["cam_left_high", "cam_left_wrist", "cam_right_wrist"]
ROWS = (12, 39)
GH, GW = 8, 11
GREY = 127


@contextlib.contextmanager
def shared_noise(seed):
    """Every row of a batch starts from the same diffusion noise."""
    orig = torch.randn

    def randn(*args, **kw):
        size = kw.pop("size", None)
        if size is None:
            size = (
                args[0]
                if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size))
                else args
            )
        size = tuple(size)
        g = torch.Generator(device=kw.get("device", "cpu")).manual_seed(seed)
        return orig((1, *size[1:]), generator=g, **kw).expand(*size).clone()

    torch.randn = randn
    try:
        yield
    finally:
        torch.randn = orig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--offsets", default="-120,-60,-30,-15,-5,0")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))

    spec = importlib.util.spec_from_file_location("pir", EX / "probe_insert_release.py")
    pir = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pir)
    importlib.import_module(Path(a.config).stem)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.image_augmentations import apply_with_replay
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from url_groot_deploy.common.g1_kinematics import G1WristKinematics

    kin = G1WristKinematics("/root/01_IKEA/url_lerobot/xr_teleoperate", waist_zero=True)

    def xyz(q):
        return np.asarray(kin.both_wrist_poses(np.asarray(q, float), np.zeros(3))[1][:3]) * 1000.0

    tag = EmbodimentTag.resolve("new_embodiment")
    modality = MODALITY_CONFIGS[tag.value]
    assert list(modality["video"].modality_keys) == VIEWS
    loader = LeRobotEpisodeLoader(dataset_path=a.dataset_path, modality_configs=modality)
    pol = Gr00tPolicy(embodiment_tag=tag, model_path=a.checkpoint, device="cuda")
    probe = pir.Probe(pol, modality, tag)
    keys, H = probe.keys, probe.horizon
    # The eval crop is applied here, once, so occlusions can be drawn on exactly the
    # image the processor would otherwise have produced; the processor then passes it through.
    eval_tf = pol.processor.eval_image_transform
    pol.processor.eval_image_transform = A.Compose([])

    variants = [("base", -1, -1, -1)]
    variants += [("cell", v, i, j) for v in range(len(VIEWS)) for i in range(GH) for j in range(GW)]
    variants += [("view", v, -1, -1) for v in range(len(VIEWS))]
    variants += [("all", -1, -1, -1)]
    n_cell = len(VIEWS) * GH * GW

    out = {
        k: []
        for k in (
            "ep",
            "off",
            "rel",
            "cells",
            "views",
            "allv",
            "noise",
            "images",
            "aim_base",
            "hole",
        )
    }
    offs = [int(o) for o in a.offsets.split(",")]
    n_eps = len(loader) if a.episodes == 0 else min(a.episodes, len(loader))
    for ep in range(n_eps):
        if ep % sn != si:
            continue
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
        for o in offs:
            t = rel + o
            if not 0 <= t < len(gt):
                continue
            parsed = probe.observation(traj, t)
            obs = next(
                iter(
                    pol._unbatch_observation(
                        {m: dict(parsed[m]) for m in ("video", "state", "language")}
                    )
                )
            )
            step = pol._to_vla_step_data(obs)
            base = {}
            for v in VIEWS:
                ims, _ = apply_with_replay(
                    eval_tf, [Image.fromarray(np.asarray(x)) for x in step.images[v]]
                )
                base[v] = ims[-1].permute(1, 2, 0).numpy().copy()
            h, w = base[VIEWS[0]].shape[:2]
            ys = np.round(np.linspace(0, h, GH + 1)).astype(int)
            xs = np.round(np.linspace(0, w, GW + 1)).astype(int)

            procs = []
            for kind, vi, i, j in variants:
                imgs = dict(base)
                if kind == "cell":
                    im = base[VIEWS[vi]].copy()
                    im[ys[i] : ys[i + 1], xs[j] : xs[j + 1]] = GREY
                    imgs[VIEWS[vi]] = im
                elif kind == "view":
                    imgs[VIEWS[vi]] = np.full_like(base[VIEWS[vi]], GREY)
                elif kind == "all":
                    imgs = {v: np.full_like(base[v], GREY) for v in VIEWS}
                s2 = copy.copy(step)
                s2.images = {v: [imgs[v]] for v in VIEWS}
                procs.append(
                    pol.processor([{"type": pir.MessageType.EPISODE_STEP.value, "content": s2}])
                )

            aims = np.zeros((a.seeds, len(variants), len(ROWS), 3))
            for s in range(a.seeds):
                for b0 in range(0, len(procs), a.batch):
                    chunk = procs[b0 : b0 + a.batch]
                    col_in = pir._rec_to_dtype(pol.collate_fn(chunk), dtype=torch.bfloat16)
                    with shared_noise(7919 * (s + 1) + t), torch.inference_mode():
                        res = pol.model.get_action(**col_in)
                    norm = res["action_pred"].float().cpu().numpy()
                    st = {
                        k: np.stack([np.asarray(step.states[k])] * len(chunk))
                        for k in modality["state"].modality_keys
                    }
                    dec = pol.processor.decode_action(norm, tag, st)
                    arm = np.concatenate([np.asarray(dec[k], np.float32)[:, :H] for k in keys], -1)[
                        ..., :14
                    ]
                    for r, row in enumerate(ROWS):
                        aims[s, b0 : b0 + len(chunk), r] = np.stack([xyz(q) for q in arm[:, row]])
            delta = np.linalg.norm(aims - aims[:, :1], axis=-1).mean(0)  # (variants, rows)
            noise = (
                np.linalg.norm(aims[0, 0] - aims[-1, 0], axis=-1)
                if a.seeds > 1
                else np.full(len(ROWS), np.nan)
            )
            cells = delta[1 : 1 + n_cell].reshape(len(VIEWS), GH, GW, len(ROWS))
            views = delta[1 + n_cell : 1 + n_cell + len(VIEWS)]
            hole = xyz(gt[rel, :14])
            for k, val in (
                ("ep", ep),
                ("off", o),
                ("rel", rel),
                ("cells", cells),
                ("views", views),
                ("allv", delta[-1]),
                ("noise", noise),
                ("images", np.stack([base[v] for v in VIEWS])),
                ("aim_base", aims[:, 0]),
                ("hole", hole),
            ):
                out[k].append(val)
            print(
                f"ep {ep} off {o:+d}: noise {noise[0]:.1f}/{noise[1]:.1f} mm | all-views {delta[-1][0]:.1f}/{delta[-1][1]:.1f} | "
                + " ".join(
                    f"{VIEWS[v]} view {views[v][1]:.1f} cellmax {cells[v, ..., 1].max():.1f}"
                    for v in range(len(VIEWS))
                )
                + f" | base aim->hole {np.linalg.norm(aims[:, 0, 1] - hole, axis=-1).mean():.1f}",
                flush=True,
            )
        del traj

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(a.out_dir) / f"occ_s{si}.npz", **{k: np.asarray(v) for k, v in out.items()}
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
