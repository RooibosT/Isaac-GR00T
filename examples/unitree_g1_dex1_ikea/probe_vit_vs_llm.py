"""Where is the ~27 mm wall -- in the ViT, or in the LLM on top of it?

The frozen read-out that saturates near 27 mm (hole 28.7, the robot's own wrist 26.9)
was taken from the LLM's last layer, which is what the action head sees. It cannot say
whether the ViT never encoded the detail or the 16 LLM layers washed it out. This
extracts the same samples at several depths in one pass and saves each depth
separately for fit_vit_vs_llm.py.

Samples and targets are the previous probe's exactly: episodes with at least two
right-gripper openings, the second being the insertion release `rel`; offsets
rel-90..rel-30; hole = FK(right wrist) of the commanded arm at rel, wrist = FK at t.
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

VIT_BLOCKS = (11, 23)  # pre-merge patch features, 1024-d; 23 is the last block
LLM_LAYERS = (3, 7, 11)  # outputs of layers 4/8/12 (deepstack is added after layers 1-3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--offsets", default="-90,-75,-60,-45,-30")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--shard", default="0/1", help="i/n: keep episodes with ep %% n == i")
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
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from url_groot_deploy.common.g1_kinematics import G1WristKinematics

    kin = G1WristKinematics("/root/01_IKEA/url_lerobot/xr_teleoperate", waist_zero=True)

    def xyz(q):
        return np.asarray(kin.both_wrist_poses(np.asarray(q, float), np.zeros(3))[1][:3]) * 1000.0

    tag = EmbodimentTag.resolve("new_embodiment")
    modality = MODALITY_CONFIGS[tag.value]
    loader = LeRobotEpisodeLoader(dataset_path=a.dataset_path, modality_configs=modality)
    pol = Gr00tPolicy(embodiment_tag=tag, model_path=a.checkpoint, device="cuda")
    probe = pir.Probe(pol, modality, tag)
    qwen = pol.model.backbone.model
    visual, lm = qwen.visual, qwen.language_model
    print(f"ViT blocks {len(visual.blocks)}, LLM layers {len(lm.layers)}", flush=True)

    grab = {}

    def keep(name):
        def hook(_module, _inputs, output):
            grab[name] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook

    for b in VIT_BLOCKS:
        visual.blocks[b].register_forward_hook(keep(f"vit_b{b + 1}"))
    visual.merger.register_forward_hook(keep("vit_merged"))
    for layer in LLM_LAYERS:
        lm.layers[layer].register_forward_hook(keep(f"llm_L{layer + 1}"))

    offs = [int(o) for o in a.offsets.split(",")]
    feats, Y, CUR, G, OFF = {}, [], [], [], []
    n_eps = len(loader) if a.episodes == 0 else min(a.episodes, len(loader))
    for ep in range(n_eps):
        if ep % sn != si:
            continue
        traj = loader[ep]
        gt = np.concatenate(
            [
                np.vstack([np.asarray(x, np.float32) for x in traj[f"action.{k}"]])
                for k in probe.keys
            ],
            -1,
        )
        ev, _ = pir.gripper_events(gt[:, 15])
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
            sub = {m: dict(parsed[m]) for m in ("video", "state", "language")}
            step = pol._to_vla_step_data(next(iter(pol._unbatch_observation(sub))))
            proc = pol.processor([{"type": pir.MessageType.EPISODE_STEP.value, "content": step}])
            col = pir._rec_to_dtype(pol.collate_fn([proc]), dtype=torch.bfloat16)
            grab.clear()
            with torch.no_grad():
                bi, _ = pol.model.prepare_input(col["inputs"])
                out = pol.model.backbone(bi)
            img = out["image_mask"][0]
            last = out["backbone_features"][0]
            row = {k: v for k, v in grab.items() if k.startswith("vit")}
            for layer in LLM_LAYERS:
                row[f"llm_L{layer + 1}_img"] = grab[f"llm_L{layer + 1}"][0][img]
            row["llm_L16_img"] = last[img]
            row["llm_L16_all"] = last  # the previous probe's feature, for the reproduction check
            for k, v in row.items():
                feats.setdefault(k, []).append(v.to(torch.float16).cpu().numpy())
            Y.append(xyz(gt[rel, :14]))
            CUR.append(xyz(gt[t, :14]))
            G.append(ep)
            OFF.append(o)
        del traj
        print(f"shard {si}: episode {ep}/{n_eps}  rel {rel}  samples {len(Y)}", flush=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"meta_s{si}.npz",
        Y=np.stack(Y),
        CUR=np.stack(CUR),
        G=np.array(G),
        O=np.array(OFF),
    )
    for k, lst in feats.items():
        shapes = sorted({x.shape for x in lst})
        n = min(x.shape[0] for x in lst)
        np.save(out_dir / f"{k}_s{si}.npy", np.stack([x[:n] for x in lst]))
        print(f"{k}: {len(lst)} samples, shapes {shapes}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
