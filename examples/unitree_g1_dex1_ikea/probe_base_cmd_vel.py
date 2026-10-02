"""Score the base velocity command, which `scan_ikea.py` does not look at.

That scan reports `mae_arm` over keys ending in `_arm`, `mae_grip` over keys containing
`gripper`, and `ee_mm` from FK on the arm block. `base_cmd_vel` lands in none of them; it
reaches only `mse`, averaged over all 19 dims, where the one live base dim is 1/19 of the
number and the two dead ones drag it toward zero. So a model that never commands the base
scores exactly like one that commands it correctly.

**Why the average is not enough either.** In `stage3_rc` the base is commanded on 8.8% of
frames. A policy that always emits zero is right 91.2% of the time, so its mean absolute
error looks small. What settles it is the split: the error *while commanded*, whether the
prediction actually moves off zero there, and how often it fires when it should not.

Reported per checkpoint:
  * per-dim MAE over the first 8 steps of the chunk (what a deployment executes),
  * for the live dim, MAE and mean prediction on commanded windows vs idle ones,
  * recall (commanded windows predicted past FIRE) and false-positive rate (idle ones),
  * correlation between predicted and ground-truth over every first-8 step,
  * the dead dims' predicted magnitude, which should be ~0 -- they normalize against a
    degenerate q01==q99 range floored at 1e-8, so the model can only emit a constant.

Usage:
    python probe_base_cmd_vel.py --checkpoint <dir> --dataset <val dir> [--stride 3]
"""

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np


EX = Path("/root/01_IKEA/Isaac-GR00T/examples/unitree_g1_dex1_ikea")
sys.path.insert(0, str(EX))
sys.path.insert(0, "/root/01_IKEA/Isaac-GR00T")

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS  # noqa: E402
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402
import scan_ikea  # noqa: E402  -- also applies its tokenizer patch


BASE_KEY = "base_cmd_vel"
COMMANDED = 1e-6  # |ground truth| above this counts as a commanded step
FIRE = 0.05  # |prediction| above this counts as the policy asking the base to move
FIRST = 8  # a deployment executes roughly the first 8 steps of each chunk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=EX / "g1_dex1_ikea_absarm_basevel_config.py")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()

    importlib.import_module(a.config.stem)
    tag = EmbodimentTag.resolve(a.embodiment_tag)
    modality = MODALITY_CONFIGS[tag.value]
    action_keys = modality["action"].modality_keys
    assert BASE_KEY in action_keys, f"{BASE_KEY} not in the config's action keys: {action_keys}"
    horizon = len(modality["action"].delta_indices)

    loader = LeRobotEpisodeLoader(dataset_path=str(a.dataset), modality_configs=modality)
    windows = scan_ikea.build_windows(loader, tag, horizon, a.stride)
    print(f"{len(windows)} windows, stride {a.stride}", flush=True)

    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(a.checkpoint), device="cuda")
    policy.model.action_head.num_inference_timesteps = a.denoising_steps

    from copy import deepcopy

    import torch

    preds, gts, offsets = [], [], None
    for i, (_, _, obs, gt) in enumerate(windows):
        torch.manual_seed(20260718 + i)
        chunk, _ = policy.get_action(deepcopy(obs))
        if offsets is None:
            offsets, s = {}, 0
            for k in action_keys:
                d = np.asarray(chunk[k]).shape[-1]
                offsets[k] = (s, s + d)
                s += d
        lo, hi = offsets[BASE_KEY]
        preds.append(np.asarray(chunk[BASE_KEY])[0][:FIRST])
        gts.append(gt[:FIRST, lo:hi])
    P, G = np.stack(preds), np.stack(gts)  # [windows, FIRST, 3]
    print(f"\n{BASE_KEY} occupies action dims {offsets[BASE_KEY][0]}:{offsets[BASE_KEY][1]}")

    live = [k for k in range(G.shape[-1]) if np.abs(G[..., k]).max() > COMMANDED]
    dead = [k for k in range(G.shape[-1]) if k not in live]
    print(f"dims carrying any command in this val: {live} | flat: {dead}")
    print(f"\n{'dim':>4} {'MAE first8':>11} {'gt |max|':>9} {'pred |max|':>11}")
    for k in range(G.shape[-1]):
        print(
            f"{k:4d} {np.abs(P[..., k] - G[..., k]).mean():11.5f} "
            f"{np.abs(G[..., k]).max():9.4f} {np.abs(P[..., k]).max():11.5f}"
        )

    out = {"checkpoint": a.checkpoint.name, "windows": len(windows), "live_dims": live}
    for k in live:
        g, p = G[..., k], P[..., k]
        cmd = np.abs(g).max(axis=1) > COMMANDED  # windows whose first 8 steps command motion
        rec = {
            "mae_all": float(np.abs(p - g).mean()),
            "mae_commanded": float(np.abs(p - g)[cmd].mean()) if cmd.any() else None,
            "mae_idle": float(np.abs(p - g)[~cmd].mean()) if (~cmd).any() else None,
            "pred_mean_commanded": float(p[cmd].mean()) if cmd.any() else None,
            "gt_mean_commanded": float(g[cmd].mean()) if cmd.any() else None,
            "pred_mean_idle": float(p[~cmd].mean()) if (~cmd).any() else None,
            "recall_fire": float((np.abs(p[cmd]).max(axis=1) > FIRE).mean()) if cmd.any() else None,
            "false_fire_idle": float((np.abs(p[~cmd]).max(axis=1) > FIRE).mean())
            if (~cmd).any()
            else None,
            "corr": float(np.corrcoef(p.ravel(), g.ravel())[0, 1]),
            "commanded_windows": int(cmd.sum()),
            "idle_windows": int((~cmd).sum()),
        }
        out[f"dim{k}"] = rec
        print(f"\n--- dim {k} (the live base command)")
        print(f"  windows commanded {rec['commanded_windows']} / idle {rec['idle_windows']}")
        print(
            f"  MAE   all {rec['mae_all']:.5f} | commanded {rec['mae_commanded']:.5f} | idle {rec['mae_idle']:.5f}"
        )
        print(
            f"  mean  gt commanded {rec['gt_mean_commanded']:+.4f} -> pred {rec['pred_mean_commanded']:+.4f}"
            f" | idle pred {rec['pred_mean_idle']:+.5f}"
        )
        print(
            f"  fires past {FIRE}: on commanded windows {100 * rec['recall_fire']:.1f}% "
            f"| on idle windows {100 * rec['false_fire_idle']:.1f}%"
        )
        print(f"  corr(pred, gt) over every first-{FIRST} step: {rec['corr']:+.3f}")
    for k in dead:
        out[f"dim{k}"] = {
            "pred_abs_max": float(np.abs(P[..., k]).max()),
            "pred_abs_mean": float(np.abs(P[..., k]).mean()),
        }
        print(
            f"\n--- dim {k} (flat in this val): pred |max| {np.abs(P[..., k]).max():.6f}, "
            f"|mean| {np.abs(P[..., k]).mean():.6f}"
        )
    if a.output:
        a.output.write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
