"""Where does the action head look while the leg goes into the hole?

The DiT reads the backbone through two kinds of cross-attention block: blocks 2, 6,
10 and 14 attend to image tokens only, the ones between them to text. Those image
blocks' weights are the model's own account of which image tokens the action tokens
draw on. This captures them at frames leading up to the insertion release (the
second right-gripper opening) and lays them back onto each view's token grid.

Two limits on reading the maps. A token is not a patch: before the DiT sees it, it
has been mixed by 16 LLM layers and 4 self-attention blocks, so a bright cell says
"this token", not "these pixels". And attention is not attribution: a token can
draw weight and still barely move the action.
"""

import argparse
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch


EX = Path("/root/01_IKEA/Isaac-GR00T/examples/unitree_g1_dex1_ikea")
sys.path.insert(0, str(EX))
sys.path.insert(0, "/root/01_IKEA/url_lerobot")

VIEWS = ["cam_left_high", "cam_left_wrist", "cam_right_wrist"]
TILE_W, TILE_H = 352, 256  # what the Qwen processor resizes a 340x256 eval crop to


class Capture:
    """Stands in for the attention processor of one block: same output, and it keeps
    the head-averaged weights."""

    def __init__(self, inner, store, block):
        self.inner, self.store, self.block = inner, store, block
        self.checked = False

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        out = self.inner(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs
        )
        assert attn.spatial_norm is None and attn.group_norm is None and not attn.norm_cross
        b, h = hidden_states.shape[0], attn.heads
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        d = k.shape[-1] // h
        q = q.view(b, -1, h, d).transpose(1, 2)
        k = k.view(b, -1, h, d).transpose(1, 2)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        mask = attention_mask.bool().reshape(b, -1)[:, -k.shape[2] :]
        logits = q.float() @ k.float().transpose(-1, -2) / math.sqrt(d)
        w = logits.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        if not self.checked:
            v = attn.to_v(encoder_hidden_states).view(b, -1, h, d).transpose(1, 2).float()
            mine = (w @ v).transpose(1, 2).reshape(b, -1, h * d)
            mine = attn.to_out[0](mine.to(attn.to_out[0].weight.dtype)) / attn.rescale_output_factor
            err = ((mine.float() - out.float()).abs().max() / out.float().abs().max()).item()
            assert err < 1e-2, (
                f"recomputed attention disagrees with the processor: rel err {err:.3g}"
            )
            print(f"block {self.block}: recomputed output matches, rel err {err:.2e}", flush=True)
            self.checked = True
        self.store.append((self.block, w.mean(1), mask))
        return out


def heat(x):
    """[0, 1] -> RGB, a plain jet ramp."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def tile(img, grid, share, label):
    """Overlay one view's token map, normalised within the view, cells kept square."""
    base = Image.fromarray(img).resize((TILE_W, TILE_H), Image.BILINEAR)
    gh, gw = grid.shape
    norm = grid / grid.max()
    cells = Image.fromarray(heat(norm)).resize((TILE_W, TILE_H), Image.NEAREST)
    out = Image.blend(base, cells, 0.45)
    dr = ImageDraw.Draw(out)
    cy, cx = np.unravel_index(np.argmax(grid), grid.shape)
    cw, ch = TILE_W / gw, TILE_H / gh
    dr.rectangle(
        [cx * cw, cy * ch, (cx + 1) * cw - 1, (cy + 1) * ch - 1], outline=(255, 255, 255), width=2
    )
    dr.rectangle([0, 0, TILE_W, 16], fill=(0, 0, 0))
    dr.text(
        (4, 2),
        f"{label}  share {share * 100:4.1f}%  peak/mean {grid.max() / grid.mean():.1f}",
        fill=(255, 255, 255),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--offsets", default="-120,-60,-30,-15,-5,0")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("pir", EX / "probe_insert_release.py")
    pir = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pir)
    importlib.import_module(Path(a.config).stem)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.image_augmentations import apply_with_replay
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    tag = EmbodimentTag.resolve("new_embodiment")
    modality = MODALITY_CONFIGS[tag.value]
    assert list(modality["video"].modality_keys) == VIEWS, modality["video"].modality_keys
    loader = LeRobotEpisodeLoader(dataset_path=a.dataset_path, modality_configs=modality)
    pol = Gr00tPolicy(embodiment_tag=tag, model_path=a.checkpoint, device="cuda")
    probe = pir.Probe(pol, modality, tag)
    head = pol.model.action_head
    dit = head.model
    n_blocks = len(dit.transformer_blocks)
    img_blocks = [
        i for i in range(n_blocks) if i % 2 == 0 and i % (2 * dit.attend_text_every_n_blocks) != 0
    ]
    store = []
    for i in img_blocks:
        attn = dit.transformer_blocks[i].attn1
        print(
            f"block {i}: processor {type(attn.processor).__name__}, heads {attn.heads}", flush=True
        )
        attn.set_processor(Capture(attn.processor, store, i))
    H = head.action_horizon
    print(
        f"image blocks {img_blocks}; denoising steps {head.num_inference_timesteps}; horizon {H}",
        flush=True,
    )

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    offs = [int(o) for o in a.offsets.split(",")]
    summary = []
    samples = []
    n_eps = len(loader) if a.episodes == 0 else min(a.episodes, len(loader))
    for ep in range(n_eps):
        traj = loader[ep]
        keys = probe.keys
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
        rows = []
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
            proc = pol.processor([{"type": pir.MessageType.EPISODE_STEP.value, "content": step}])
            col_in = pir._rec_to_dtype(pol.collate_fn([proc] * a.seeds), dtype=torch.bfloat16)
            thw = col_in["inputs"]["image_grid_thw"][: len(VIEWS)].tolist()
            store.clear()
            torch.manual_seed(1000 + t)
            with torch.inference_mode():
                pol.model.get_action(**col_in)
            steps = len(store) // len(img_blocks)
            W = torch.stack([w for _, w, _ in store]).float()  # (steps*blocks, B, Tq, S)
            mask = store[0][2][0]
            W = W.view(steps, len(img_blocks), *W.shape[1:])[..., -H:, :]  # action queries only
            W = W[..., mask]  # image tokens, in prompt order
            per_block = W.mean(dim=(0, 2, 3)).cpu().numpy()  # (blocks, n_img)
            overall = per_block.mean(0)
            grids, tiles, shares = [], [], []
            start = 0
            for v, (tt, gh, gw) in zip(VIEWS, thw):
                n = tt * (gh // 2) * (gw // 2)
                grids.append(overall[start : start + n].reshape(gh // 2, gw // 2))
                start += n
            assert start == overall.size, (start, overall.size, thw)
            total = overall.sum()
            view_imgs = []
            for v, g in zip(VIEWS, grids):
                img, _ = apply_with_replay(
                    pol.processor.eval_image_transform,
                    [Image.fromarray(np.asarray(x)) for x in step.images[v]],
                )
                share = float(g.sum() / total)
                shares.append(share)
                arr = img[-1].permute(1, 2, 0).numpy()
                view_imgs.append(arr)
                tiles.append(tile(arr, g, share, f"{v} t{o:+d}"))
            row = Image.new("RGB", (TILE_W * len(VIEWS), TILE_H))
            for j, tl in enumerate(tiles):
                row.paste(tl, (j * TILE_W, 0))
            rows.append(row)
            gh2, gw2 = grids[0].shape
            samples.append(
                {
                    "ep": ep,
                    "off": o,
                    "rel": rel,
                    "images": np.stack(view_imgs),
                    "grids": np.stack(grids),
                    "block_grids": per_block.reshape(len(img_blocks), len(VIEWS), gh2, gw2),
                }
            )
            block_shares = [
                [
                    float(per_block[bi][s : s + n].sum() / per_block[bi].sum())
                    for s, n in ((0, 88), (88, 88), (176, 88))
                ]
                for bi in range(len(img_blocks))
            ]
            summary.append(
                {
                    "ep": ep,
                    "rel": rel,
                    "off": o,
                    "grid_thw": thw,
                    "view_share": shares,
                    "block_view_share": block_shares,
                    "grids": [g.tolist() for g in grids],
                }
            )
        sheet = Image.new("RGB", (TILE_W * len(VIEWS), TILE_H * len(rows)))
        for r, row in enumerate(rows):
            sheet.paste(row, (0, r * TILE_H))
        sheet.save(out_dir / f"ep{ep:03d}_rel{rel}.png")
        del traj
        print(
            f"episode {ep}: release {rel}, {len(rows)} frames -> {out_dir / f'ep{ep:03d}_rel{rel}.png'}",
            flush=True,
        )

    (out_dir / "summary.json").write_text(json.dumps(summary))
    np.savez_compressed(
        out_dir / "attn.npz",
        ep=np.array([x["ep"] for x in samples]),
        off=np.array([x["off"] for x in samples]),
        rel=np.array([x["rel"] for x in samples]),
        images=np.stack([x["images"] for x in samples]),
        grids=np.stack([x["grids"] for x in samples]),
        block_grids=np.stack([x["block_grids"] for x in samples]),
        blocks=np.array(img_blocks),
    )
    print(
        f"\nview share of image attention, mean over episodes ({len({s['ep'] for s in summary})} eps)"
    )
    print(f"{'offset':>7} " + " ".join(f"{v:>16}" for v in VIEWS))
    for o in offs:
        s = np.array([x["view_share"] for x in summary if x["off"] == o])
        if len(s):
            print(f"{o:>7} " + " ".join(f"{m * 100:15.1f}%" for m in s.mean(0)))
    print("DONE")


if __name__ == "__main__":
    main()
