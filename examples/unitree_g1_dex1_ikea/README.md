# Unitree G1 + Dex1 — IKEA table assembly

Fine-tuning GR00T N1.7 on [`carroll511/IKEA_table_assembly`](https://huggingface.co/datasets/carroll511/IKEA_table_assembly):
a G1 with Dex1 grippers assembling a children's table from a fixed standing position.

Measurements, rejected ablations and the reasoning behind every setting are in
**[EXPERIMENTS.md](EXPERIMENTS.md)**. This file is just how to run things.

## Files

| | |
|---|---|
| `convert_ikea_v3_to_v2.py` | LeRobot v3.0 → v2.1 re-layout + session-level train/val split |
| `sessions.json` | 27 recording-session boundaries, detected from stance and head-tilt jumps |
| `make_waist_action_variant.py` | 19-dim action variant that lines up with the BCT model's output slots |
| `g1_dex1_ikea_relarm_3view_aug_config.py` | **the main config** — 46-dim state, 16-dim action |
| `g1_dex1_ikea_waistact_config.py` | 19-dim variant (waist prepended); use with the `_wa_*` datasets |
| `g1_dex1_ikea_relarm_3view_torsograv_config.py` | rejected ablation, kept for the record |
| `g1_dex1_ikea_armvel_config.py` | **adopted** — adds arm joint velocities (state → 60); velocities sit mid-list |
| `g1_dex1_ikea_armvel_bctalign_config.py` | same 60 dims with the velocities appended, so columns 0-45 still match a BCT checkpoint |
| `run_finetune_ikea.sh` | training launcher |
| `scan_ikea.py` | open-loop checkpoint scan — **this is what selects the model**, not `eval_loss` |
| `scan_when_done.sh` | wait for a run to exit, then scan it on two GPUs |

## Setup

`torchcodec` needs FFmpeg 7 shared libraries, which the venv does not ship. The
launcher adds them if `~/micromamba/envs/ffmpeg7` exists; for standalone scripts:

```bash
export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg7/lib:$LD_LIBRARY_PATH"
source .venv/bin/activate
```

## 1. Get and convert the dataset

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('carroll511/IKEA_table_assembly', repo_type='dataset', \
  local_dir='datasets/carroll511/IKEA_table_assembly')"

python examples/unitree_g1_dex1_ikea/convert_ikea_v3_to_v2.py \
    --src datasets/carroll511/IKEA_table_assembly \
    --out datasets/carroll511/G1_Dex1_IKEA_table_30hz \
    --sessions-file examples/unitree_g1_dex1_ikea/sessions.json \
    --val-sessions 3,12,19,23
```

Produces the full set plus `_train` (250 ep) and `_val` (26 ep); the splits
hardlink their videos, so all three cost 3.0 GB total. `--verify` (on by default)
decodes sampled frames out of the cut clips and compares them against the source —
this is the only real check that the seek arithmetic is right, and it matters:
146 of 276 episodes start on an odd frame, where a stream copy would land one
frame early.

Then statistics for each split (the loader requires `meta/stats.json`; it does
not compute it on the fly):

```bash
for s in _train _val; do
  python -m gr00t.data.stats \
    --dataset-path datasets/carroll511/G1_Dex1_IKEA_table_30hz$s \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/unitree_g1_dex1_ikea/g1_dex1_ikea_relarm_3view_aug_config.py
done
```

## 2. Train

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash examples/unitree_g1_dex1_ikea/run_finetune_ikea.sh --use-ddp --ddp-comm-bf16
```

Defaults: 2 GPUs, effective batch 64 (global 16 × accum 4), 20,000 steps,
checkpoint every 2,000. ~8 h on 2× A100.

**Two GPUs, not four, and always `--ddp-comm-bf16`.** At a fixed effective batch
more ranks only split the same 64 samples further while the gradient all-reduce
stays once per step; measured 1 GPU 2.46 / 2 GPU 1.46 / 4 GPU 2.24 s/step on this
NVLink-less host. Spare GPUs are better spent on parallel ablations — three
concurrent 2-GPU runs showed no slowdown.

Useful overrides:

| var | for |
|---|---|
| `CONFIG` | ablations — state/action key changes need no data rebuild |
| `DATASET_ROOT` | the `_wa` variant |
| `BASE_MODEL_PATH` | warm-starting from another checkpoint |
| `EXP_SUFFIX` | output directory suffix |
| `MASTER_PORT` | **required** when running concurrently (default 29500 collides) |
| `COLOR_JITTER_PARAMS`, `STATE_DROPOUT`, `MAX_STEPS`, `SAVE_STEPS` | |

The launcher refuses to start if free disk is below what the run needs
(~12 GB per checkpoint). This is not paranoia: a full disk kills a run mid-save,
hours in, and `--save-only-model` means it cannot be resumed.

## 3. Select a checkpoint

`eval_loss` does not track action quality — it rose after ~7.5k steps in five BCT
runs while open-loop accuracy kept improving. Scan instead:

```bash
python examples/unitree_g1_dex1_ikea/scan_ikea.py \
    --checkpoints-dir outputs/<exp>/<exp> \
    --dataset-path datasets/carroll511/G1_Dex1_IKEA_table_30hz_val \
    --config examples/unitree_g1_dex1_ikea/g1_dex1_ikea_relarm_3view_aug_config.py \
    --stride 10 --output outputs/<exp>/<exp>/scan.json
```

Or queue it behind a running job (`--steps` shards the checkpoint list, so this
uses two GPUs):

```bash
bash examples/unitree_g1_dex1_ikea/scan_when_done.sh <exp_name> <config.py> 0 1
```

Reports MSE, arm MAE, wrist-position error via FK, and the first-5/8/16-step
figures, split per task. Read the **8-step** numbers — that is what deployment
executes before re-inferring. Judging on the whole 40-step chunk understates a
change that helps early: async deployment executes ~8 steps after latency
compensation and spends the rest on RTC overlap, so steps past ~16 never run.

`--zero-state-keys <keys>` zeroes those keys after normalization — the same thing
`--state-dropout-keys` does in training. Scanning a run with and without it
measures how much that run actually leans on those inputs, which an A/B between
two runs cannot separate from a general accuracy difference.

Two traps: a 19-dim run must be scanned against the `_wa_val` split (`VAL=` for
the waiter), and MSE is not comparable across different action widths — the
near-constant waist dilutes it. Compare arm / EE8 / gripper instead.

Differences below these bands are noise (measured from one run's own 16k/18k/20k
spread): MSE 2.0%, arm 0.36%, EE8 0.94%, gripper 4.9%.

## Current model

**`RooibosT/gr00t-n1.7-g1-dex1-ikea-relarm-30hz-h40`** (private) — 20,000 steps.
Executed-window accuracy: 5 steps 1.39° · **8 steps 1.80°** · 16 steps 2.67°.
