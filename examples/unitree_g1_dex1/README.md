# Unitree G1 + Dex1 (sim) — pick red block

Fine-tuning setup for [RooibosT/g1_pick_redblock_dex1_sim_merged_107demo](https://huggingface.co/datasets/RooibosT/g1_pick_redblock_dex1_sim_merged_107demo)
(LeRobot v3.0, `Unitree_G1_Dex1_Sim`, 107 episodes / 95,143 frames @ 30 fps).

- **State/action (16 dims):** `left_arm` (7) + `right_arm` (7) + `left_gripper` (1) + `right_gripper` (1)
- **Cameras:** `cam_left_high` (third-person), `cam_left_wrist`, `cam_right_wrist` (640x480, AV1)
- **Task:** "Pick up the red cube and place it in the yellow region."

## 1. Convert v3.0 -> v2.1

The GR00T loader consumes LeRobot v2.1. Convert with the bundled script (own venv, see
`scripts/lerobot_conversion/README.md`):

```bash
cd scripts/lerobot_conversion
./.venv/bin/python convert_v3_to_v2.py \
  --repo-id RooibosT/g1_pick_redblock_dex1_sim_merged_107demo \
  --root <repo>/demo_data
```

## 2. Add modality.json

```bash
cp examples/unitree_g1_dex1/modality.json \
   demo_data/RooibosT/g1_pick_redblock_dex1_sim_merged_107demo/meta/modality.json
```

## 3. Generate statistics

```bash
python gr00t/data/stats.py \
  --dataset-path demo_data/RooibosT/g1_pick_redblock_dex1_sim_merged_107demo \
  --embodiment-tag new_embodiment \
  --modality-config-path examples/unitree_g1_dex1/g1_dex1_config.py
```

## 4. Fine-tune

> The VLM backbone `nvidia/Cosmos-Reason2-2B` is a gated HF repo — request access at
> https://huggingface.co/nvidia/Cosmos-Reason2-2B and authenticate (`hf auth login` or `HF_TOKEN`)
> before the first run.

```bash
bash examples/finetune.sh \
  --base-model-path <path-to>/GR00T-N1.7-3B \
  --dataset-path demo_data/RooibosT/g1_pick_redblock_dex1_sim_merged_107demo \
  --embodiment-tag new_embodiment \
  --modality-config-path examples/unitree_g1_dex1/g1_dex1_config.py \
  --output-dir <output-dir>
```

## 5. Action-representation ablations

`variants/` holds alternative modality configs; the baseline (`g1_dex1_config.py`)
is variant **A** (RELATIVE joint arms + ABSOLUTE grippers, horizon 16):

| Variant | Arms     | Grippers | Horizon |
|---------|----------|----------|---------|
| A       | RELATIVE | ABSOLUTE | 16      |
| B       | ABSOLUTE | ABSOLUTE | 16      |
| A40     | RELATIVE | ABSOLUTE | 40      |
| B40     | ABSOLUTE | ABSOLUTE | 40      |

Run B, A40, B40 sequentially (5000 steps each; refuses to start while GPUs are busy):

```bash
bash examples/unitree_g1_dex1/run_ablations.sh
```

Relative-action stats are fingerprinted per config and regenerated automatically at
training start, so no manual stats step is needed between variants. Compare variants
by simulator success rate, not train loss (different representations use different
normalization statistics).
