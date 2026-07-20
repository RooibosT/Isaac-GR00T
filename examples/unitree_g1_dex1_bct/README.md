# G1 Dex1 — Building Children Table (BCT) subtask finetune

Finetunes GR00T N1.7 on real teleop data from
[BitRobot/G1_WBT_Dex1_Building-Children-Table](https://huggingface.co/datasets/BitRobot/G1_WBT_Dex1_Building-Children-Table)
(upper body: xr_teleoperate, lower body: HOMIE), segmented into 5 manipulation
subtasks and converted to two action-space variants:

| dataset (local, `~/.cache/lerobot/BitRobot/`) | dims | layout |
|---|---|---|
| `G1_Dex1_BCT_subtask_joint` | 31 | 12 legs + 3 waist + 7+7 arms + 2 grippers (q_7..q_35 of the 36-dim source) |
| `G1_Dex1_BCT_subtask_ee`    | 14 | left [xyz+rpy] + right [xyz+rpy] + 2 grippers |

Both: 1338 episodes, 270,668 frames @15 fps (source was 30 fps), AV1 GOP-2 video,
4 cams (cam_0/cam_1 = head stereo pair, cam_2/cam_3 = left/right wrist).

## Dataset review findings (2026-07-19)

- **Subtask balance**: frames are already balanced — each of the 5 subtasks holds
  ~54k frames (20.0-20.1%). Episode counts differ (129-544) but GR00T samples at
  the frame level (`episode_sampling_rate` only shards, it does not subsample),
  so no extra reweighting is needed.
- **Dropped source tasks**: locomotion segments ("move to table", "move table
  base") and the full-task label were excluded during segmentation — appropriate
  for a manipulation finetune.
- **Lower body** (verified against the original 36-dim export with base odom):
  pelvis **height barely moves** — median z range 0.4-1.6 cm per segment, max
  7 cm, even on "flip table" (2-3 cm). The knee/ankle joint ranges are torso
  lean + balance corrections; the lean itself lives in the waist joints, which
  ARE in the action space. The real lower-body contribution is small
  **repositioning steps**: 22% of subtask segments move the base >10 cm in xy
  or >15° in yaw (flip median 8.5 cm / 6.3°; "pick table leg" is the worst at
  145/396 segments). Hence the primary config predicts **waist+arms+grippers
  only** and keeps legs as state; `joint_wbc` predicts legs too for WBC-replay
  experiments. There is no operator command channel in the export
  (`action.robot_q_desired[0:7]` equals the current base state), but a cmd_vel
  action (vx, vy, ωz) can be derived from odometry finite differences if a
  re-export is warranted — deployable through HOMIE's velocity interface.
- **Alignment**: subtask episode boundaries fall exactly on video keyframes
  (GOP=2 + boundary keyframes), and post-conversion parquet/video/meta lengths
  match exactly on sampled episodes.
- **EE rotations**: euler channels are continuous (max frame-to-frame jump
  1.17 rad, no ±π wraparound), safe to regress directly.
- **Short episodes**: 52 episodes (51 of them "rotate table base") are shorter
  than the 40-step action horizon and are dropped by the loader (~2% of frames).
- **Format**: the HF export is LeRobot v3.0; the GR00T loader needs v2.1.
  Convert once per variant with `scripts/lerobot_conversion/convert_v3_to_v2.py`
  (the original is kept alongside as `*_v3.0`).
- **Corrupt video segments (fixed 2026-07-20)**: 7 of 10,704 extracted episode
  videos (joint: cam_2 eps 165/501, cam_3 eps 771/1026; ee: cam_2 eps 146/1304,
  cam_3 ep 301) had undecodable leading AV1 packets — the corruption exists in
  the source v3 concatenated streams, so stream-copy extraction reproduced it.
  Repaired by re-encoding those exact segments (libx264, frame-exact, verified
  offset 0 against the source) and refreshing the split hardlinks. If you ever
  re-convert from scratch, re-run a decode scan over all episode videos.

## Usage

```bash
# one-time per variant: v3 -> v2.1 conversion (dedicated venv)
cd scripts/lerobot_conversion
.venv/bin/python convert_v3_to_v2.py --repo-id BitRobot/G1_Dex1_BCT_subtask_joint \
    --root /NHNHOME/WORKSPACE/chan/.cache/lerobot

# train (variant: joint | joint_h16 | joint_wbc | ee; default joint)
bash examples/unitree_g1_dex1_bct/run_finetune_bct.sh joint
```

`joint_h16` is the horizon-16 twin of `joint` (1.07 s lookahead @15 fps, no
short-episode drops, ~25% faster) — run it after `joint` for the first clean
h16-vs-h40 comparison at ABS: redblock's weak-h40 datapoint was A40 (RELATIVE,
confounded) and B40 was never closed-loop tested.

The runner installs the variant's `modality_*.json` into the dataset, creates a
train/val split (every 20th episode; subtasks are interleaved so the split stays
proportional), generates stats, and launches `examples/finetune.sh` with the
validated 2-GPU recipe: effective batch 256 (32×8), lr 1e-4, state dropout 0.2
(repo default; override with `STATE_DROPOUT=0.1` to mirror the redblock recipe),
horizon 40, 16 workers/GPU, `DATALOADER_FFMPEG_THREADS=1`. Default 25,000 steps
≈ 31 epochs (817 steps/epoch over 209k effective horizon-40 samples); pick the
final checkpoint by val MSE (`eval_steps` 2500, checkpoints kept every 2500).
