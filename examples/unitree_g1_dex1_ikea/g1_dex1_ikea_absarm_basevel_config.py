# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`g1_dex1_ikea_absarm_3view_aug_config` plus the base velocity command in the action.

For `RooibosT/stage3_rc`, where the robot drives to the stage-3 location before it
touches the table. Every other IKEA recording here was stationary work with
`action.base_cmd_vel` exactly zero, so every other config drops it; this one keeps it.

**Action is 19 dims**, not 16: arms 7+7 and grippers 1+1 as before, plus `base_cmd_vel`
3. State, views and horizon are untouched, so a run against
`g1_dex1_ikea_absarm_3view_aug_config` isolates the base command alone.

**What is actually in that block** (measured over all 41,027 frames of stage3_rc):

| dim | nonzero frames | range | q01 / q99 | what it drives |
|---|---:|---|---|---|
| 16 | 0.2% | -0.151 .. 0 | 0.000 / 0.000 | nothing measurable (|corr| <= 0.09 with any base axis) |
| 17 | **8.8%** | -0.200 .. 0 | -0.200 / -0.000 | measured `base_lin_vel` y, corr **+0.75** |
| 18 | 0.0% | constant 0 | 0.000 / 0.000 | — |

So one of the three dims carries the signal, and it is a real one: it fires in **101 of
121 episodes** (median 9% of frames within those, up to 59%), and when it fires the base
actually moves — measured |lin| 0.165 against 0.011 idle.

Dims 16 and 18 are kept anyway because `meta/modality.json` declares `base_cmd_vel` as one
3-wide block and a modality key cannot be split. They are harmless rather than useful:
normalization is q01/q99 (`finetune_config.use_percentiles = True`) with the range floored
at 1e-8 and outputs clipped to [-1, 1], so a degenerate dim normalizes to a constant and
un-normalizes back to ~0. The model simply learns to emit zero there. Do not read a
prediction on those two as a command the data taught.

`ABSOLUTE`, not `RELATIVE`: a velocity command has no anchor in the state vector to take a
delta against (the 46-dim state carries `base_gravity` but neither `base_lin_vel` nor
`base_ang_vel`), and `RELATIVE` here would need relative-action statistics for a key that
has no matching state block.

**Deployment must widen its action unpacking to 19.** A client built for the 16-dim IKEA
models will silently mis-slice this model's output; the extra three are
`[base_cmd_vel_x, base_cmd_vel_y, base_cmd_vel_yaw]` in the dataset's order, of which only
the middle one was ever commanded.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


ABS_JOINT = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

g1_dex1_ikea_absarm_basevel_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_left_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "legs",
            "waist",
            "left_arm",
            "right_arm",
            "left_gripper",
            "right_gripper",
            "base_gravity",
            "left_eef",
            "right_eef",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_gripper",
            "right_gripper",
            "base_cmd_vel",
        ],
        action_configs=[
            ABS_JOINT,  # left_arm
            ABS_JOINT,  # right_arm
            ABS_JOINT,  # left_gripper
            ABS_JOINT,  # right_gripper
            ABS_JOINT,  # base_cmd_vel -- absolute velocity command, no state anchor
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    g1_dex1_ikea_absarm_basevel_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
