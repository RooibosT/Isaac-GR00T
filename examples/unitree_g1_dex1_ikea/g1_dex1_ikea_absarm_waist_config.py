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

"""`g1_dex1_ikea_absarm_3view_aug_config` plus the waist yaw command in the action.

For `RooibosT/nature_pouch_new`, the first IKEA recording made with the recorder's [w] waist
turn: [w] ramps the waist to -pi/4 (45 deg to the robot's right) and [w] again brings it back.
Every earlier IKEA set held the waist at its startup pose and has no waist action, so every
other config here outputs arms and grippers only and could never produce the turn.

**Action is 17 dims**, not 16: arms 7+7 and grippers 1+1 as before, plus `waist_yaw` 1,
appended last so dims 0-15 keep the 16-dim models' order. State (46-dim), views and horizon
are untouched.

**What is in that block** (measured over all 43,506 frames):

- range -0.7854 .. 0, q01 / q99 = -0.7854 / 0, so the q01/q99 normalization spans the whole
  turn and nothing is clipped.
- every one of the 99 episodes turns out once and back once; 50% of frames are off zero
  (20% held at -pi/4, 30% ramping). The ramp is 0.35 rad/s median, so a full turn is ~2.3 s,
  longer than the 40-step (1.33 s) action horizon.
- the measured waist yaw, state `waist` dim 0, follows the command 1 frame behind (corr
  +1.000, worst gap 0.041 rad). The model therefore sees where the waist is; what it has to
  learn from the images is when to start and when to return.

`ABSOLUTE`, like the arms: it is the ramped joint target `teleop_ikea.py` puts on
`rt/arm_sdk` slot 12. `RELATIVE` would need a state block of the same name and width, and the
state's `waist` is 3-wide (yaw, roll, pitch) while the action carries yaw alone.

Two other blocks in this set are degenerate and stay as the base config has them:
`base_cmd_vel` is constant zero and is not in the action; `left_gripper` is held open (q01 =
q99 = 5.4), and with the range floored at 1e-8 it normalizes to a constant, so the model
emits 5.4 there.

**Deployment must widen its action unpacking to 17** and send the last dim as the waist yaw
target on `rt/arm_sdk` slot 12 (roll and pitch were never commanded). A client built for the
16-dim IKEA models will drop it and the robot will never turn.
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

g1_dex1_ikea_absarm_waist_config = {
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
            "waist_yaw",
        ],
        action_configs=[
            ABS_JOINT,  # left_arm
            ABS_JOINT,  # right_arm
            ABS_JOINT,  # left_gripper
            ABS_JOINT,  # right_gripper
            ABS_JOINT,  # waist_yaw -- the [w] turn request, rt/arm_sdk slot 12
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    g1_dex1_ikea_absarm_waist_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
