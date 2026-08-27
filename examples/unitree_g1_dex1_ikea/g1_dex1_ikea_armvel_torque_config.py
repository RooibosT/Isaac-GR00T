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

"""IKEA ablation: the 60-dim arm-velocity config plus arm joint torque (state -> 74).

Everything else is identical to g1_dex1_ikea_armvel_config, so any difference in
the open-loop scan is attributable to these 14 dims alone. Needs a dataset built
from the torque re-export (`URL-RFM/IKEA_table_assembly_torque`), which is the
same 178 episodes and the same video files as the original -- state[:86] and
action[:19] are bit-identical -- with 31 torque columns appended.

Torque is genuinely new, unlike torso_gravity (99.85% recoverable from state
already present and therefore dropped): a ridge fit from the whole 60-dim
arm-velocity state recovers only R2 0.65 of left arm torque and 0.68 of right.

What is left out and why. Gripper torque is 87-88% recoverable and, more to the
point, the gripper is the metric this run is judged on -- the arm-velocity config
left gripper *velocity* out for exactly that reason, and putting a contact signal
back in would spoil the same measurement. Waist torque is 89% recoverable. Legs
torque is the standing load on a robot that does not walk.

The shortcut this opens, and how to read the scan because of it. Joint torque on
a position-controlled arm tracks the servo error, which is close to the relative
action target the model predicts, so torque leaks the answer more directly than
velocity does at the first step: a linear fit of the relative left-arm target
gives R2 0.59 from torque against 0.40 from velocity. But it decays much faster,
because torque reflects the command being executed now and not where the arm is
headed:

    horizon step        1      4      8     16     28     40
    from arm_vel      0.40   0.62   0.60   0.41   0.23   0.14
    from arm_torque   0.59   0.34   0.20   0.11   0.07   0.07
    from both         0.86   0.83   0.69   0.45   0.27   0.20

Over the first 8 steps -- the window deployment actually executes -- torque adds
about 9 points of R2 on top of velocity, so it is a real but modest extra
shortcut. Judge this run the same way the arm-velocity one is judged: on the late
chunk steps and on the gripper, not on the first-8 arm error, and treat a gain
that appears only at step 1-4 as the shortcut rather than as skill.
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
REL_JOINT = ActionConfig(
    rep=ActionRepresentation.RELATIVE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

g1_dex1_ikea_armvel_torque_config = {
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
            "left_arm_vel",
            "right_arm_vel",
            "left_arm_torque",
            "right_arm_torque",
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
        ],
        action_configs=[
            REL_JOINT,  # left_arm
            REL_JOINT,  # right_arm
            ABS_JOINT,  # left_gripper
            ABS_JOINT,  # right_gripper
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    g1_dex1_ikea_armvel_torque_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
