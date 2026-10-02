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

"""IKEA multi-task: 60-dim state (arm velocities) with the 19-dim base-commanding action.

The two halves already exist separately:

* `g1_dex1_ikea_absarm_armvel_config` adds `left_arm_vel` / `right_arm_vel` (7+7) to the
  46-dim state. The gain is the most reproduced result in this project -- arm8 -16.3% on
  RELATIVE targets (section 27), -15.1% on the subtask split (32), -18.0% on the rotate
  set (24), -8.1% on ABSOLUTE targets (28/32), and again on fliptable v2 (37).
* `g1_dex1_ikea_absarm_basevel_config` adds `base_cmd_vel` (3) to the 16-dim action, for
  `RooibosT/stage3_rc`, the one recording where the robot drives.

This config is both, for the three-set merge. Nothing else changes: same three views, same
horizon 40, same ABSOLUTE arms and grippers.

⚠️ **Inference must feed real `arm_dq`.** Zeroing the velocity block at deploy time makes a
60-dim model worse than the 46-dim baseline it was supposed to beat (section 16); the
deploy-side spec for these inputs is `IKEA_ARMVEL_SPEC`. Together with the 19-dim action
unpacking that `base_cmd_vel` forces, a client built for the 46/16 models needs both
changes before it can run this one.

⚠️ The velocity gain is smaller under ABSOLUTE targets than under relative ones -- about
half, by the measurement in section 32 -- because velocity is information about the delta
and only the relative parameterisation predicts the delta directly. It has never been
measured on top of a multi-task merge, which is what the run against this config is for.
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

g1_dex1_ikea_absarm_armvel_basevel_config = {
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
    g1_dex1_ikea_absarm_armvel_basevel_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
