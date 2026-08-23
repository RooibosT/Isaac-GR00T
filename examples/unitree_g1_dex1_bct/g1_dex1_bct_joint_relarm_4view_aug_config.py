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

"""BCT config: RELATIVE arms, 4 views, augmented 46-dim state, stall-free data.

State adds two blocks over the 31-dim joint state, both chosen to add signal
without adding leakage (see EXPERIMENTS.md for the measurements):
  * base_gravity — projected gravity in the pelvis frame. Carries torso
    roll/pitch and is yaw-invariant, unlike the raw quaternion whose w/z
    components are ~all session heading (yaw std 109 deg vs roll 0.6).
  * left/right_eef — wrist pose recomputed by FK from the arm joints in the
    same row. The recorded ee_state lagged the joints by ~10 frames (22 mm
    disagreement at lag 0), so it was a contradictory input; the FK version
    is exact at lag 0.

Root x,y,z are deliberately NOT in the state: x,y vary ~90x more between clips
than within one (a clip-ID shortcut) and z is a deterministic function of the
leg joints already present.

Dataset: *_aug_clean, i.e. teleop stalls removed and subtasks re-balanced.
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

g1_dex1_bct_joint_relarm_4view_aug_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_head",
            "cam_head_right",
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
            "waist",
            "left_arm",
            "right_arm",
            "left_gripper",
            "right_gripper",
        ],
        action_configs=[
            ABS_JOINT,  # waist
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
    g1_dex1_bct_joint_relarm_4view_aug_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
