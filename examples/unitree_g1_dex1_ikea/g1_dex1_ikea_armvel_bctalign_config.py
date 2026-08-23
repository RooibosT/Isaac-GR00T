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

"""IKEA ablation: arm joint velocities appended *after* the base 46 dims.

Same 60-dim state as g1_dex1_ikea_armvel_config, different column order, and the
order is the whole point. The state encoder is one Linear over the concatenated,
zero-padded state vector, so a key's meaning is its column range. The BCT
fine-tune records its `new_embodiment` state as

    legs, waist, left_arm, right_arm, left_gripper, right_gripper,
    base_gravity, left_eef, right_eef

which is exactly the base IKEA config, so warm-starting from BCT lines up
perfectly there. `g1_dex1_ikea_armvel_config` inserts the two velocity blocks
after `right_arm`, which shifts the last five keys by 14 columns and hands BCT's
gripper/gravity/eef weights to velocity columns. Appending instead keeps
columns 0-45 byte-identical to what BCT learned and gives the velocities fresh
columns 46-59.

Use this one when BASE_MODEL_PATH is a BCT checkpoint. Use the middle-insertion
`g1_dex1_ikea_armvel_config` when warm-starting from stock GR00T, where there is
no layout to preserve and it is the config R3/R4 were trained with.
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

g1_dex1_ikea_armvel_config = {
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
            "left_arm_vel",
            "right_arm_vel",
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

register_modality_config(g1_dex1_ikea_armvel_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
