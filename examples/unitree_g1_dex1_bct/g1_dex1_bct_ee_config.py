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

"""BCT end-effector config: ABSOLUTE wrist poses (xyz + euler) + grippers.

Dataset: G1_Dex1_BCT_subtask_ee (14 dims = 6 left ee + 6 right ee + 2 grippers).
Each ee block is [x, y, z, roll, pitch, yaw]. The euler channels are continuous
in this data (max frame-to-frame jump 1.17 rad, no wraparound), so they are
trained as plain normalized vectors (NON_EEF/DEFAULT) rather than the 9d
xyz+rot6d EEF path, which would need a dataset relayout. Upper body only —
lower body is handled by the balance controller at deployment.
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


ABS_VEC = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

g1_dex1_bct_ee_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_head", "cam_left_wrist", "cam_right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_ee",
            "right_ee",
            "left_gripper",
            "right_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "left_ee",
            "right_ee",
            "left_gripper",
            "right_gripper",
        ],
        action_configs=[ABS_VEC] * 4,
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_dex1_bct_ee_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
