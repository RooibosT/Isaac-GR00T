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

"""BCT joint-space config (primary): upper-body ABSOLUTE joint actions.

Dataset: G1_Dex1_BCT_subtask_joint (31 dims = 12 legs + 3 waist + 7+7 arms + 2 grippers).

Action = waist + arms + grippers only. Legs stay state-only on purpose:
they are outputs of the HOMIE balance policy (ankle chatter, small hip/knee
corrections), not operator commands, and joint-position leg targets cannot be
replayed on a balancing robot anyway — at deployment the lower body runs its
own balance controller. The model still conditions on leg state (crouch height
matters for e.g. "flip table"). ABSOLUTE arms follow the validated B-variant
recipe (B >> A on the redblock closed-loop eval). Horizon 40 = pretrained
native (2.67 s lookahead at this dataset's 15 fps).
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

g1_dex1_bct_joint_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_head", "cam_left_wrist", "cam_right_wrist"],
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
        action_configs=[ABS_JOINT] * 5,
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_dex1_bct_joint_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
