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

"""BCT joint-space config: RELATIVE arms (+ ABSOLUTE waist/grippers) on FOUR views.

Two changes over `joint_30hz` (absolute, 3 views), both expected to help:
  * action rep: arms RELATIVE — won the 2026-08-16 open-loop A/B
    (MSE 0.0270 vs 0.0295, first-5 MAE 0.0282 vs 0.0422 = -33%)
  * vision: adds `cam_head_right` (cam_1), completing the head stereo pair.
    Depth cues around the table/leg contact should help the phases where a
    single head view is ambiguous.

Confounded on purpose (both knobs at once); if this loses, compare against the
3-view relarm run to attribute. Trained on a long schedule and selected by
open-loop scan, not eval_loss — see EXPERIMENTS.md.
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

g1_dex1_bct_joint_relarm_4view_config = {
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
    g1_dex1_bct_joint_relarm_4view_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
