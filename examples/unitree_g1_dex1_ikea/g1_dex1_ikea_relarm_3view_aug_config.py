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

"""IKEA table-assembly config: RELATIVE arms, 3 views, 46-dim augmented state.

This is the BCT winning recipe (examples/unitree_g1_dex1_bct, EXPERIMENTS.md)
carried over unchanged wherever the new dataset allows it. All nine state keys
of the 46-dim augmented state exist in this dataset under the same names and
widths, and `_ikea.eef_frame` in its modality.json declares the same FK
convention the BCT eef blocks were built with, so the state vector is identical
in meaning.

Two differences from BCT, both forced by what was recorded:

  * **No waist action.** `teleop_ikea.py` commands the arms only and holds the
    waist at its startup pose; measured range within an episode is 0.8 deg. BCT
    had to predict a waist that turned out to be a locomotion-controller output,
    and the deployment code drops it. Here it is simply absent.
  * **`base_cmd_vel` is excluded.** It is in the dataset's action block but is
    exactly zero on all 66,600 frames (this was stationary work), so it carries
    no signal and its q01/q99 normalization range is degenerate.

Action is therefore 16 dims: arms 7+7 RELATIVE, grippers 1+1 ABSOLUTE.

The fourth camera (`cam_right_high`) is left out: the BCT view-count ablation
measured the second head camera at 0.00% change in arm accuracy for a 33% higher
decode cost. `cam_left_high` is the verified left eye of the head stereo pair
(feature disparity is consistently negative against `cam_right_high`).
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

g1_dex1_ikea_relarm_3view_aug_config = {
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
    g1_dex1_ikea_relarm_3view_aug_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
