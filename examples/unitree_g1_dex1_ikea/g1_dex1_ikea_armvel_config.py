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

"""IKEA ablation: the base 46-dim config plus arm joint velocities (state -> 60).

Everything else is identical to g1_dex1_ikea_relarm_3view_aug_config, so any
difference in the open-loop scan is attributable to these 14 dims alone.

Unlike torso_gravity — which was 99.85% recoverable from state already present —
velocity is genuinely new: the model sees a single frame (delta_indices=[0]), and
you cannot get a rate from one position sample. The recorded dq is close to but
not identical to the position difference (corr 0.967, residual 14% of signal std).

The reason to be careful rather than merely hopeful: measured on the val split,
plain constant-velocity extrapolation predicts the first 8 steps to 1.49 deg,
against 2.23 deg for holding the current pose and 1.31 deg for the trained model.
So velocity alone buys ~88% of the model's accuracy over exactly the window
deployment executes, which means giving it to the model also removes much of the
gradient pressure on the visual pathway. Extrapolation collapses later in the
chunk (12.19 deg at step 40, worse than holding still at 8.52), so a whole-chunk
shortcut is not available — but a partial one is.

Judge this run on the late chunk steps and on the gripper, not on the first-8
arm error. Gripper timing cannot be extrapolated from joint velocity, so if it
degrades, the visual pathway got weaker. Gripper velocities are deliberately
left out to keep that signal clean; legs/waist velocities are near-zero on a
stationary robot and base velocities have std 0.006-0.015, which normalization
would only amplify into noise.
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
