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

"""The 46-dim ABSOLUTE config plus a goal-pose auxiliary block on the action.

`make_goalpose_variant.py` appends `right_goal_9d` to the action: for every frame,
the right wrist pose at the next right-gripper transition -- the pose the leg goes
into the hole at, while inserting. The table moves between recordings and the
state says nothing about where it is, so this target can only be reached through
the vision path, unlike the torque, history and gripper-phase inputs sections 31
and 34 rejected.

The block is predicted, never commanded. The deploy client reads its own action
keys, so the extra nine columns are ignored there and the observation contract
does not move; on this side they only add to the flow-matching loss.

Against `g1_dex1_ikea_absarm_3view_aug_config` on the same split and schedule,
the pair isolates the auxiliary loss alone.

Below is the baseline's own description, unchanged.

IKEA ablation: the 46-dim config with ABSOLUTE arms instead of relative.

Identical to `g1_dex1_ikea_absarm_3view_aug_config` in every other respect --
same 46-dim state, same three views, same 16-dim action layout, same horizon --
so a run against it isolates the action representation alone.

Relative is the adopted setting and this is expected to lose. The BCT round-2
A/B (`examples/unitree_g1_dex1_bct/EXPERIMENTS.md` section 2) measured RELATIVE
joint targets beating ABSOLUTE by 33% on first-5 MAE and 11% on the arm as a
whole, and the reasoning is structural: a relative target is anchored to the
state the model already observes, so the network spends its capacity on the
delta rather than re-deriving the absolute pose. What this run buys is that
number on *this* dataset -- long single-instruction stage-1 episodes rather than
the short per-subtask cuts the original comparison used, where the arm travels
much further within one episode and the anchor is worth correspondingly more.

Grippers were already ABSOLUTE in the relative config, so only the two arm
blocks change here.
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

g1_dex1_ikea_absarm_goal_config = {
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
            "right_goal_9d",
        ],
        action_configs=[
            ABS_JOINT,  # left_arm
            ABS_JOINT,  # right_arm
            ABS_JOINT,  # left_gripper
            ABS_JOINT,  # right_gripper
            # The goal block is a plain absolute vector, not an EEF action: it is
            # supervision, never commanded, and ActionType.EEF would drag in the
            # pose composition and its matching state-block requirement.
            ABS_JOINT,  # right_goal_9d
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    g1_dex1_ikea_absarm_goal_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
