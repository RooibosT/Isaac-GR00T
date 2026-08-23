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

"""IKEA ablation P1: the base 46-dim config plus `torso_gravity` (state -> 49).

Everything else is identical to g1_dex1_ikea_relarm_3view_aug_config, so any
difference in the open-loop scan is attributable to these three dims alone.

`torso_gravity` is projected gravity from `rt/secondary_imu`, mounted in the
torso — i.e. the head cameras' own tilt, and a signal BCT never had. It is worth
testing here because that tilt is not fixed: per-episode mean pitch ranges over
4.4-17.6 deg (std 3.75) as the rig was re-set between sessions, with a visible
regime change around episode 85 (~7 deg -> ~16 deg). The model reads the ego
view from that camera without otherwise being told how it is oriented.

Note this is torso, not pelvis: `base_gravity` (already in the state) comes from
the pelvis IMU, and the two differ by the waist joints. Never recover the waist
by differencing the two IMUs' yaw — they integrate yaw independently and have
been observed ~2 rad apart; the waist is `waist` in the state.

Deployment cost: one extra DDS subscription to `rt/secondary_imu`, which the
teleop stack already reads.
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

g1_dex1_ikea_relarm_3view_torsograv_config = {
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
            "torso_gravity",
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
    g1_dex1_ikea_relarm_3view_torsograv_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
