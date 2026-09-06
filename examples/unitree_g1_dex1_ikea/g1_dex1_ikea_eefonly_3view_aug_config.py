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

"""IKEA stage-1 config that outputs END-EFFECTOR actions instead of joint ones.

The arms are commanded as a wrist pose (xyz + rot6d, 9 dims each) relative to the
current wrist pose, not as joint targets. Grippers stay absolute. Action is
therefore 20 dims: 9+9 EEF, 1+1 gripper. No arm velocity in the state.

This is not the `eefaux` config. That one predicts the EEF *alongside* the joints
as redundant supervision and still commands joints; here the EEF is the command,
so nothing in the action tells the robot what the joints should be.

Why relative rather than absolute. The composition is a proper SE(3) one --
`pose.relative_transformation` builds T_ref^-1 @ T_t from homogeneous matrices via
scipy Rotation, not a subtraction of Euler angles -- so relative EEF carries none
of the wraparound pathology that makes naive implementations of it bad. And the
measured prior points the same way twice: BCT section 2 put relative joints ahead
of absolute by 33% on first-5, and section 27 of this file measured the same
comparison on this very dataset at 77%. The mechanism is that the target is
anchored to something the model already observes, which is representation
independent. In EEF it should if anything be stronger: the pose lives in a
torso-relative FK frame with the waist zeroed, so an absolute target asks the
network to place the wrist in a frame it cannot see, while a relative one asks
only for the displacement the scene actually shows.

## What the state has to carry

`_convert_to_absolute_action` asserts `reference_state.shape[0] == action.shape[1]`,
so the reference block must be the same 9D xyz+rot6d as the action. The dataset's
own `left_eef` is 6D (xyz + extrinsic-xyz euler) and cannot serve. Run
`make_eef_action_variant.py` first: it adds `{left,right}_wrist_eef_9d` to both
state and action, converting the stored euler exactly (rot6d round trip 1.4e-15)
rather than re-running FK, and solving FK only for the action side, which has no
eef block of its own.

State is the 46-dim set with the two 6D eef blocks swapped for their 9D forms:
52 dims over nine keys.

## Two things this costs

Deployment needs IK. A joint policy commands the motors directly; this one hands
back a wrist pose that something has to solve, which adds a failure mode the
joint runs do not have (section 16). The IROS `decoupled` lane wants EE output,
which is the reason to care.

`scan_ikea.py` scores the arm in joint space and derives the wrist by FK from the
predicted joints. With no joint block in the action it reports the wrist error
directly from the predicted pose instead -- the same quantity in the same frame,
so `ee_mm` stays comparable with every earlier run -- and no `arm_deg`, which has
no meaning here.
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


def REL_EEF(state_key: str) -> ActionConfig:
    """Wrist pose relative to the current one, xyz+rot6d, anchored on `state_key`."""
    return ActionConfig(
        rep=ActionRepresentation.RELATIVE,
        type=ActionType.EEF,
        format=ActionFormat.XYZ_ROT6D,
        state_key=state_key,
    )


g1_dex1_ikea_eefonly_3view_aug_config = {
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
            "left_wrist_eef_9d",
            "right_wrist_eef_9d",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "left_wrist_eef_9d",
            "right_wrist_eef_9d",
            "left_gripper",
            "right_gripper",
        ],
        action_configs=[
            REL_EEF("left_wrist_eef_9d"),
            REL_EEF("right_wrist_eef_9d"),
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
    g1_dex1_ikea_eefonly_3view_aug_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
