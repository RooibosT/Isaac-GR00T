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

"""IKEA stage1_v2: ABSOLUTE arms, arm velocity, and arm joint torque (state -> 74).

`g1_dex1_ikea_absarm_armvel_config` plus `left_arm_torque` and
`right_arm_torque`, so a run against it isolates those 14 dims. Identical in
every other respect -- same three views, same 16-dim ABSOLUTE action layout,
same horizon 40.

**Why this exists is a failure mode, not a table.** On the robot the policy
inserts the leg and then does not go on to rotate it tight. Sections 21 and 23
already scored torque as a general state addition and rejected it, but they
scored it on whole-episode mean MAE, and this is not a whole-episode phenomenon
-- `probe_insert_phase.py` says so in its own docstring: the full-episode scan is
dominated by transit, and a change that helps only at the contact is invisible
in the aggregate. What that rejection actually established is narrower than it
reads, and it does not cover the question being asked here.

Torque on a position-controlled arm has two components and they behave
differently at deployment:

* In free space it is the servo error, which is close to the *next commanded
  position*. Open loop feeds the demonstrator's logged torque, so this hands the
  model part of its own label -- a linear fit of the relative left-arm target
  reads R2 0.59 from torque at horizon step 1 against 0.40 from velocity. It is
  also the component that **disappears** on the robot, where the torque coming
  back was produced by the model's own previous chunk. This is the leak sections
  21 and 23 caught.
* In contact it is the reaction force, and that one **survives** closed loop:
  command the hand down, the seated leg blocks it, the torque rises whether the
  command came from a demonstrator or from the policy.

The bet here is on the second component. Seating a leg is a force event at a
millimetre scale under the hand, where the wrist cameras see very little, and
`probe_micro_adjust.py` measured that the model does not currently emit the
settling correction at all -- at 85 held-out reversals the demonstrator's wrist
rises a median 1.99 mm over the 40-step chunk while the model predicts -0.09 mm,
inside a 4.40 mm MAE. There is no channel for feeling the leg bottom out, which
is consistent with never producing the behaviour that follows.

**How to read the scan, because the leak is real.** The first-8 arm error, which
selects checkpoints for every other run here, is exactly where the leak lives, so
it is not the metric for this one. Section 21's rule stands:

    horizon step        1      4      8     16     28     40
    from arm_vel      0.40   0.62   0.60   0.41   0.23   0.14
    from arm_torque   0.59   0.34   0.20   0.11   0.07   0.07

A genuine gain is flat across the horizon; a leak decays along that curve --
section 23 measured EE ratio 0.634 at step 1 rising to 0.980 at step 40 and read
it, correctly, as a leak. Judge this run on:

1. `probe_insert_phase.py` approach/contact against the 60-dim run,
2. the gripper, which is not linearly recoverable from arm servo error and so is
   the control channel, and
3. the late chunk (steps 24-40), where the leak has decayed to R2 0.07.

Gripper torque stays out for the reason section 21 gives -- it is the control
channel, and putting a contact signal into it would spoil the one measurement
that can tell skill from shortcut. Waist torque (89% recoverable from the state
already present) and legs torque (standing load on a robot that does not walk)
stay out too.

The dataset needs no rebuild: `IKEA_pick_leg_stage1_v2` already carries all 117
state dims and `meta/stats.json` covers every one of them.
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

g1_dex1_ikea_absarm_armvel_torque_config = {
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
            "left_arm_torque",
            "right_arm_torque",
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
            ABS_JOINT,  # left_arm
            ABS_JOINT,  # right_arm
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
    g1_dex1_ikea_absarm_armvel_torque_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
