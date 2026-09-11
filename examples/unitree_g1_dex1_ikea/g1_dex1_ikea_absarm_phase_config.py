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

"""IKEA stage1_v2: the 46-dim ABSOLUTE config plus a subtask phase bit (47).

**Why this one is not the previous four.** Torque, observation history and a
gripper-derived phase all failed, and all failed the same way: what they added
was already in the observation. Arm torque is 65-68% linearly recoverable from
the 60-dim state; a history is derived from the same stream, and the current
frame already predicts the next action at R2 0.997; a phase read off the gripper
is the policy's own action, so conditioning on it is circular.

The turn count is none of those. The leg is turned four times before the base is
aligned, and **after two turns and after four the robot's state is the same** --
same joint angles, same forces, same scene but for thread depth. No amount of
data, capacity, history or torque recovers it. It is textbook partial
observability, and on the robot it shows up exactly as you would predict: the
policy keeps turning and never moves on.

An external module counts the turns. `RooibosT/IKEA-pick-leg-stage1_v2_subtask`
carries its labels, and `convert_subtask_v3_to_v2.py` puts them in the state as
column 117 rather than splitting the instruction -- splitting makes it two
policies sharing weights, which is what an earlier attempt did and what degraded
at the transition.

**Raw 0/1 is the right encoding, and no scaling is wanted.** State is min/max
normalised against q01/q99 and clipped to [-1, 1]. The align phase is 8.3% of
frames, so q01 is 0 and q99 is 1, and the two values land on exactly -1 and +1:
the widest range this pipeline gives anything, with both phases an active signal
rather than one of them being the absence of one.

**The phase must be exempt from state dropout**, which is why the run passes
`--state-dropout-keep-keys phase`. Dropping the one input that disambiguates two
behaviours, on 20% of samples, while still demanding the right action, teaches
the model to hedge on precisely the signal it is being given. See that flag's
note for what it changes about the dropout schedule.

**The open-loop scan cannot see whether this worked.** Fed demonstration
observations, a model that would turn forever on the robot still scores well,
and the transition is one moment per episode inside a phase worth 8.3% of
frames. Judge it on the robot, or on a probe that scores the boundary.
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

g1_dex1_ikea_absarm_phase_config = {
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
            "phase",
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
    g1_dex1_ikea_absarm_phase_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
