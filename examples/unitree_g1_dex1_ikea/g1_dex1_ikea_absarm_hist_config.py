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

"""IKEA stage1_v2: the 74-dim ABS state, observed over a one-second history.

`g1_dex1_ikea_absarm_armvel_torque_config` with the state read at four instants
instead of one -- t-30, t-15, t-5, t at 30 Hz. Everything else is identical, so
a run against it isolates the history.

**Why.** On the robot the policy inserts the leg and does not go on to rotating
it tight. Every observation this model has ever had is a single instant
(`state_history_length` 1, video `delta_indices` [0]), and the evidence that
seating happened is a *transient*: torque rises and then plateaus, the wrist
stops descending. From one frame "pressing in" and "already seated" look alike.
The 74-dim torque run could not use torque for this reason -- it was given the
magnitude and never the trend.

**Which instants.** Span matters more than resolution here, because `arm_vel` is
already in the state and already is the local derivative. What one frame cannot
supply is seconds-scale context: how long the arm has been loaded. Hence roughly
geometric spacing over a full second rather than a tight uniform stride -- t-5
keeps a near sample for control, t-30 reaches back past the whole contact event.

**Which blocks the history carries** is set on the launch line, not here, since
it is a regularisation choice rather than a data one. Measured on this dataset
(ridge dR2 over the current 74-dim state, insertion-phase frames, at horizon 40):

    both arm torque   +0.0116      both eef xyz     +0.0062
    both eef full     +0.0099      both eef rot     +0.0041
    torque+eef+grip   +0.0210      left eef z only  +0.0009
    all 74 dims       +0.0363

so `left/right_arm_torque`, `left/right_eef`, `left/right_gripper` -- 28 of the
74 dims -- carry 58% of what the dense history offers in that phase, and the 42%
given up is arm joint and velocity history, which is exactly the channel a
policy uses to extrapolate its own motion instead of reading the scene.

**How to read the scan.** History is the classic causal-confusion input, so the
same caution as sections 21 and 23 applies, but the leak signature is different
and worth stating because it is *not* the torque one. The current frame already
predicts the next action at R2 0.997, so there is nothing left to leak at the
front of the chunk; the measured gain is at the back:

    horizon step         1        8       20       40
    dR2 from history  +0.0008  +0.0032  +0.0152  +0.0375

That is the opposite shape from torque (R2 0.59 at step 1 decaying to 0.07). It
rules out the pathology section 21 detected, not causal confusion in general --
a linear fit cannot tell "knows the phase" from "extrapolates the motion", and
both grow with horizon. Judge on the gripper, the late chunk, and
`probe_insert_phase.py`, the same as the torque run.
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

g1_dex1_ikea_absarm_hist_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_left_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ],
    ),
    "state": ModalityConfig(
        # One second, ending at the current frame. The launcher reads the length
        # from here and turns on index padding, so the opening frames of an
        # episode repeat frame 0 rather than wrapping to the end of it.
        delta_indices=[-30, -15, -5, 0],
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
    g1_dex1_ikea_absarm_hist_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
