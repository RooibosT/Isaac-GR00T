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

"""Sim (lightwheel) side of the IKEA co-training mixture, on its own embodiment tag.

Registered against ``NEW_EMBODIMENT_2`` so it gets projector slot 12 while the
real recordings keep slot 10. That separation is the whole point: the two
sources do not agree on what a state or action vector means, and giving them one
projector would ask a single linear map to read 33 raw joint angles and 46
structured real-robot features as the same thing.

What the two sides therefore share is only the trunk -- the DiT and the frozen
backbone. That is also the only place sharing could plausibly help here: the
hypothesis being tested is that "approach a target smoothly and stop at contact"
is a domain-general skill, while the pixels are not (measured frozen-encoder
distance IKEA<->SIM is 5.2x each set's internal spread on the ego view, 3.2x on
the wrist views, against 1.3-1.9x for the two real datasets).

Choices that are not obvious:

* **The base blocks are dropped from the action.** `base_velocity`,
  `base_height` and `torso_orientation` are in the source action, but the clips
  were selected for having no base motion, so those three are constant by
  construction: base_height is 0.7400 everywhere and torso pitch is 0. Training
  on constants teaches nothing and costs action-head width.
* **Everything is ABSOLUTE, not RELATIVE.** The real config makes the arms
  relative by subtracting the matching state block. There is no end-effector
  block in this state (it is raw joint angles only), so there is nothing to
  subtract; the poses stay absolute.
* **The quaternions ride along as plain numbers** (`ActionFormat.DEFAULT`, not
  one of the rotation formats, which expect xyz+rot6d / xyz+rotvec rather than
  the w-first quaternion this dataset stores). Verified safe on the cut clips:
  every quaternion is unit norm with w > 0, so the q/-q double cover never
  shows up and plain normalisation is well behaved.
* **State is the single 33-dim `joint_position` block.** The source has no IMU
  quaternion, so there is no `base_gravity` to build, and no forward-kinematics
  end-effector block either. Synthesising them would put a constant in a slot
  that varies on the real robot.
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


ABS = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

lw_sim_ee_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "ego_view",
            "left_wrist_view",
            "right_wrist_view",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_position"],
    ),
    "action": ModalityConfig(
        # 40 frames is 2.0 s here against 1.33 s on the 30 fps real robot. Kept
        # equal in *frames* so both sides fill the same action-head width.
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "left_hand",
            "right_hand",
            "left_ee",
            "right_ee",
        ],
        action_configs=[ABS, ABS, ABS, ABS],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(lw_sim_ee_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT_2)
