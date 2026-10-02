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

"""IKEA multi-task: 68-dim state (every recorded velocity) with the 19-dim action.

The two halves already exist separately:

* `g1_dex1_ikea_absarm_armvel_config` adds `left_arm_vel` / `right_arm_vel` (7+7) to the
  46-dim state. The gain is the most reproduced result in this project -- arm8 -16.3% on
  RELATIVE targets (section 27), -15.1% on the subtask split (32), -18.0% on the rotate
  set (24), -8.1% on ABSOLUTE targets (28/32), and again on fliptable v2 (37).
* `g1_dex1_ikea_absarm_basevel_config` adds `base_cmd_vel` (3) to the 16-dim action, for
  `RooibosT/stage3_rc`, the one recording where the robot drives.

This config is both, for the three-set merge. Nothing else changes: same three views, same
horizon 40, same ABSOLUTE arms and grippers.

Beyond the 60-dim armvel set this adds the two blocks that were recorded and never used:

* **`left_gripper_vel` / `right_gripper_vel` (1+1).** The same relation to the gripper
  position that `*_arm_vel` has to the arm: correlation with the 30 Hz finite difference is
  0.946 for both. Arm velocity is the most reproduced gain in this project and the gripper
  never got the equivalent input, so the policy sees where each gripper is but not whether
  it is opening or closing -- and "lets go without seating the leg" (section 34) is the
  failure this could bear on. Magnitudes are large, not marginal: mean |v| 0.48-1.02 with
  per-dim sd 1.5-2.0, against 0.12-0.22 for the arm velocities.
* **`base_lin_vel` / `base_ang_vel` (3+3).** Dead weight until now -- every earlier IKEA set
  was stationary work -- but `stage3_rc` drives, and this action space *commands* the base.
  Measured base velocity is the only feedback the policy could have for its own
  `base_cmd_vel`; without it the model commands the drive open-loop within the chunk. In
  the merge these are small but alive (per-dim sd up to 0.058 linear, 0.084 angular, against
  0.004-0.017 in the stationary sets).

Left out on purpose: `legs_vel` and `waist_vel` (small in the assembly, 0.72-0.75 correlated
with their own position differences), the torque blocks (excluded by operator decision, and
section 23/31 measured torque as a shortcut signal), and `torso_gravity` (section 5 tested it
and it lost). Every one of these q01/q99 bands is non-degenerate in the merged split, checked
before launch, so none of the new dims normalises to a constant.

⚠️ This run changes three blocks at once against the 46-dim baseline. Arm velocity is
established (-8% on ABSOLUTE targets); the gripper and base velocities are not. A win here
says the 68-dim state is better, not which block did it.

⚠️ **Inference must feed real `arm_dq`.** Zeroing the velocity block at deploy time makes a
60-dim model worse than the 46-dim baseline it was supposed to beat (section 16); the
deploy-side spec for these inputs is `IKEA_ARMVEL_SPEC`. Together with the 19-dim action
unpacking that `base_cmd_vel` forces, a client built for the 46/16 models needs both
changes before it can run this one.

⚠️ The velocity gain is smaller under ABSOLUTE targets than under relative ones -- about
half, by the measurement in section 32 -- because velocity is information about the delta
and only the relative parameterisation predicts the delta directly. It has never been
measured on top of a multi-task merge, which is what the run against this config is for.
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

g1_dex1_ikea_absarm_allvel_basevel_config = {
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
            "left_gripper_vel",
            "right_gripper_vel",
            "base_lin_vel",
            "base_ang_vel",
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
            "base_cmd_vel",
        ],
        action_configs=[
            ABS_JOINT,  # left_arm
            ABS_JOINT,  # right_arm
            ABS_JOINT,  # left_gripper
            ABS_JOINT,  # right_gripper
            ABS_JOINT,  # base_cmd_vel -- absolute velocity command, no state anchor
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    g1_dex1_ikea_absarm_allvel_basevel_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
