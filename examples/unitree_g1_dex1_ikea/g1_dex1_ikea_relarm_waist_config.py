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

"""`g1_dex1_ikea_absarm_waist_config` with RELATIVE arms -- the REL half of the waist pair.

For `RooibosT/nature_pouch_new`. Everything is `g1_dex1_ikea_absarm_waist_config` (46-dim
state, 3 views, H40, 17-dim action with `waist_yaw` last) except the representation of the two
arm blocks: 7+7 RELATIVE, as in `g1_dex1_ikea_relarm_3view_aug_config`. The model predicts each
arm target as an offset from the arm state at the last observed frame, and the policy adds that
state back before it returns the chunk, so the client still receives absolute joint targets
in the same 17-dim layout.

Grippers and `waist_yaw` stay ABSOLUTE, so the arm representation is the only thing that
differs from the ABS run:

- grippers: every REL run here has kept them ABSOLUTE; they are open/close targets.
- `waist_yaw`: RELATIVE needs a state block of the same name and width, and the state's
  `waist` is 3-wide (yaw, roll, pitch) while the action carries yaw alone. Making it RELATIVE
  would mean a 1-wide `waist_yaw` state key, which changes the 46-dim state.

RELATIVE needs `meta/relative_stats.json` entries for `left_arm` / `right_arm`;
`chain_nature_pouch_new_relarm_waist.sh` generates them with this config before training.

**Deployment** is the same as the ABS waist model: unpack 17 dims, and send the last as the
waist yaw target on `rt/arm_sdk` slot 12.
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

g1_dex1_ikea_relarm_waist_config = {
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
            "waist_yaw",
        ],
        action_configs=[
            REL_JOINT,  # left_arm
            REL_JOINT,  # right_arm
            ABS_JOINT,  # left_gripper
            ABS_JOINT,  # right_gripper
            ABS_JOINT,  # waist_yaw -- the [w] turn request, rt/arm_sdk slot 12
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    g1_dex1_ikea_relarm_waist_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
