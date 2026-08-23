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

"""IKEA config with the waist kept in the action, matching the BCT slot layout.

Identical to g1_dex1_ikea_relarm_3view_aug_config except the action carries a
leading waist(3), giving

    [ waist(3) | left_arm(7) | right_arm(7) | left_gripper(1) | right_gripper(1) ]

which is exactly what the BCT fine-tune emits. Warm-starting from that model only
transfers its action head if the output slots mean the same thing; without the
waist every arm slot shifts by three and BCT's waist weights land on the left
shoulder.

The waist is not commanded by teleop_ikea — it holds the startup pose — so this
block is the measured waist and the model learns to echo it. That costs three
near-free output dims and is ignored at deployment, the same as for BCT.

Use with the *_wa_train / *_wa_val datasets built by make_waist_action_variant.py.
Also serves as the control run from the stock GR00T base, so that "BCT
warm start vs base" and "16-dim vs 19-dim action" stay separable.
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

g1_dex1_ikea_waistact_config = {
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
            "waist",
            "left_arm",
            "right_arm",
            "left_gripper",
            "right_gripper",
        ],
        action_configs=[
            ABS_JOINT,  # waist — echoed, ignored at deployment
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

register_modality_config(g1_dex1_ikea_waistact_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
