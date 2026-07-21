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

"""Nubzuki pick-and-place joint-space config: upper-body ABSOLUTE joint actions.

Dataset: RooibosT/g1-nubzuki-pickandplace-260715 (43 dims = 12 legs + 3 waist +
7+7 arms + 7+7 dex3 hands), sonic teleop @ 50 fps, single head camera (ego_view).

Action = waist + arms + hands only. Legs stay state-only: they are sonic WBC
balance outputs, nearly static in this task (per-joint std < 0.13 rad), and at
deployment the lower body runs its own balance controller. The model still
conditions on leg state. ABSOLUTE arms follow the validated B-variant recipe
(B >> A on the redblock closed-loop eval). ABSOLUTE dex3 hands: teleop sent
binary open/close snapping to fixed angles (stats confirm bimodal 0 <-> +-1.57/
+-1.75), matching the built-in unitree_g1 configs' "hand as gripper" choice.
Horizon 40 = pretrained native, and the same choice the built-in
unitree_g1_sonic posttrain config makes at this dataset's 50 fps (0.8 s
lookahead). Deployment client must match --actions_per_chunk=40.
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

g1_dex3_nubzuki_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "legs",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        action_configs=[ABS_JOINT] * 5,
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_dex3_nubzuki_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
