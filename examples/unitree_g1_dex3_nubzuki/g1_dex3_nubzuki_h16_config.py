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

"""Nubzuki pick-and-place — ABSOLUTE arms, horizon 16 variant.

Identical to g1_dex3_nubzuki_config.py (all-ABSOLUTE joint actions) except the
action chunk is 16 instead of 40 (0.32 s vs 0.8 s lookahead @ 50 fps). Shorter
chunks replan more often (tighter reaction to visual feedback, less open-loop
drift) at the cost of a smaller latency buffer at deploy. Paired with the h16
REL variant to sweep horizon x representation on the spare GPUs.

Deployment client: --actions_per_chunk=16 (must match).
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

g1_dex3_nubzuki_h16_config = {
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
        delta_indices=list(range(0, 16)),
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

register_modality_config(g1_dex3_nubzuki_h16_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
