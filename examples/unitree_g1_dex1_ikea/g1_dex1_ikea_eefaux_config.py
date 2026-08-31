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

"""RAMEN 레시피 완전판: armvel state + **EEF를 action에 추가** (action 16 -> 34).

GPU 0,1의 `g1_dex1_ikea_ramen_config.py` 런과 **EEF 하나만 다르다** (배치 16 / 200k /
회전 5도 / REAL_G1 slot 25는 동일). 그래서 두 런의 차이가 EEF 중복 지도의 값을 그대로
가격한다. 동시에 이쪽이 RAMEN 레시피의 완전판이다.

RAMEN이 하는 것을 그대로 옮긴 것이다: 손목 EEF 9D를 팔 관절과 **함께** 예측한다
(`RAMEN_COMPARISON.md` §2). joint를 EEF로 대체하는 게 아니다 — EEF = FK(joint)라 정보로는
완전히 중복이므로, 이건 **중복 지도(auxiliary supervision)** 다. action expert가 task-space
모션을 명시적으로 표현하도록 손실을 건다.

우리 §16의 "FK EE를 state로 넣어도 팔은 동률"은 EEF를 **입력**으로 준 실험이다. 정보를 더
주는 것과, 그 표현을 만들도록 손실을 거는 것은 기전이 다르다. 이 config가 재는 것은 후자다.

    action 16차원 (armvel)          ->  34차원 (여기)
      left_arm      7  REL              left_arm       7  REL
      right_arm     7  REL              right_arm      7  REL
      left_gripper  1  ABS              left_gripper   1  ABS
      right_gripper 1  ABS              right_gripper  1  ABS
                                        left_wrist_eef_9d    9  REL  EEF  XYZ_ROT6D   <- 추가
                                        right_wrist_eef_9d   9  REL  EEF  XYZ_ROT6D   <- 추가

state에도 `*_eef_9d`가 들어간다. 기존 6D eef(6+6)를 9D(9+9)로 갈아끼우므로 60 -> 66이다.
선택이 아니라 필수다: `ActionType.EEF` +
`RELATIVE`는 델타를 현재 EEF 프레임에서 잡고, 그 기준 프레임을
`EndEffectorPose.from_action_format(reference_state, action_format)`으로 만든다.
`_convert_to_absolute_action`이 `reference_state.shape[0] == action.shape[1]`을 assert하므로
**기준 state가 action과 같은 9D여야 한다.** 기존 `left_eef`(6D euler)로는 안 된다.
`state_key`를 명시해 어느 state 블록이 기준인지 못 박아 둔다.

블록 이름이 NVIDIA의 것과 같은 것은 우연이 아니다 — REAL_G1 태그에서는 프로세서 생성 시점에
베이스 체크포인트의 real_g1 relative_action 키 목록으로 검증을 받으므로, 그 목록에 있는
이름이어야 학습이 시작된다. `make_eef_action_variant.py` 주석 참고.

기존 6D `left_eef`/`right_eef`는 state에서 **뺐다.** 같은 양을 두 표현으로 넣으면 60 -> 78이
아니라 84가 되고, 중복 채널이 늘어난 것이 EEF 지도의 효과와 섞인다. 9D 쪽이 rot6d라
불연속이 없으므로 6D를 대체하는 편이 낫다.

데이터셋: `G1_Dex1_IKEA_leg_30hz_eef` (`make_eef_action_variant.py` 산출).
state 117 -> 135, action 33 -> 51. 통계 재생성 필요.
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


def _rel_eef(state_key: str) -> ActionConfig:
    return ActionConfig(
        rep=ActionRepresentation.RELATIVE,
        type=ActionType.EEF,
        format=ActionFormat.XYZ_ROT6D,
        state_key=state_key,
    )


g1_dex1_ikea_eefaux_config = {
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
            "base_gravity",
            "left_wrist_eef_9d",
            "right_wrist_eef_9d",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_gripper",
            "right_gripper",
            "left_wrist_eef_9d",
            "right_wrist_eef_9d",
        ],
        action_configs=[
            REL_JOINT,  # left_arm
            REL_JOINT,  # right_arm
            ABS_JOINT,  # left_gripper
            ABS_JOINT,  # right_gripper
            _rel_eef("left_wrist_eef_9d"),
            _rel_eef("right_wrist_eef_9d"),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# REAL_G1 = projector slot 25. GPU 0,1의 ramen 런과 맞춰야 EEF 하나만 다른 쌍이 된다.
register_modality_config(g1_dex1_ikea_eefaux_config, embodiment_tag=EmbodimentTag.REAL_G1)
