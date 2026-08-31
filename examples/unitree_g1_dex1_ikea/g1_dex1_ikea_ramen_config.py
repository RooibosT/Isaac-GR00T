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

"""RAMEN 레시피 실험용: armvel 60차원 config를 REAL_G1 태그로 등록한 것.

`g1_dex1_ikea_armvel_config.py`와 **모달리티 정의가 한 글자도 다르지 않다.**
바뀌는 것은 등록 태그뿐이고, 그 결과 projector 슬롯이 달라진다:

    NEW_EMBODIMENT -> slot 10   (베이스 체크포인트에서 아무도 안 쓰는 슬롯 = 초기화 상태)
    REAL_G1        -> slot 25   (N1.7 사전학습에서 실제로 학습된 슬롯)

`_PROJECTOR_INDEX_GROUPS`(gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py)가 그 매핑을
들고 있다. 베이스 가중치에서 `action_head.state_encoder.layer1.W`를 슬롯별로 재보면
차이가 눈으로 보인다 — 미사용 슬롯은 전부 L2 7.33~7.38(초기화 baseline)에 몰려 있고
slot 25는 8.31, slot 26(r1_pro)은 8.66이다.

Team-RAMEN이 이 슬롯을 쓴다(`RAMEN_COMPARISON.md` §2). 우리가 그대로 가져올 수 없는
이유도 거기 적어 뒀다: slot 25 가중치는 NVIDIA의 49차원 컬럼 의미(EEF 9D가 [0:9] 등)에
맞춰진 것이고 우리 60차원은 legs가 [0:12]로 시작한다. **컬럼 대응이 맞지 않으므로 이건
"맞는 사전지식"이 아니라 "실제 로봇 데이터를 본 적 있는, 잘 조건화된 행렬"일 뿐이다.**
중립이거나 오히려 해로울 수 있다 — 틀린 초기값이 랜덤보다 나쁜 경우가 있다.

그래서 이 파일은 가설이 아니라 측정 도구다. 값이 있으면 §2의 "레이아웃까지 통째로
채택"을 검토할 근거가 되고, 없으면 그 선택지를 닫는다.

주의: `register_modality_config`는 태그당 1회만 허용한다(assert). `real_g1_relative_eef_
relative_joints`는 `MODALITY_CONFIGS` 사전 등록 8개에 없으므로 충돌하지 않는다. 다만 이
파일과 `g1_dex1_ikea_armvel_config.py`를 **같은 프로세스에서 동시에 import 하지 말 것** —
서로 다른 태그라 assert는 통과하지만, 어느 쪽이 쓰이는지 헷갈릴 이유가 없다.

`meta/stats.json`은 embodiment 태그와 무관하다 (feature 이름으로만 fingerprint를 잡는다,
gr00t/data/stats.py). 태그를 바꿔도 재생성할 필요가 없다.
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

g1_dex1_ikea_ramen_config = {
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

register_modality_config(g1_dex1_ikea_ramen_config, embodiment_tag=EmbodimentTag.REAL_G1)
