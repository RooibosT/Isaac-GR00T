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

"""IKEA: ABSOLUTE arms *with* arm joint velocities in the state (60-dim).

The reason this exists is a deployment result, not a table. Section 27 measured
three corners of a 2x2 on the stage-1 data -- same val, same 35,000 steps, same
effective batch 64 -- and read them as settling the action representation
(arm8 deg / EE8 mm at checkpoint-35000):

                     no armvel (46)      armvel (60)
      REL arms       1.377 / 10.96       1.153 /  9.74
      ABS arms       2.261 / 15.48       this run

On that evidence ABS is the worst corner by a wide margin: +64% on arm8, +41%
on EE8, and the gap concentrates at the front of the chunk, which is the part
deployment actually executes. Section 27 drew the obvious conclusion from that
and said so explicitly -- "the first ~8 steps are what reaches the robot, so
this is the difference that reaches the robot."

**On the robot it came out the other way.** Running both on the real G1, the
ABS policy's task success rate looks better than the relative one's. That is an
operator observation over a handful of attempts rather than a scored trial set,
so it is not yet a number -- but it points the opposite direction from a 64%
open-loop gap, and a difference that large should not be invisible to, let
alone reversed by, closed-loop execution.

So the open-loop scan is not measuring what we assumed it measures. Two ways
that happens, and they are not exclusive:

  * **Relative targets accumulate.** Open loop, each chunk is scored against
    ground truth from the same anchor the model saw. Closed loop, the anchor is
    the arm's own drifted position, so a relative policy integrates its own
    error while an absolute one is re-anchored to the commanded pose every
    chunk. Low per-chunk MAE and stable long-horizon behaviour are different
    properties, and only the second one assembles a table.
  * **MAE is not the failure mode.** Success is gated on grasp and insertion
    tolerances, not on mean joint error. A policy can be worse on average and
    still cross the thresholds that matter more often.

This run therefore is not an ablation looking for the best corner of the 2x2.
It adds arm velocity to the representation that deploys better, and the useful
question is whether the -16.3% arm8 / -11.1% EE8 that velocity bought under
RELATIVE targets (section 27, reproducing section 16 on different data) also
appears under ABSOLUTE ones. Velocity is first-order information and the
relative/absolute choice is a zeroth-order anchoring choice, so the two should
be close to independent and the gain should carry over roughly intact.

Two things to watch, both of which would mean the transfer is not clean:

  * If ABS+armvel *closes* most of the open-loop gap to REL, then what the
    relative parameterization was contributing is rate information it leaks
    implicitly, and section 16's and section 27's readings both need revising.
  * If the gripper degrades, the visual pathway got weaker -- gripper timing
    cannot be extrapolated from joint velocity, so that is the clean signal.
    Section 16 is the reason to judge an armvel run on the late chunk steps and
    the gripper rather than on first-8 arm error: constant-velocity
    extrapolation alone already predicts the first 8 steps to 1.49 deg on this
    family of data, so a first-8 improvement partly measures the shortcut.

Whatever this run scores, the open-loop scan has lost its standing as the
deployment proxy until the ABS-vs-REL success rates are measured properly. That
measurement is the blocking work, not another config.

Mechanically this file is `g1_dex1_ikea_armvel_config` with the two arm action
blocks switched from RELATIVE to ABSOLUTE, so the diff against it and against
`g1_dex1_ikea_absarm_3view_aug_config` is one thing in each direction.

Grippers are ABSOLUTE in every config in this family, so only the arms differ.

    Inference note: like every armvel model, this one must be fed real `arm_dq`.
    Feeding zeros is worse than the 46-dim baseline over the execution window
    (section 16), and `url_groot_deploy/observation.py` still has no 60-dim
    ObservationSpec.
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

g1_dex1_ikea_absarm_armvel_config = {
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
    g1_dex1_ikea_absarm_armvel_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
