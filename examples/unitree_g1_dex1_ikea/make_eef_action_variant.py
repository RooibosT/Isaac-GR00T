# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add 9D end-effector blocks to both state and action, for the EEF-auxiliary run.

RAMEN이 `left_wrist_eef_9d`/`right_wrist_eef_9d`를 **팔 관절과 함께** 예측한다
(`RAMEN_COMPARISON.md` §2). joint를 EEF로 *대체*하는 게 아니라 둘 다 지도한다 —
EEF = FK(joint)라 정보로는 완전히 중복이고, 그래서 이건 중복 지도(auxiliary
supervision)다. task-space 모션을 명시적으로 표현하도록 action expert를 압박한다.

우리 §16의 "FK EE를 state로 넣어도 팔은 동률"은 EEF를 **입력**으로 준 실험이라 기전이
다르다. 정보를 더 주는 것과, 그 표현을 만들도록 손실을 거는 것은 별개다.

## 왜 state에도 넣어야 하는가

`ActionType.EEF` + `RELATIVE`이면 델타를 현재 EEF 프레임에서 잡는다. 그 기준 프레임을
`StateActionProcessor._convert_to_relative_action`이
`EndEffectorPose.from_action_format(reference_state, action_format)`로 만들고,
`_convert_to_absolute_action`은 `reference_state.shape[0] == action.shape[1]`을 assert한다.
**즉 기준 state 블록이 action과 같은 9D(xyz+rot6d)여야 한다.** 기존 `left_eef`는 6D
(xyz + extrinsic-xyz euler)라 그대로는 못 쓴다.

## 무엇을 계산하는가

  state.{left,right}_eef_9d   기존 6D `*_eef`의 euler를 rot6d로 바꾼 것. **FK를 다시 돌리지
                              않는다** — 기존 블록에서 정확 변환하므로 `left_eef`와
                              `left_eef_9d`가 어긋날 여지가 없다.
  action.{left,right}_eef_9d  FK(action.{left,right}_arm). action 블록에는 eef가 없으므로
                              지령 관절에서 새로 풀어야 한다. waist는 0으로 고정 —
                              데이터셋의 `_ikea.eef_frame.waist_joints = "excluded_from_fk"`.

FK는 배포 코드 `url_groot_deploy/g1_kinematics.py`의 `G1WristKinematics`를 그대로 쓴다.
`FK(state.arm)`을 저장된 `state.eef`와 대조해 1.5e-08 m / 5.8e-08 rad로 재현됨을 확인했고,
이는 §0에 기록된 값과 같다. URDF sha256도 modality.json의 기대값과 일치한다.

## 규약

rot6d는 회전행렬의 **첫 두 행**을 편 것이다 (`gr00t/data/state_action/pose.py`의
`_matrix_to_rot6d` = `R[:2,:].flatten()`). euler는 scipy `"xyz"` = extrinsic으로,
데이터셋의 `euler_extrinsic_xyz_rad`와 같다. 둘 다 repo 구현에서 확인했다.

## 덧붙이기만 한다

state 117 -> 135, action 33 -> 51. 기존 블록은 인덱스까지 그대로라 같은 데이터셋을
armvel config(60차원 state / 16차원 action)로도 계속 쓸 수 있다. 통계는 폭이 바뀌었으므로
재생성해야 한다.

사용:
    python examples/unitree_g1_dex1_ikea/make_eef_action_variant.py \
        --src-prefix datasets/carroll511/G1_Dex1_IKEA_leg_30hz \
        --dst-prefix datasets/carroll511/G1_Dex1_IKEA_leg_30hz_eef
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


DEPLOY_DIR = "/home/chan/IKEA/url_lerobot/url_groot_deploy"
XR_REPO_DIR = "/home/chan/IKEA/url_lerobot/xr_teleoperate"

# 덧붙일 블록. state/action 양쪽에 같은 이름·같은 폭으로 들어간다.
#
# **이름은 RAMEN(=NVIDIA)의 것을 그대로 쓴다.** 임의로 지으면 학습이 시작조차 못 한다:
# REAL_G1 태그로 돌리면 프로세서가 생성 시점에 베이스 체크포인트의 real_g1 통계로
# `StateActionProcessor._compute_normalization_parameters`를 통과해야 하는데, 거기서
# RELATIVE인 action 키가 `statistics["relative_action"]`에 없으면 ValueError를 던진다.
# NVIDIA의 real_g1 relative_action 키는
#     [left_wrist_eef_9d, right_wrist_eef_9d, left_arm, left_hand, right_arm, right_hand]
# 이므로 `left_eef_9d` 같은 이름은 거기서 죽는다. (실제 값은 이 검증 직후 우리 데이터셋
# 통계로 덮인다 — R 런의 dataset_statistics.json에서 확인함.)
EEF_BLOCKS = [("left_wrist_eef_9d", 9), ("right_wrist_eef_9d", 9)]


def _load_fk():
    sys.path.insert(0, DEPLOY_DIR)
    from g1_kinematics import G1WristKinematics

    return G1WristKinematics(XR_REPO_DIR, waist_zero=True)


def _euler_to_rot6d(rpy: np.ndarray) -> np.ndarray:
    """(N, 3) extrinsic-xyz euler -> (N, 6) rot6d (첫 두 행)."""
    m = Rotation.from_euler("xyz", rpy).as_matrix()  # (N, 3, 3)
    return m[:, :2, :].reshape(len(m), 6)


def _rot6d_to_euler(rot6d: np.ndarray) -> np.ndarray:
    """검증용 역변환. 직교화 없이 그대로 되돌린다."""
    r = rot6d.reshape(-1, 2, 3)
    third = np.cross(r[:, 0], r[:, 1])
    m = np.concatenate([r, third[:, None, :]], axis=1)
    return Rotation.from_matrix(m).as_euler("xyz")


def convert_split(src: Path, dst: Path, fk) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    mod = json.loads((src / "meta/modality.json").read_text())
    s_blk = {k: (v["start"], v["end"]) for k, v in mod["state"].items()}
    a_blk = {k: (v["start"], v["end"]) for k, v in mod["action"].items()}
    s_width = max(v[1] for v in s_blk.values())
    a_width = max(v[1] for v in a_blk.values())

    for f in ("episodes.jsonl", "tasks.jsonl", "info.json"):
        shutil.copy2(src / "meta" / f, dst / "meta" / f)

    # modality.json: 뒤에 덧붙이기만 한다
    s_start, a_start = s_width, a_width
    for name, w in EEF_BLOCKS:
        mod["state"][name] = {"start": s_start, "end": s_start + w}
        s_start += w
        mod["action"][name] = {"start": a_start, "end": a_start + w}
        a_start += w
    (dst / "meta/modality.json").write_text(json.dumps(mod, indent=4))

    info = json.loads((dst / "meta/info.json").read_text())
    for feat, width in (("observation.state", s_start), ("action", a_start)):
        if feat in info.get("features", {}):
            info["features"][feat]["shape"] = [width]
            info["features"][feat].pop("names", None)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))

    for vid in sorted((src / "videos").rglob("*.mp4")):
        out = dst / vid.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.hardlink_to(vid)

    al, ar = a_blk["left_arm"], a_blk["right_arm"]
    el, er = s_blk["left_eef"], s_blk["right_eef"]

    n_ep = n_row = 0
    worst_roundtrip = 0.0
    lag_pos = []
    for pq in sorted(src.glob("data/chunk-*/episode_*.parquet")):
        d = pd.read_parquet(pq)
        S = np.stack(d["observation.state"]).astype(np.float64)
        A = np.stack(d["action"]).astype(np.float64)

        # state: 기존 6D euler -> rot6d (FK 재실행 없음)
        s_eef = []
        for lo, hi in (el, er):
            blk = S[:, lo:hi]
            s_eef.append(np.concatenate([blk[:, :3], _euler_to_rot6d(blk[:, 3:])], axis=1))
            back = _rot6d_to_euler(s_eef[-1][:, 3:])
            # euler는 ±pi에서 감기므로 회전행렬로 비교한다
            m0 = Rotation.from_euler("xyz", blk[:, 3:]).as_matrix()
            m1 = Rotation.from_euler("xyz", back).as_matrix()
            worst_roundtrip = max(worst_roundtrip, float(np.abs(m0 - m1).max()))

        # action: FK(지령 관절)
        a_eef = []
        for (lo, hi), side in ((al, "left"), (ar, "right")):
            q = A[:, lo:hi]
            pose = np.array([fk.wrist_pose(side, q[i]) for i in range(len(q))])
            a_eef.append(np.concatenate([pose[:, :3], _euler_to_rot6d(pose[:, 3:])], axis=1))

        # 온전성: 지령 EEF는 2~3프레임 뒤 측정 EEF와 가까워야 한다 (§0의 action-state lag)
        if len(S) > 3:
            lag_pos.append(float(np.abs(a_eef[0][:-3, :3] - s_eef[0][3:, :3]).mean()))

        d["observation.state"] = list(np.concatenate([S] + s_eef, axis=1).astype(np.float32))
        d["action"] = list(np.concatenate([A] + a_eef, axis=1).astype(np.float32))
        out = dst / pq.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(out, index=False)
        n_ep += 1
        n_row += len(d)

    print(
        f"  {dst.name}: {n_ep} ep / {n_row} frame, "
        f"state {s_width}->{s_start}, action {a_width}->{a_start}"
    )
    print(f"    rot6d 왕복 최대오차 {worst_roundtrip:.3e} (회전행렬 원소)")
    print(f"    지령EEF vs 3프레임 뒤 측정EEF 평균거리 {np.mean(lag_pos) * 1000:.2f} mm")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-prefix", required=True, type=Path)
    ap.add_argument("--dst-prefix", required=True, type=Path)
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()

    fk = _load_fk()
    for split in args.splits.split(","):
        convert_split(
            args.src_prefix.with_name(args.src_prefix.name + f"_{split}"),
            args.dst_prefix.with_name(args.dst_prefix.name + f"_{split}"),
            fk,
        )
    print("DONE")


if __name__ == "__main__":
    main()
