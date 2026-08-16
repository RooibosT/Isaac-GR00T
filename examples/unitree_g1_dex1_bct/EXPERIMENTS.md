# BCT 30Hz 실험 기록

2026-08-14~ / 서버·속도 세팅은 [A100_TRAINING_NOTES.md](../../A100_TRAINING_NOTES.md) 참고.

## 데이터셋

| 이름 | 내용 | 상태 |
|---|---|---|
| `datasets/RooibosT/G1_Dex1_BCT_subtask_joint_30hz` (+`_train`/`_val`) | 30fps 재추출본. 1,319클립 / 541,791프레임, 태스크당 60분 균형, v2.1. 샤드 머지 file_index 버그 수리 완료(매핑 복원 + 전체 재인코딩, 전수 검증) | 학습 사용 중 |
| `datasets/RooibosT/G1_Dex1_BCT_subtask_joint` | 15Hz 원저자본, v2.1 변환 + 손상 4개 수리 | 보관 |
| `datasets/RooibosT/G1_Dex1_BCT_subtask_ee` | 15Hz EE 14차원, v2.1 변환 완료 (스캔/수리 미실시) | 학습 보류 |
| `datasets/G1_WBT_Dex1_Building-Children-Table` | 원본 parquet/meta만 (3.3GB, 비디오 삭제) — cmd_vel 유도 등 재가공용 | 보관 |

## 학습된 / 학습 중 모델

공통 레시피: effective batch 256, lr 1e-4, warmup 0.05, wd 1e-5, cosine, state_dropout 0.2,
horizon 40, 3뷰(cam_head + 양 손목), state 31차원, 4×A100 DDP (`--use-ddp`), seek approximate.

### 1. joint_30hz — 완료 ✅
- **전 키 ABSOLUTE** (waist+arms+grippers 19차원 @30Hz)
- 50k 스텝 상한 중 13,477에서 조기 종료. **최적 = ckpt-7500** (4.1 epochs)
- val open-loop: 액션 MSE 0.0295, 팔 MAE 4.4~4.7°, 허리 0.66°, 그리퍼 0.21u(0-4.5), h1 MAE 0.037rad
- 호라이즌 오차 성장 h1→h40: 0.037→0.120 → **배포 재추론 주기 10~20스텝 권장**
- HF: `RooibosT/gr00t-n1.7-g1-dex1-bct-joint-30hz-h40` (private)
- **교훈: eval_loss(flow-matching 목적함수)는 7.5k 이후 올라갔지만 액션 지표는 정체(악화 아님).
  체크포인트 선택은 eval_loss가 아니라 open-loop 스캔(`eval_val_mse.py`)으로 할 것**

### 2. joint_30hz_relarm — 완료 ✅ **(open-loop A/B 승리)**
- **팔 RELATIVE + 허리·그리퍼 ABSOLUTE** (`g1_dex1_bct_joint_relarm_config.py`)
- 근거: CK-Sung brainco 모델(팔 REL + 핸드 ABS)의 실배포 성공. redblock의 ABS>>REL ablation은
  배포 교란이 있었음(README도 A40을 "confounded"로 기록) → 교란 없는 재검증
- 12,502스텝에서 종료(eval_loss 반등 기준), **최적 = ckpt-12500** (스캔상 마지막까지 개선 중이었음)

| 지표 (val 66eps, open-loop) | relarm ckpt-12500 | joint_30hz ckpt-7500 (ABS) | 차이 |
|---|---|---|---|
| 액션 MSE | **0.0270** | 0.0295 | −8% |
| MAE | **0.0753** | 0.0826 | −9% |
| **first-5 MAE** | **0.0282** | 0.0422 | **−33%** |
| 팔 MAE | **0.0700 (4.0°)** | 0.0790 (4.5°) | −11% |
| 그리퍼 MAE | 0.208 | 0.215 | 동급 |

- **RELATIVE가 근미래(청크 초반) 정확도에서 특히 강함** — 예측이 현재 상태에 앵커되므로 구조적 이점.
  최종 판정은 실기 closed-loop 비교로.

### ⚠️ 가장 중요한 교훈: eval_loss로 판단하지 말 것
두 런에서 eval_loss(flow-matching 목적함수)와 실제 액션 품질이 **서로 다른 방향으로** 갈렸다:
- joint_30hz: eval_loss 7.5k 이후 상승, 액션 지표는 **정체**(악화 아님)
- relarm: eval_loss 7.5k 이후 상승, 액션 지표는 **12.5k까지 계속 개선**

→ 체크포인트 선택도, **조기 종료 판단도** `eval_val_mse.py` open-loop 스캔으로 할 것 (체크포인트당 ~3분).
→ VLA는 액션 재현 정확도가 목표라 목적함수가 평평해져도 더 긴 스케줄이 이득일 수 있다
   (OpenVLA도 같은 이유로 스텝을 크게 늘림). **다음 학습은 스텝 상한을 넉넉히 두고
   2,500스텝마다 스캔으로 판단**하는 방식을 기본으로.

### 참고 기준점
- 15Hz joint (원저자): `RooibosT/gr00t-n1.7-g1-dex1-bct-joint-h40`
- CK-Sung brainco: 17k프레임/43epochs/effective 64, REL arms, 실배포 양호

## 앞으로 실험 후보 (우선순위순)

0. **relarm 장기 학습**: ckpt-12500이 스캔 마지막까지 개선 중이었음 → 25~30k 스텝으로 재학습해
   수렴점 확인 (bf16 통신 적용 시 ~16시간). 4-view와 묶어 한 번에 돌리는 것도 방법
1. **4-view**: cam_head_right(cam_1) 추가 — config 한 줄, 30Hz 데이터에 이미 포함. 스테레오 단서 기대, 저위험
2. **ee (14차원 EE)**: 데이터 준비됨. 보류 사유 — 배포 시 IK 필요, joint 대비 이점 불명확, 레포 기본도 joint
3. **state_dropout 0.1/0.3**: 실기에서 시각/상태 의존 편차가 보일 때만
4. ~~h16/h32~~: 30Hz에서 h16=0.53s로 너무 짧음. 반응성은 h40 청크의 앞 10~20스텝만 실행하는 배포 전략으로 해결
5. ~~cmd_vel (odometry 미분)~~: **실측 기각** — 30Hz 유한차분 SNR 0.2 + 추정기 점프 글리치. 차기 데이터 녹화 때 오퍼레이터 명령을 직접 채널로 기록해서 해결

## 차기 데이터 녹화 시 추가할 채널 (원본 데이터셋의 한계 보완)

- **cmd_vel (vx, vy, ωz)**: 오퍼레이터 베이스 이동 명령 — 로코모션 학습의 전제
- **관절 속도 dq / 토크**: G1 LowState에 이미 있음, 기록 비용 낮음 (진단·후속 알고리즘용)
- 첫 프레임 센서 워밍업 (쿼터니언 [0,0,0,0] 아티팩트 방지), 프레임 단위 서브태스크 라벨링
