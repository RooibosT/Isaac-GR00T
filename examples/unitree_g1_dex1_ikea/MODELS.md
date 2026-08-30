# GR00T 체크포인트 목록

`chan@143.248.94.3:/media/chan/T9/IKEA/hub/models` 로 전송한 모델 목록.
2026-08-30 기준. 상세 근거는 같은 폴더의 `EXPERIMENTS.md` 절 번호를 참조.

## 폴더 구조 — `models/` 와 `outputs/` 의 차이

| | 무엇인가 | 어디서 왔나 |
|---|---|---|
| **`models/`** | 학습의 **입력** | 남이 만든 것 — NVIDIA 베이스, 웜스타트 소스, 팀 배포본 |
| **`outputs/`** | 학습의 **결과** | 우리가 돌린 런. `<런이름>/<런이름>/checkpoint-N/` |

레포 규약 그대로다 (`run_finetune_ikea.sh` 가 `--base-model-path models/...` 를 읽고
`--output-dir outputs/<실험명>` 에 쓴다). 즉 `models/` 를 지우면 새 학습을 못 돌리고,
`outputs/` 를 지우면 지금까지의 결과를 잃는다.

## 공통 레시피

`nvidia/GR00T-N1.7-3B` 에서 시작, backbone frozen (`tune_llm=False`, `tune_visual=False`).
학습되는 것은 projector + DiT 1.09B (전체 3.14B).

| | |
|---|---|
| 카메라 | 3뷰 — `cam_left_high`, `cam_left_wrist`, `cam_right_wrist` |
| action | 16차원 — 팔 7+7 **RELATIVE**, 그리퍼 1+1 **ABSOLUTE** |
| horizon | 40 @ 30 Hz (배포는 앞 ~8스텝만 실행) |
| batch | effective 64 (global 16 × accum 4), 2×A100 DDP + `--ddp-comm-bf16` |
| 스케줄 | cosine, warmup 5%, lr 1e-4, wd 1e-5, `state_dropout 0.2` |

**state 차원은 config 로 갈린다:**

| 차원 | config | 내용 |
|---:|---|---|
| 46 | `g1_dex1_ikea_relarm_3view_aug_config.py` | legs 12 + waist 3 + 양팔 14 + 그리퍼 2 + base_gravity 3 + FK EE 12 |
| 60 | `g1_dex1_ikea_armvel_config.py` | 46 + **양팔 관절 속도 14** (`right_arm` 뒤에 삽입) |
| 74 | `g1_dex1_ikea_armvel_torque_config.py` | 60 + **양팔 관절 토크 14** |

> ⚠️ 60/74차원 모델은 **추론 때 그 입력이 반드시 있어야 한다.** 속도에 0을 먹이면 실행 구간이
> +26% 나빠져서, 애초에 속도가 없던 46차원 모델보다도 못하다 (§12).

---

## A. IKEA 현행 계보 — 배포 후보 (5태스크)

태스크: `pick and place the table leg` · `rotate leg to tighten` ·
`turn the tabletop square` · `flip table`

| 런 | ckpt | state | 한 줄 설명 |
|---|---:|---:|---|
| **`..._2h_r1v`** | **28000** | 60 | ⭐ **지표상 최선.** rotate 를 재취득 양손본 + set2 로 갈고 **한 instruction 으로 통합**. 전체 그리퍼 0.2105 로 전 모델 최선 (§24) |
| `..._2h_rv` | 28000 | 60 | 같은 데이터인데 rotate 를 **두 instruction 으로 분할** — 손해로 판명 (그리퍼 +2.3%) |
| `..._2h_r` | 30000 | 46 | 분할본의 46차원판. arm velocity 를 못 쓸 때 대비 |
| **`..._pnp_pv`** | 28000, 30000 | 60 | **pick+insert 병합 + 속도.** 허브 업로드됨. 분리 라벨 대비 그리퍼 −26~42% (§23) |
| `..._pnp_p` | 30000 | 46 | 병합본 46차원판 |
| `..._pnp_pvt` | 30000 | **74** | 병합 + 속도 + **팔 토크**. 개루프 −29% 이지만 **지름길로 판정 — 이 숫자로 배포하지 말 것** (§23) |
| `..._leg_armvel` | 40000 | 60 | insert 2배 데이터(`IKEA_pickuptheleg`) **3태스크 전용**. insert EE8 −2.2% |

### 배포 시 instruction 순서

`..._2h_r1v` / `..._pnp_pv` 기준, 다리 하나당 2개 + 나머지 2개:

1. `pick and place the table leg`
2. `rotate leg to tighten`
3. `turn the tabletop square`
4. `flip table`

## B. 라벨 ablation 대조군 (§18 · §19)

| 런 | ckpt | state | 한 줄 설명 |
|---|---:|---:|---|
| `..._alltask_u` | 26000 | 46 | rotate 3세트를 **한 문자열**로 (unified) — §23 비교 기준 |
| `..._alltask_s` | 26000 | 46 | set2 만 다른 문자열로 분할 |
| `..._alltask_n` | 26000 | 46 | u 와 같되 rotate 를 자연스러운 이름 `rotate table base` 로 |
| `..._alltask_x` | 26000 | 46 | **새 태스크만** 학습 (기존 3태스크 제외) |
| `..._alltask_uv` | 22000, 26000 | 60 | u + 속도. **이전 배포본** (허브 업로드됨) |
| `..._alltask_m16` | 18000, 28000 | 46 | 기존 태스크 `mix_ratio` **×1.6** — insert EE8 −5.70% |
| `..._alltask_m30` | 24000, 28000 | 46 | `mix_ratio` **×3.0** |
| `..._alltask_mv` | 18000, 28000 | 60 | ×1.6 + 속도 — uv 보다 insert 나쁨 |

## C. v2 데이터셋 세대 (§12 · §14) — 3태스크

| 런 | ckpt | state | 한 줄 설명 |
|---|---:|---:|---|
| `..._v2` | 20000 | 46 | v2 재변환본 기준선 (R1) |
| `..._v2_armvel` | 20000 | 60 | R3 — **arm velocity 최초 확인** (팔 1–8스텝 −16.7%) |
| `..._v2_30k` | 26000 | 46 | 20k→30k 연장. 그리퍼만 −6.4% |
| `..._v2_armvel_30k` | 26000 | 60 | 속도 30k. "그리퍼는 26k까지 개선" 의 근거 (§14) |

## D. v1 기준선

| 런 | ckpt | state | 한 줄 설명 |
|---|---:|---:|---|
| `ikea_relarm_3view_aug_b64` | 20000 | 46 | v1 데이터 최초 IKEA 모델. §1~8 표 전체의 기준선 |

## E. BCT — Building Children's Table 사전학습 계보

500k 프레임, effective batch 192~256. IKEA 와 데이터셋도 val 도 다르므로 **수치를 IKEA 와 직접
비교하지 말 것.**

| 런 | ckpt | 한 줄 설명 |
|---|---:|---|
| `bct_joint_30hz_b256` | 7500 | 전 키 ABSOLUTE 19차원. 아래 A/B 의 기준선 |
| `bct_joint_30hz_relarm_b256` | 12500 | **팔 RELATIVE 확정** — first-5 MAE −33%. 지금 모든 모델이 이 결정 위에 있다 |
| `bct_..._relarm_3view_clean_b256_abl` | (root) | 라운드1 **3뷰 채택** 승자. 4뷰는 같은 이득에 GPU +33% + 카메라 1대라 기각 |
| `bct_..._relarm_3view_aug_b256_abl` | — | 라운드2 **채택본의 메타만**. 가중치는 `models/bct-relarm-aug-adopted` 로 일원화 |

> ⚠️ 이름 주의: 삭제된 `..._b256_final` 은 "최종" 이 아니라 **기각된 라운드3 30k** 였다.
> 실제 채택본은 `_abl` (라운드2 ckpt-15000) 이다.

## F. `models/` — 학습 입력

| | 크기 | 설명 |
|---|---:|---|
| `GR00T-N1.7-3B` | 6.5G | **NVIDIA 베이스 모델 — 없으면 새 학습 불가** |
| `bct-relarm-aug-adopted` | 12G | BCT 라운드2 ckpt-15000. 웜스타트 소스. 허브 공개본 `RooibosT/gr00t-n1.7-g1-dex1-bct-relarm-aug-30hz-h40` 과 동일 |
| `Team-RAMEN/` × 3 | 39G | 팀 배포 모델 — `insert_leg`, `insert_and_tighten`, `rotate_leg_to_tighten` 각 13G |

> BCT 웜스타트는 **§13 에서 사용 중단 결론**이 났다 (실기에서 개루프의 맞교환이 재현되지 않음).
> 소스는 계보 추적용으로 남긴다.

---

## 허브 업로드 대응표

**URL-RFM (전부 Private)**

| repo | 로컬 |
|---|---|
| `gr00t-n1.7-g1-dex1-ikea-pnp-armvel-30hz-h40` | `..._pnp_pv` / 28000 |
| `gr00t-n1.7-g1-dex1-ikea-pnp-armvel-torque-30hz-h40` | `..._pnp_pvt` / 30000 |
| `gr00t-n1.7-g1-dex1-ikea-alltask-armvel-30hz-h40` | `..._alltask_uv` / 26000 |
| `gr00t-n1.7-g1-dex1-ikea-alltask-30hz-h40` | `..._alltask_u` |
| `gr00t-n1.7-g1-dex1-ikea-alltask-rotsplit-30hz-h40` | `..._alltask_s` |

**RooibosT (공개)** — `bct-relarm-aug-30hz-h40`, `bct-joint-30hz-h40`,
`ikea-relarm-30hz-h40` 및 `-v2` / `-v2-30k` / `-v2-armvel` / `-v2-armvel-30k` / `-v2-bctinit`

**최신 `..._2h_r1v` 는 아직 허브에 없다.**

---

## 이 전송에 없는 것

| 런 | 크기 | 이유 |
|---|---:|---|
| `ikea_ramen_b16_leg_r` | 223G | 200k 완주했으나 **val 미실시** — 베스트 ckpt 를 못 골라서 18개를 다 보내는 건 낭비 |
| `ikea_ramen_b16_leg_reef` | 211G | **학습 중** |

둘 다 나중에 `sync_models_to_t9.sh` 를 인자 없이 다시 돌리면 추가된다
(`--partial --append-verify` 라 이미 옮긴 것은 건드리지 않는다).

## 체크포인트를 고르는 방법

**개루프 스캔이 유일한 선택 신호다.** `eval_loss` 는 액션 품질을 따라가지 않는다 — 정확도가
좋아지는 동안에도 올라간다 (§4).

```bash
python examples/unitree_g1_dex1_ikea/scan_ikea.py \
  --checkpoints-dir <런>/<런> --dataset-path <val> \
  --config examples/unitree_g1_dex1_ikea/<config>.py --output scan.json
```

읽을 때는 **1–8스텝 수치**를 본다. 비동기 배포는 지연 보정 후 40스텝 중 8개 정도만 실행하고
나머지는 RTC 오버랩에 쓰므로, 후반 청크는 실행되지 않는다 (§12).

**라벨 체계가 다른 모델끼리 비교할 때는 반드시 같은 val 에 `--relabel` 로 올릴 것.** 각자 자기
val 에서 채점하면 대조군에 "instruction 을 정확한 순간에 바꿔주는" 특권이 생겨서 결론이
뒤집힌다 (§23).
