# Team-RAMEN 서브태스크 정책 — 분해와 우리 스택 대비

2026-08-28. IROS 2026 Team-RAMEN이 공개한 GR00T N1.7 IKEA 서브태스크 정책 3종을
체크포인트에서 직접 뜯어보고 우리 라인과 대조한 기록.

- 대상: `Team-RAMEN/IROS2026_RAMEN_takada_{insert_leg, rotate_leg_to_tighten, insert_and_tighten}_optimal_gr00t_200k`
- 받아둔 곳: `models/Team-RAMEN/` (각 13.82 GB, fp32, 3.455 B)
- 근거: 공개 config·processor 파일 + `nvidia/GR00T-N1.7-3B`와의 텐서 단위 diff
- 상세 문서: <https://claude.ai/code/artifact/5e5cae86-8d5d-453b-a6ed-3007a1e1c1d5>

원본 데이터셋은 비공개(404). **에피소드 수만 확정값**이고 프레임·epoch은 추정이다.

---

## 1. 같은 것 — 대부분

양쪽 다 같은 베이스에서 출발해 독립적으로 같은 결론에 도달한 항목들.

| | 값 |
|---|---|
| 베이스 모델 | `nvidia/GR00T-N1.7-3B` |
| freeze | `tune_llm=False` `tune_visual=False` / projector·DiT·vlln 학습 |
| 실학습 비율 | 3.46 B 중 1.30 B (37.8%) — 백본 1.83 B는 bit-identical |
| 정규화 방식 | q01/q99 min–max + 클리핑 (`use_percentiles=True`) |
| 카메라 | 머리 + 양 손목 3뷰, history 없음 (`delta_indices=[0]`) |
| action horizon | 40 |
| 팔 / 그리퍼 표현 | 팔 RELATIVE joints, 그리퍼 ABSOLUTE 1 dim/측 |
| lr / wd / warmup | 1e-4 / 1e-5 / 5% cosine |
| state_dropout | 0.2 |
| GPU당 micro-batch / 런당 GPU | 8 / 2장 |

**freeze는 고를 것이 없다.** 우리 `FinetuneConfig` 기본값과 네 플래그가 정확히 같고,
텐서 대조로 백본 1834.7 M이 bit-for-bit 동일함을 확인했다.

---

## 2. 다른 것 — 10개 축

| 축 | RAMEN | 우리 |
|---|---|---|
| **서브태스크** | 태스크별 분리 정책 3개 (+ insert·tighten 융합 정책) | 단일 멀티태스크, instruction으로 선택 |
| **projector 슬롯** | slot 25, **사전학습됨** (L2 8.31) | slot 10 (`NEW_EMBODIMENT`), **초기화 상태** (L2 7.37) |
| **정규화 출처** | NVIDIA 사전학습 통계 **상속** | `gr00t.data.stats`로 자체 산출 |
| **액션 공간** | 53 dim / 37 지도. 손목 EEF 9D를 **예측** | 16 dim. EEF는 state 전용(6D) |
| **배치 / 스텝** | global 16 × 200k = 3.2 M 샘플 | eff 64 × 26–30k = 1.7–1.9 M 샘플 |
| **실행 창** | 40 중 **16** | 40 중 **8** |
| **이미지 증강** | ColorJitter + sharpness + **RandomAffine ±5°/5%** | ColorJitter + random crop. 기하 변환 없음 |
| **체크포인트 선정** | eval loss, 10k마다 ≤512 샘플 | open-loop 스캔 (arm8/EE8/그리퍼) |
| **데이터 큐레이션** | 학습 리스트에서 9% 제외 (`curated_optimal`) | 변환 무손실, 학습 후 스캔에서 선택 |
| **val 분할** | 비율 10–11%, 세션 고려 여부 불명 | 녹화 **세션 경계**로 분할 |

---

## 3. 데이터 — 가장 크게 갈리는 곳

### 규모

| 세트 | train ep | frames | f/ep | H40 윈도우 |
|---|---:|---:|---:|---:|
| **RAMEN** `insert_table_leg` | **1,339** | 59–73만 (추정) | — | — |
| **RAMEN** `rotate_leg_to_tighten` | 361 | 16–20만 (추정) | — | — |
| **RAMEN** `insert_and_tighten` | 258 | 11–14만 (추정) | — | — |
| 우리 `table_30hz` (v1) | 250 | 59,589 | 238 | 49,839 |
| 우리 `table_30hz_v2` (armvel 채택본) | 163 | 71,715 | 440 | 65,358 |
| 우리 `all_30hz_unified` (U) | 282 | 128,200 | 455 | 117,202 |
| 우리 `all_30hz_pnp` | 236 | 128,200 | 543 | 118,996 |
| 우리 `leg_30hz` | 307 | 141,947 | 462 | 129,974 |

프레임 추정은 우리 실측 440–543 f/ep를 대입한 값.
**목표 태스크 데이터가 에피소드로 4.4배, 프레임으로 4.2–5.1배 차이난다.**

### ⚠️ epoch은 오히려 뒤집힌다

| 런 | steps | batch | 샘플 | epoch |
|---|---:|---:|---:|---:|
| RAMEN insert | 200,000 | 16 | 3.20 M | **4.7–6.0** |
| RAMEN rotate | 200,000 | 16 | 3.20 M | 17.6–22.1 |
| RAMEN ins+tight | 200,000 | 16 | 3.20 M | 24.6–30.9 |
| 우리 armvel (선정) | 26,000 | 64 | 1.66 M | **25.5** |
| 우리 leg | 40,000 | 64 | 2.56 M | 19.7 |
| 우리 pnp | 30,000 | 64 | 1.92 M | 16.1 |

**200k step은 보이는 것보다 작다.** RAMEN은 accum 없이 global batch 16이라
(`effective_bs = batch_size × num_processes`) 총량이 우리의 6.7배가 아니라 **1.7–1.9배**다.
그리고 insert 세트가 워낙 커서 그 런은 우리 26k보다 **데이터를 적게 돈다.**
세 태스크에 같은 200k를 준 결과, 데이터 크기에 따라 epoch이 4–6배 벌어져 있다.

### 취사선택 지점

RAMEN의 `dataset.episodes`는 구멍 뚫린 리스트다 — insert 1,510/1,664(154개 제외, 9.3%),
rotate 402/439(37개), ins+tight 289/318(29개). **그들은 입력에서 걸러내고, 우리는
출력(스캔)에서 고른다.**

---

## 4. state는 dim 대 dim 비교가 안 된다

| 블록 | RAMEN | 우리 (46) | 우리 (60) |
|---|---:|---:|---:|
| 손목 EEF | 9+9 (rot6d) | 6+6 | 6+6 |
| 팔 관절 | 7+7 | 7+7 | 7+7 |
| 팔 각속도 | — | — | 7+7 |
| 핸드/그리퍼 | 7+7 | 1+1 | 1+1 |
| 허리 | 3 | 3 | 3 |
| 다리 | — | 12 | 12 |
| base gravity | — | 3 | 3 |
| **합계** | **49** | **46** | **60** |

- RAMEN의 핸드 14 dim 중 **12개가 Dex1에서 죽은 패딩**이다 (`loss_excluded_indices`가
  각 측 dim 1–6과 `base_height`·`navigate`를 제외 → 53 중 37만 지도).
  NVIDIA projector를 유효하게 유지하려고 Dex3 스키마 값을 치르는 것.
- 우리는 다리·base gravity를 싣고, 채택 config엔 **팔 각속도**가 있다 (arm8 −17.9%,
  EE8 −11.3%, §16). **그들 레이아웃엔 이걸 넣을 자리가 없다** — 넣는 순간 상속한
  projector가 무의미해진다.

---

## 5. 서브태스크 — 우리 쪽에만 측정치가 있다

| 실험 | 결과 | 효과 |
|---|---|---|
| §18 방향 분할 | 라벨을 나누니 그 태스크가 **나빠짐**. 모델이 이미 관측에서 방향을 읽고 있었다 | arm8 **+2.64%** (대역 ±0.26) |
| §23 pick+insert 병합 | 라벨 컷이 남긴 무지도 전환 구간이 복구됨 | 그리퍼 **−25 ~ −43%** |
| §18 무관 태스크 추가 | `insert`에 간섭, 스텝으로 안 메워짐 | EE8 +12.06% |

우리 근거: **공유 동작을 쪼개면 손해, 인접 단계 병합은 이득, 무관 태스크는 간섭.**
RAMEN 설계는 두 번째와 일치하고(세 번째 정책이 정확히 insert+tighten 병합), 세 번째는
한 모델에 무관 태스크를 안 넣어 피해간다. 갈리는 건 첫 번째다.

> **⚠️ 비교 방법.** §23에서 확인했듯 각 모델을 *자기* val에서 채점하면 결론이 뒤집힌다 —
> 분리 모델에 실기엔 없는 오라클 instruction 전환을 주기 때문. RAMEN 체크포인트와
> 비교하려면 **하나의 공유 val에 올리고 instruction 하나를 끝까지 쥐어야 한다.**
> 그러지 않으면 분리 설계가 구조적으로 이긴다.

---

## 6. 가져올 것 / 안 가져올 것

**✅ RandomAffine 증강 — 가장 싸다.** ±5° 회전 + 5% 평행이동은 우리에게 없는 기하
변형이다. §16에서 이미지 증강은 미확정으로 남겨둔 항목(color jitter ×0.5만 봤고 0으로 끈
런 없음)이고, frozen 인코더는 color jitter로 불변성을 못 배우지만 기하 변환은 action
expert가 견뎌야 할 대상 자체를 바꾸므로 기전이 다르다. config만 고치면 된다.

**❌ 사전학습 projector 슬롯 — 패키지로 온다.** `REAL_G1`(slot 25)를 쓰면 인코더가 노이즈가
아니라 맞춰진 가중치에서 시작한다. 다만 그 가중치는 NVIDIA의 49/53 레이아웃에서만 의미가
있어서, 채택하려면 state 벡터를 통째로 받아야 하고 **팔 각속도와 base gravity를 잃고 죽은
핸드 12 dim을 얻는다.** 수지가 안 맞는다. 단, *그들이 global batch 16으로도 되는 이유*
설명으로는 유효하다.

**➡️ 목표 태스크 데이터.** §18이 이미 `insert`의 근본 해법을 스텝이 아니라 에피소드 추가
취득으로 결론냈고 §22에서 2배로 늘렸다. RAMEN의 1,339 ep는 같은 태스크에서 "충분"의
크기를 보여주는 참고값이다.

---

## 부록 — 확인 방법

```bash
# processor/config 세 모델 대조 (전처리·후처리는 바이트 동일, config는 repo_id만 다름)
sha256sum models/Team-RAMEN/*/policy_pre*.json models/Team-RAMEN/*/policy_post*.json

# 정규화 통계가 베이스와 같은지
python - <<'EOF'
import json, numpy as np
b=json.load(open("models/GR00T-N1.7-3B/statistics.json"))["real_g1_relative_eef_relative_joints"]
r=json.load(open("models/Team-RAMEN/IROS2026_RAMEN_takada_insert_leg_optimal_gr00t_200k/"
                "policy_preprocessor.json"))["steps"][2]["config"]["raw_stats"]
print(all(np.allclose(b[g][k][t], r[g][k][t])
          for g in r for k in r[g] for t in ("q01","q99")))   # -> True
EOF
```

`embodiment_id.json`에 `new_embodiment`가 없다는 것과, base 체크포인트에서 slot 10이
미사용 슬롯 baseline(L2 7.33–7.38)에 있고 slot 25가 8.31이라는 것은
`action_head.state_encoder.layer1.W`를 슬롯별로 재보면 확인된다.
