# 데이터셋 목록

`chan@143.248.94.3:/media/chan/T9/IKEA/hub/datasets` 로 전송한 데이터셋. 2026-08-30 기준.
68개 / 75 GB (`_train`/`_val` 은 full 본에서 하드링크라 개별 크기의 합보다 훨씬 작다).

## 형식 두 가지

| | 무엇 | GR00T 가 읽나 |
|---|---|---|
| **v3.0** | 허브에서 받은 원본. 전 에피소드를 parquet 하나 + 긴 mp4 몇 개로 묶음 | ❌ |
| **v2.1** | 에피소드마다 parquet 1개 + 카메라마다 mp4 1개 + `meta/episodes.jsonl` | ✅ |

변환기는 `examples/unitree_g1_dex1_ikea/` 에 있다:

| 스크립트 | 하는 일 |
|---|---|
| `convert_ikea_v3_to_v2.py` | IKEA 본 데이터 v3.0 → v2.1. `--merge-tasks` 로 pick+insert 재결합, `--val-sessions` 로 분할 |
| `convert_urlrfm_v3_to_v2.py` | rotate/flip 을 stage1 에서 v2.1 로 자르고, stage2 에서 IKEA 본체와 합침 |
| `convert_lightwheel_v3_to_v2.py` | 시뮬 원본에서 base 정지 구간의 `insert`/`rotate` 만 잘라냄 |

**⚠️ v3.0 export 의 `file_index` 를 믿지 말 것.** `IKEA_pickuptheleg` 은 두 high 카메라의
에피소드 0–177 을 전부 `file_index 1` 로 적어 놓았는데 앞 91 개는 물리적으로 `file-000` 에
있다 — 파일 쌍이 쌍의 마지막 인덱스로 뭉개졌다. wrist 카메라는 파일이 2 개뿐이라 멀쩡해서
피해가 조용하고 부분적이다. 컨버터의 `resolve_file_index` 가 **타임스탬프가 0 으로 리셋되는
지점** 을 세어 복원한다 (타임스탬프는 파일 내부 위치라 같은 방식으로 틀릴 수 없다).

---

## 1. 허브 원본 v3.0 — 17.6 GB

| 데이터셋 | eps | frames | state/act | 설명 |
|---|---:|---:|---|---|
| `G1_WBT_Dex1_Building-Children-Table` | 533 | 6,276,443 | — | **BCT 최상위 원본** |
| `carroll511/IKEA_table_assembly` | 178 | 79,205 | 86/19 | **IKEA 본 데이터** — pick / insert / rotate leg |
| `URL-RFM/IKEA_table_assembly_torque` | 178 | 79,205 | **117**/33 | 위와 **같은 녹화 재export**. state[:86] 와 action[:19] 가 비트 동일, torque 31 차원 추가 |
| `URL-RFM/IKEA_pickuptheleg` | **322** | 149,437 | 117/33 | 위 178 개 + **z축 의도 추가분 144 개**. 앞 178 은 순서·길이까지 동일 |
| `URL-RFM/IKEA_rotatetable1` | 31 | 13,678 | 117/33 | rotate table, 다리 1 개. **한손 기법** |
| `URL-RFM/IKEA_rotatetable2` | 31 | 13,700 | 117/33 | 다리 2 개. 한손. **실기에서 되는 것** |
| `URL-RFM/IKEA_rotatetable3` | 31 | 11,430 | 117/33 | 다리 3 개. 한손 |
| `URL-RFM/IKEA_rotatetable1_v2` | 31 | 11,773 | 117/33 | set1 **양손 기법 재취득**. 오른팔 ROM 4.8배, 13% 빠름 |
| `URL-RFM/IKEA_fliptable` | 43 | 26,405 | 117/33 | flip table |
| `carroll511/IKEA_table_assembly_v1_20260819` | 276 | 66,600 | 86/19 | **v1** — 필터가 빡빡했던 구버전 (§12 에서 대체) |
| `carroll511/lightwheel_lerobot` | 114 | 516,453 | 33/23 | **시뮬 원본** (§15 기각) |
| `carroll511/IKEA_table_assembly_pi05` | 178 | 79,205 | **46/16** | π0.5 용 변환 (다른 라인) |
| `carroll511/g1_brainco_..._v3.0` | 55 | 17,087 | 26/26 | brainco 손 실험 |

### state 117 차원의 구성

`URL-RFM/*` 계열이 전부 이 레이아웃이다:

```
legs 0:12 · waist 12:15 · left_arm 15:22 · right_arm 22:29 · grippers 29:31
legs_vel 31:43 · waist_vel 43:46 · left_arm_vel 46:53 · right_arm_vel 53:60
gripper_vel 60:62 · base_lin_vel 62:65 · base_ang_vel 65:68
base_gravity 68:71 · torso_gravity 71:74 · left_eef 74:80 · right_eef 80:86
legs_torque 86:98 · waist_torque 98:101
left_arm_torque 101:108 · right_arm_torque 108:115 · gripper_torque 115:117
```

학습 config 가 여기서 골라 쓴다 — 46 / 60 / 74 차원은 이 중 어느 블록을 넣느냐의 차이다.

## 2. stage1 v2.1 — 3.2 GB

`URL-RFM/v2/` — rotate·flip 원본을 **에피소드별로 자른 중간 산출물**. stage2 가 이것과 IKEA
본체를 합쳐 학습용을 만든다. 5 개 (`rotatetable1`, `1_v2`, `2`, `3`, `fliptable`).

## 3. IKEA 학습용 변환본 — 26 GB

**⭐ = 현행 모델이 쓰는 것.**

| 데이터셋 | eps | frames | state | 쓰는 모델 |
|---|---:|---:|---:|---|
| ⭐ `..._all_30hz_twohand1_*` | 209 | 116,599 | 86 | **`2h_r1v`** — rotate 통합, 현재 최선 |
| ⭐ `..._all_30hz_twohand_*` | 209 | 116,599 | 86 | `2h_rv` / `2h_r` — rotate 분할 |
| ⭐ `..._all_30hz_pnp_*` | 236 | 128,200 | 86 | `pnp_p` / `pnp_pv` — pick+insert 병합 |
| ⭐ `..._all_30hz_pnptq_*` | 236 | 128,200 | **117** | `pnp_pvt` — 병합 + torque |
| ⭐ `..._leg_30hz_*` | 307 | 141,947 | 117 | `leg_armvel` · ramen 런들 — insert 2 배 |
| `..._leg_30hz_eef_*` | 307 | 141,947 | **135/51** | `ramen_b16_leg_reef` — action 에 EEF 추가 |
| `..._all_30hz_unified_*` | 282 | 128,200 | 86 | `alltask_u`/`uv`/`m16`/`m30`/`mv` |
| `..._all_30hz_split_*` | 282 | 128,200 | 86 | `alltask_s` |
| `..._all_30hz_renamed_*` | 282 | 128,200 | 86 | `alltask_n` |
| `..._newonly_30hz_*` | 119 | 56,485 | 86 | `alltask_x` — 새 태스크만 |
| `..._table_30hz_v2_*` | 163 | 71,715 | 86 | `v2` / `v2_armvel` 등 3 태스크 |
| `..._table_30hz_pnp{,tq}_*` | 117 | 71,715 | 86/117 | 위 pnp 세트의 IKEA 부분 |
| `..._table_30hz_lbl_*` | 163 | 71,715 | 86 | **안 씀** — 라벨만 병합한 변형 |
| `..._table_30hz{,_wa}_*` | 250 | 59,589 | 86 | v1 세대 + waist 19 차원 변형 |
| `G1_Dex1_LW_sim_20hz_still` | 913 | 329,270 | 33 | 시뮬 잘라낸 것 (§15 기각) |

### `_train` / `_val` 분할 규칙

IKEA 본 데이터는 **세션 단위**로 나눈다 — `--val-sessions 3,6,19` (`sessions_v2.json`),
곧 소스 에피소드 `{13,14,15} ∪ {36,37,38} ∪ {89..97}` = 15 개 / 7,490 프레임.
`IKEA_pickuptheleg` 의 앞 178 개가 원본과 같은 순서라 같은 인자로 **같은 val** 이 나온다.
그래서 옛 3 태스크 모델과 새 모델을 재스캔 없이 비교할 수 있다.

rotate·flip 은 소스마다 세션이 하나뿐이라 **에피소드 단위** 로 뽑는다 (seed 20260826,
소스당 4~5 개). `SOURCES` 딕셔너리 순서가 rng 를 소비하므로 **새 소스는 반드시 맨 뒤에
추가** 해야 기존 변형의 val 이 안 바뀐다.

## 4. BCT 학습용 변환본 — 31 GB

`RooibosT/` 11 개. 계보: `joint_30hz`(31차원) → `_aug`(46차원) → `_clean`(정지구간 제거)
→ `_train`/`_val`

| 데이터셋 | eps | state | 비고 |
|---|---:|---:|---|
| `..._joint_30hz_aug_clean_train` | 1249 | 46 | **BCT 최종 모델 학습본** |
| `..._joint_30hz_aug_clean_val` | 66 | 46 | 채점용 |
| 나머지 7 개 | | 31/46 | 중간 단계 |

> 15 fps 원본 3 개 (`subtask_ee`, `subtask_ee_v3.0`, `subtask_joint`, 합 36 GB) 는
> 2026-08-29 에 삭제했다. 30 Hz 변환본이 있고 그 이후 실험은 전부 30 Hz 기준이다.

---

## 알아두면 좋은 관계

- **`IKEA_table_assembly` ⊂ `IKEA_table_assembly_torque` ⊂ `IKEA_pickuptheleg`.**
  첫 둘은 같은 녹화이고 torque 31 차원만 차이, 셋째는 그 178 개에 144 개를 덧붙인 것이다.
  즉 앞의 둘은 이제 부분집합이다.
- **IKEA 원본은 연속 비디오다.** 각 mp4 의 프레임 수가 그 안 에피소드 길이의 합과 정확히
  같고 `[0, N)` 을 빈틈없이 덮는다 — **버려진 프레임이 없다.** `pick` / `insert` /
  `rotate leg` 경계는 스플라이스가 아니라 라벨 절단이고, 그래서 다시 이어붙일 수 있다
  (`--merge-tasks`). 다만 실패해서 재시도한 지점도 잘려 있으므로, 컨버터는 경계의 state
  점프가 그 차원의 에피소드 내부 최대치를 넘으면 병합을 거부한다 (63 개 중 14 개 거부).
- **`meta/stats.json` 은 학습 전에 반드시 생성해야 한다.** 없으면 런처가 거부한다. 여러 런이
  동시에 쓰면 파일이 깨지므로 미리 한 번만 만들어 둘 것. 생성은 `gr00t.data.stats` 의
  `generate_stats` + `generate_rel_stats` 로 한다.
- **`mix_ratio` 도 `episode_sampling_rate` 도 데이터를 버리지 않는다.** 전자는 어느 데이터셋의
  shard 를 다음에 뽑을지의 확률, 후자는 타임스텝을 인터리브해 shard 로 나누는 입도다.
  자세한 것은 `mixtures/README.md`.
