# gr00t-client — Isaac-GR00T N1.7 모델의 Isaac Sim 원격 평가

`unitree_lerobot`의 remote deploy 세팅(`eval_g1_sim_remote.py`)을 **Isaac-GR00T N1.7
체크포인트**에 연결하는 클라이언트 패키지입니다. 기존 `groot-server` 경로는 LeRobot의
GR00T **N1.5** 정책만 로드할 수 있어서 (`"type": "groot"` config), N1.7
(`model_type: Gr00tN1d7`) 체크포인트에는 쓸 수 없습니다.

**구성 (방안 B):** GPU 서버는 Isaac-GR00T의 스톡 `run_gr00t_server.py`(ZMQ+msgpack,
포트 5555)를 그대로 쓰고, 데스크톱 클라이언트만 이 패키지의
`Gr00tSimPolicyClient`로 교체합니다. `url_lerobot` 원본 코드는 수정하지 않습니다.

```
[데스크톱]  Isaac Sim (sim_main.py)          ← 기존 그대로
[데스크톱]  eval_g1_sim_groot.py             ← 이 패키지 (eval_g1_sim_remote.py의 GR00T판)
                │ ZMQ REQ/REP + msgpack, TCP :5555 (SSH 터널 권장)
[GPU 서버]  run_gr00t_server.py (Isaac-GR00T 스톡 서버, 커스텀 코드 없음)
```

## 파일

| 파일 | 실행 위치 | 역할 |
|---|---|---|
| `serve_variant.sh` | GPU 서버 | HF Hub에서 변형(A/B/A40/B40) 체크포인트 받아 스톡 서버 실행 |
| `gr00t_transport.py` | 데스크톱 | gr00t 패키지 없이 동작하는 ZMQ/msgpack 미니 클라이언트 |
| `gr00t_sim_policy_client.py` | 데스크톱 | `RemoteSimPolicyClient` 인터페이스 호환 클라이언트 (관측/액션 형식 변환 + 비동기 액션 버퍼) |
| `eval_g1_sim_groot.py` | 데스크톱 | `eval_g1_sim_remote.py`와 동일 제어 루프, 클라이언트만 교체 |
| `test_roundtrip.py` | GPU 서버 | 서버 기동 후 가짜 관측으로 왕복 검증 |

## 1. GPU 서버 셋업 (추론 서버)

**드라이버/GPU 요구사항:**

- **GPU: Ampere(sm80) 이상** (RTX 30/40, A100, H100 등) — flash-attn 2의 하드 요구사항
- **드라이버: ≥ 525.60.13** — CUDA 12.8 공식 짝은 드라이버 570이지만, torch cu128
  휠은 CUDA 런타임을 휠에 번들하고 CUDA 12.x minor version compatibility가 적용되어
  드라이버 525 이상이면 동작합니다. **driver 535.216.01 (CUDA 12.2) 서버 OK.**
  (전제: GPU가 sm80~90 — torch/flash-attn 휠에 프리컴파일 SASS가 있어 구드라이버의
  PTX JIT 제약을 타지 않음)

### 방법 1 — Docker (권장)

repo 루트에서 추론 전용 이미지 빌드 (base가 CUDA 12.2라 driver 535 호스트에서
그대로 실행됨; NVIDIA Container Toolkit 필요):

```bash
git clone https://github.com/RooibosT/Isaac-GR00T && cd Isaac-GR00T
docker build -f gr00t-client/docker/Dockerfile.inference -t gr00t-inference .
```

실행 — 컨테이너가 시작 시 GPU/flash-attn 커널 체크를 하고, Hub private repo에서
체크포인트를 받아 서버를 띄웁니다:

```bash
docker run --rm -it --gpus all --ipc=host -p 5555:5555 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/gr00t_checkpoints:/checkpoints \
  -e HF_TOKEN=<hf_token> \
  -e MODEL_PATH=RooibosT/gr00t-n1.7-g1-dex1-A \
  gr00t-inference
# "Server is ready and listening" 출력까지 대기
# 변형 교체: Ctrl+C 후 -e MODEL_PATH=...-B / ...-A40 / ...-B40 으로 재실행
```

로컬 체크포인트를 쓰려면 `-v /path/to/ckpt:/checkpoints/model:ro -e MODEL_PATH=/checkpoints/model`.

### 방법 2 — venv 직접 (드라이버 ≥ 525, uv 사용 가능 환경)

```bash
git clone https://github.com/RooibosT/Isaac-GR00T && cd Isaac-GR00T
# 주의: scripts/deployment/*/wheels/*.whl 이 git-lfs 파일 — git lfs install 필요
uv sync                       # 학습 extras 불필요, 기본 sync면 충분
source .venv/bin/activate
hf auth login                 # private 체크포인트 repo 접근용 (RooibosT 계정)
bash gr00t-client/serve_variant.sh A 5555 0
```

| 변형 | 팔 표현 | horizon → 클라이언트 `--actions_per_chunk` |
|---|---|---|
| A   | RELATIVE joint | **16** |
| B   | ABSOLUTE joint | **16** |
| A40 | RELATIVE joint | **40** |
| B40 | ABSOLUTE joint | **40** |

서버 기동 후 같은 머신에서 왕복 검증 (변형 A/B는 `--horizon 16`, A40/B40은 40):

```bash
# venv 방식:
python gr00t-client/test_roundtrip.py --port 5555 --horizon 16
# Docker 방식 (서버 컨테이너 안에서):
docker exec -it <container> python gr00t-client/test_roundtrip.py --port 5555 --horizon 16
# "ROUNDTRIP TEST PASSED" 확인
```

## 2. 데스크톱 셋업

기존 `unitree_lerobot` conda env를 그대로 쓰고, 가벼운 의존성 3개만 추가:

```bash
conda activate unitree_lerobot
pip install pyzmq msgpack msgpack-numpy
```

데스크톱에는 이 `gr00t-client/` 디렉토리만 있으면 됩니다 (Isaac-GR00T repo 전체를
clone하거나 이 디렉토리만 복사 — gr00t 패키지 설치는 불필요).

## 3. 실행 런북 (기존 README §0.2와 동일 순서, 포트만 5555)

```bash
# ── GPU 서버 ──────────────────────────────────────────────
bash serve_variant.sh A 5555 0        # 변형 선택

# ── 데스크톱 터미널 1: SSH 터널 (권장) ──────────────────────
ssh -N -L 5555:localhost:5555 <user>@<gpu_server_ip>

# ── 데스크톱 터미널 2: Isaac Sim (unitree_sim_env) ─────────
conda activate unitree_sim_env
cd /path/to/unitree_sim_isaaclab
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint \
  --enable_dex1_dds --robot_type g129 --keyboard_termination_key r

# ── 데스크톱 터미널 3: deploy 클라이언트 (unitree_lerobot) ──
conda activate unitree_lerobot
cd /path/to/Isaac-GR00T/gr00t-client
python eval_g1_sim_groot.py \
  --policy_server_address=127.0.0.1:5555 \
  --repo_id=RooibosT/g1_pick_redblock_dex1_sim_merged_107demo \
  --actions_per_chunk=16 \
  --frequency=30 --arm=G1_29 --ee=dex1
# 프롬프트에서 's' + Enter
```

A40/B40 평가 시에는 서버 재시작 + 클라이언트 `--actions_per_chunk=40`.

## 4. 동작 원리 (형식 변환)

클라이언트가 매 라운드트립마다:

- **관측**: `observation.state`(16,) → modality.json 레이아웃으로 분할
  (`left_arm` 0:7, `right_arm` 7:14, `left_gripper` 14:15, `right_gripper` 15:16,
  각 (1,1,D) float32), `observation.images.cam_X` → `video.cam_X` (1,1,H,W,3) uint8,
  task 문자열 → `language`. 비디오/state/언어 키 목록은 서버의
  `get_modality_config`에서 자동으로 가져옵니다.
- **액션**: 서버 응답 `{left_arm: (1,T,7), ...}`을 modality config 키 순서로 concat
  → (T,16) → 스텝당 (16,) 텐서로 액션 큐에 적재. un-normalize와 relative→absolute
  복원은 서버(Gr00tPolicy) 안에서 끝나므로 클라이언트에 나오는 값은 데이터셋과
  같은 물리 단위(관절 rad, 그리퍼 0~5.4)입니다.

비동기 버퍼 동작(`chunk_size_threshold`, 큐 드레인 시 마지막 명령 유지)은 기존
`RemoteSimPolicyClient`와 동일한 의미론을 따릅니다.

## 5. 트러블슈팅

| 증상 | 원인/조치 |
|---|---|
| `Cannot reach GR00T policy server` | 서버 미기동/터널 끊김. 서버 로그에서 "listening" 확인, `ssh -N -L 5555:...` 재연결 |
| 서버 로그에 shape assert | `--actions_per_chunk`이 변형 horizon과 불일치하거나 state 레이아웃 불일치 |
| 액션 값이 이상하게 크거나 작음 | 다른 변형의 체크포인트를 서빙 중인지 확인 (A/B 혼동) |
| 움직임이 덜컥거림 (hitch) | 왕복 지연 대비 버퍼 부족 — `--chunk_size_threshold` 올리거나 `--frequency` 낮추기 |
| 첫 응답이 매우 느림 | 정상 — 첫 추론에 CUDA 그래프/캐시 워밍업 포함. 이후 안정화 |
