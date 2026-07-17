# gr00t-client — Isaac-GR00T N1.7 모델의 Isaac Sim 원격 평가

`unitree_lerobot`의 remote deploy 세팅을 **Isaac-GR00T N1.7 체크포인트**에 연결합니다.
(기존 `groot-server` 경로는 LeRobot의 GR00T N1.5 전용이라 N1.7을 로드할 수 없음)

```
[데스크톱]  Isaac Sim (sim_main.py)   ← 기존 그대로
[데스크톱]  eval_g1_sim_groot.py      ← 이 패키지
                │  TCP :5555 (SSH 터널)
[GPU 서버]  Docker: gr00t-inference   ← Isaac-GR00T 스톡 서버
```

**요구사항**
- GPU 서버: NVIDIA 드라이버 ≥ 570, Ampere 이상 GPU(RTX A5000 확인됨), Docker + NVIDIA Container Toolkit
- 데스크톱: 기존 `unitree_sim_env` / `unitree_lerobot` conda env 그대로

---

## 실행 순서 (위에서부터 그대로 따라하기)

### 0. 최초 1회 준비

**GPU 서버** — 이미지 빌드:

```bash
git clone https://github.com/RooibosT/Isaac-GR00T && cd Isaac-GR00T
docker build -f gr00t-client/docker/Dockerfile.inference -t gr00t-inference .
```

**데스크톱** — 클라이언트 의존성 3개 추가 + 이 디렉토리 복사:

```bash
conda activate unitree_lerobot
pip install pyzmq msgpack msgpack-numpy
# gr00t-client/ 디렉토리를 데스크톱에 복사 (repo clone 또는 scp — gr00t 패키지 설치 불필요)
```

### 1. GPU 서버: 정책 서버 실행

```bash
docker run --rm -it --gpus all --ipc=host -p 5555:5555 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/gr00t_checkpoints:/checkpoints \
  -e HF_TOKEN=<hf_token> \
  -e MODEL_PATH=RooibosT/gr00t-n1.7-g1-dex1-A \
  gr00t-inference
```

시작 로그에서 순서대로 확인: `GPU: NVIDIA RTX A5000 (sm86)` → `flash-attn ... kernel OK`
→ 체크포인트 다운로드(최초 1회) → **`Server is ready and listening`**.

(선택) 같은 서버에서 왕복 검증:

```bash
docker exec -it $(docker ps -q -f ancestor=gr00t-inference) \
  python gr00t-client/test_roundtrip.py --port 5555 --horizon 16
# "ROUNDTRIP TEST PASSED" + 추론 latency 확인
```

### 2. 데스크톱 터미널 1: SSH 터널 (열어둔 채 유지)

```bash
ssh -N -L 5555:localhost:5555 <user>@<gpu_server_ip>
```

### 3. 데스크톱 터미널 2: Isaac Sim (`unitree_sim_env`)

```bash
conda activate unitree_sim_env
cd /path/to/unitree_sim_isaaclab
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint \
  --enable_dex1_dds --robot_type g129 --keyboard_termination_key r
# 창을 한 번 클릭해 포커스를 주고 그대로 둠. 'r' = 씬 리셋
```

### 4. 데스크톱 터미널 3: deploy 클라이언트 (`unitree_lerobot`)

```bash
conda activate unitree_lerobot
cd /path/to/gr00t-client
python eval_g1_sim_groot.py \
  --policy_server_address=127.0.0.1:5555 \
  --repo_id=RooibosT/g1_pick_redblock_dex1_sim_merged_107demo \
  --actions_per_chunk=16 \
  --frequency=30 --arm=G1_29 --ee=dex1
# 프롬프트에서 's' + Enter → 팔이 초기 자세로 이동 후 평가 시작
```

### 변형 교체

GPU 서버에서 Ctrl+C 후 `MODEL_PATH`만 바꿔 1번을 다시 실행하고,
클라이언트(4번)의 `--actions_per_chunk`를 맞춥니다:

| `MODEL_PATH` | 팔 표현 | `--actions_per_chunk` |
|---|---|---|
| `RooibosT/gr00t-n1.7-g1-dex1-A`   | RELATIVE joint | **16** |
| `RooibosT/gr00t-n1.7-g1-dex1-B`   | ABSOLUTE joint | **16** |
| `RooibosT/gr00t-n1.7-g1-dex1-A40` | RELATIVE joint | **40** |
| `RooibosT/gr00t-n1.7-g1-dex1-B40` | ABSOLUTE joint | **40** |

재실행할 때: 서버/터널은 그대로 두고, Isaac Sim에서 `r`로 씬 리셋 → 터미널 3만 재시작.

---

## 트러블슈팅

| 증상 | 원인/조치 |
|---|---|
| `Cannot reach GR00T policy server` | 서버 미기동 또는 터널 끊김 — 서버 로그의 "listening" 확인, 터미널 1 재연결 |
| 서버 로그에 shape assert | `--actions_per_chunk`이 변형 horizon과 불일치 (표 참고) |
| 액션 값이 이상하게 크거나 작음 | 의도한 변형을 서빙 중인지 `MODEL_PATH` 확인 |
| 움직임이 덜컥거림 (hitch) | 왕복 지연 대비 버퍼 부족 — `--chunk_size_threshold 0.6~0.7` 또는 40-horizon 변형 사용 |
| 첫 응답이 매우 느림 | 정상 — 첫 추론은 워밍업 포함. 이후 안정화 |

---

## 부록 A — Docker 없이 venv로 서빙

GPU 서버에서 Docker를 쓰지 않을 때의 대체 방법입니다 (`serve_variant.sh`가
컨테이너 entrypoint와 같은 일 — 체크포인트 다운로드 + 스톡 서버 실행 — 을 합니다.
Docker 방식에서는 필요 없습니다):

```bash
git clone https://github.com/RooibosT/Isaac-GR00T && cd Isaac-GR00T
git lfs install && git lfs pull   # scripts/deployment/*/wheels/*.whl 이 LFS 파일
uv sync && source .venv/bin/activate
hf auth login                     # private 체크포인트 접근용
bash gr00t-client/serve_variant.sh A 5555 0   # <변형> [포트] [GPU]
```

## 부록 B — 파일 설명

| 파일 | 실행 위치 | 역할 |
|---|---|---|
| `docker/Dockerfile.inference` | GPU 서버 | 추론 전용 이미지 (CUDA 12.8 base, lockfile 설치) |
| `docker/entrypoint.sh` | (컨테이너) | GPU/flash-attn 체크 → Hub 체크포인트 다운로드 → 서버 실행 |
| `serve_variant.sh` | GPU 서버 | 부록 A(venv 방식) 전용 — entrypoint의 대체재 |
| `eval_g1_sim_groot.py` | 데스크톱 | 평가 엔트리포인트 (`eval_g1_sim_remote.py`의 GR00T판) |
| `gr00t_sim_policy_client.py` | 데스크톱 | 관측/액션 형식 변환 + 비동기 액션 버퍼 |
| `gr00t_transport.py` | 데스크톱 | gr00t 패키지 없이 동작하는 ZMQ/msgpack 클라이언트 |
| `test_roundtrip.py` | GPU 서버 | 가짜 관측으로 서버 왕복 검증 |

## 부록 C — 동작 원리 (형식 변환)

- **관측**: `observation.state`(16,)를 `left_arm` 0:7 / `right_arm` 7:14 /
  `left_gripper` 14:15 / `right_gripper` 15:16 으로 분할해 각 (1,1,D) float32로,
  `observation.images.cam_X`는 `video.cam_X` (1,1,H,W,3) uint8로, task 문자열은
  `language`로 변환. 키 목록은 서버의 `get_modality_config`에서 자동 취득.
- **액션**: 서버 응답 `{left_arm: (1,T,7), ...}`을 키 순서로 concat → (T,16) →
  스텝당 (16,) 텐서로 액션 큐에 적재. un-normalize와 relative→absolute 복원은
  서버(Gr00tPolicy) 내부에서 끝나므로 클라이언트 값은 물리 단위(rad, 그리퍼 0~5.4).
- 비동기 버퍼(`chunk_size_threshold`, 큐 드레인 시 마지막 명령 유지)는 기존
  `RemoteSimPolicyClient`와 동일한 의미론.
