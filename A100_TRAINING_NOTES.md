# A100 서버 학습 속도 노트 (GR00T N1.7 finetune)

2026-08-14, BCT 30Hz finetune 튜닝 과정에서 확인된 이 서버 고유의 특성과 권장 설정.

## 서버 하드웨어 특성

| 항목 | 내용 | 함의 |
|---|---|---|
| GPU | A100-SXM4-80GB × 8, **NVLink 없음** (`nvidia-smi topo -m` 전부 SYS, 0-1/2-3/4-5/6-7만 PIX 페어) | GPU 간 통신이 호스트 PCIe 경유로 매우 느림 (~1-2GB/s 실효) |
| NUMA | GPU 0-3 = NUMA0 (코어 0-47), GPU 4-7 = NUMA1 (코어 48-95) | NUMA 경계를 넘는 8-GPU 통신은 더 느림. **4-GPU 단일 NUMA가 최적** |
| 연산력 | A100 bf16 dense 312 TFLOPS (B200 ~2,250의 1/7) | B200 대비 스텝 시간 차이는 대부분 하드웨어 정직한 차이 |
| CPU/디스크 | 96코어, SSD, RAM 1TB | 데이터로더는 병목 아님 (단독 실측 배치당 0.14s) |

## 실측 결과 요약 (effective batch 256 동일)

| 구성 | s/it | GPU당 samples/s | 비고 |
|---|---|---|---|
| 8 GPU, DeepSpeed ZeRO-2, bs8×accum4 | 26 | 1.2 | 최악 — ZeRO-2는 accum 마이크로스텝마다 reduce |
| 8 GPU, ZeRO-2, bs32×accum1 | 9.2 | 3.5 | 통신(~6.5GB/step)이 스텝의 ~70% 차지 |
| 4 GPU(단일 NUMA), **DDP**, bs32×accum2 | 4.9 | 13.1 | 통신 절반 + backward와 중첩 |
| 4 GPU, DDP, **bs64×accum1** | 4.7 | 13.7 | VRAM 54/80GB. 전력 50-70W 위주 = 통신 대기 지배 |
| 4 GPU, DDP + **bf16 통신**, bs64×accum1, **4뷰** | **3.4** | **18.9** | **권장.** 뷰가 하나 더 늘었는데도 −28%. VRAM 64/80GB, 전력 230-250W 지속 |
| (참고) CK 레시피 재현: 4 GPU, DDP+bf16comm, bs16×accum1 (eff 64) | 2.2 | 7.3 | CK 원본 2.7s/it 대비 25%↑ — 소배치는 고정비 지배라 bs64 대비 효율 절반 |

핵심: 학습 파라미터가 1.62B(51%)라 ZeRO-2의 스텝당 통신량(reduce-scatter+allgather ~6.5GB)이 NVLink 없는 이 서버에서 치명적. DDP는 gradient allreduce만 하고 backward와 겹쳐서 노출 통신이 작음.

**bf16 통신 실측 (2026-08-16)**: gradient allreduce를 fp32→bf16으로 바꾼 것만으로 3뷰 4.7s/it → 4뷰 3.4s/it.
이미지 토큰이 33% 늘어 연산이 더 무거워졌는데도 28% 빨라졌으므로, 통신 절감분은 그보다 큼(3뷰 환산 ~3s/it 추정).
확증: GPU 전력이 50-70W(NCCL 스핀) 중심에서 **230-250W 지속**으로 바뀜 — 실연산 시간 비중이 크게 상승.
이 서버 단일 최대 효과 항목이므로 멀티 GPU 학습에는 항상 `--ddp-comm-bf16`을 붙일 것.

## 다음 학습 권장 커맨드

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \        # 또는 4,5,6,7 — 반드시 같은 NUMA 4장
NUM_GPUS=4 GLOBAL_BATCH_SIZE=256 GRAD_ACCUM=1 \   # 64/GPU, accum 없음
DATALOADER_NUM_WORKERS=16 DATALOADER_SEEK_MODE=approximate \
MASTER_PORT=29500 \                   # 동시 실행 시 런마다 다르게 (29500/29600)
bash examples/unitree_g1_dex1_bct/run_finetune_bct.sh <variant> --use-ddp --ddp-comm-bf16
```

- `--use-ddp`: 이번에 추가한 플래그 (`FinetuneConfig.use_ddp`). 이 서버에서는 항상 켤 것
- `--ddp-comm-bf16`: gradient allreduce를 fp32 대신 bf16으로 (트래픽 절반). **실측 −28%** (위 표).
  DeepSpeed 검증 레시피의 `communication_data_type: bf16`과 동일한 수치 특성 — fp32 마스터/옵티마이저는
  그대로라 학습 품질 영향 없음 (loss 곡선 정상 확인). 활성화되면 랭크마다
  "DDP gradient communication: bf16 compress hook enabled" 로그가 찍힘
- `DATALOADER_SEEK_MODE=approximate`: torchcodec 디코더 오픈 시 전체 스캔 생략 (fetch 74→7ms). 고정 fps + 짧은 GOP 영상에서만 사용 (프레임 동일성 검증됨)
- 하이퍼파라미터는 검증 레시피 유지: effective 256, lr 1e-4, warmup 0.05, state_dropout 0.2
- epoch 가이드: 선례 27~43 epochs (15Hz BCT 31, brainco 43). eval 2,500스텝마다 val MSE 확인해 2-3회 연속 정체 시 조기 종료, 최소 체크포인트 선택
- **두 학습 동시 실행 가능**: GPU 0-3 + GPU 4-7 (포트만 분리). 워커 합계 128 > 96코어라 5% 내외 상호 간섭 있음

## 진단 요령

- **GPU util%는 믿지 말 것** — NCCL 스핀 대기도 100%로 찍힘. `nvidia-smi dmon -s pu`로 **전력(W)과 mem%**를 볼 것: 지속 50-75W = 통신/대기, 150W+ 버스트 = 실연산. 건강한 상태의 기준선은 **230-250W 지속**(bf16 통신 + bs64 4뷰에서 관측)
- bf16 여부: `bf16=True` (HF autocast 혼합정밀도, fp32 마스터 웨이트). 로드 시 "Flash Attention 2 only supports fp16/bf16" 경고는 정보성 — 실제 연산은 bf16+FA2
- 데이터로더 의심 시: 파이프라인 단독 벤치 (`scratchpad/dl_bench.py` 참고 — ShardedMixture + DataLoader만 돌려 배치 간격 측정)

## 로그 확인

```bash
tail -f datasets/train_launch.log                      # 전체 로그 (tqdm 진행바 포함)
tail -f outputs/<EXP_NAME>/train.log                   # 동일 내용 (tee 사본)
tr '\r' '\n' < datasets/train_launch.log | grep -E "'loss'|eval_loss" | tail   # loss 추이
ls outputs/<EXP_NAME>/<EXP_NAME>/                      # checkpoint-*, experiment_cfg, processor
# wandb offline 기록은 레포 루트 wandb/offline-run-* 에 쌓임
# 나중에 대시보드로 보려면: wandb login 후 wandb sync wandb/offline-run-<날짜>-<id>
```
