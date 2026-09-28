# Week 4 - Accuracy / Latency Pareto 실습

## 실행 환경 (실습 지침서와 다른 점)

지침서는 Jetson Orin Nano 기준이지만, 이 폴더는 현재 워크스테이션에서 그대로 재현한 결과다.

| 항목 | 지침서 (Jetson) | 여기 (실측) |
|---|---|---|
| Device | Jetson Orin Nano / L4T R36.4.7 | x86_64 워크스테이션 |
| GPU | Orin | NVIDIA GeForce RTX 3090 (4장, 0번만 사용) |
| Python | 3.10 | 3.11.9 |
| PyTorch | 2.8.0 (aarch64 wheel) | 2.4.0 (CUDA 11.8) |
| torchvision | 0.23.0 | 0.19.0 |
| NumPy | 1.26.4 | 1.26.4 (동일) |

Jetson wheel / cuSPARSELt 설치(Part C 9~10절)는 aarch64 전용이라 생략했다.
코드 자체는 지침서와 동일하며, Colab 전용 `files.download()`만 제거했다.

## 파일

```
week4_performance/
├─ train_tinymlp.py            # Part A: TinyMLP 학습 → weights/TinyMLP.pth
├─ check_gpu.py                # Part D: GPU/CUDA + cuBLAS 동작 확인
├─ latency_toy.py              # Part E: ToyCNN latency 측정 원리
├─ pareto_check.py             # Part F: Accuracy/Latency Pareto 분석
├─ weights/TinyMLP.pth         # 학습된 가중치 (state_dict)
├─ pareto_results.csv          # 결과 표
└─ pareto_accuracy_latency.png # Pareto Front 그래프
```

## 실행 순서

```bash
python3 train_tinymlp.py   # 학습 (Colab 대신 로컬에서 수행)
python3 check_gpu.py
python3 latency_toy.py
python3 pareto_check.py
```

## 실측 결과

- `check_gpu.py` : CUDA available True, RTX 3090, 100x100 matmul SUCCESS
- `latency_toy.py` (ToyCNN, single image) : Mean 0.1371 ms / Median 0.1371 ms / Std 0.0060 ms
- `train_tinymlp.py` : 5 epoch, Test Accuracy 92.54 % (2회 실행 92.10 / 92.54 % 로 재현)
- `pareto_check.py` (TinyMLP) : Accuracy 92.54 %, Latency 0.0760 ms

| Model | Accuracy | Latency (ms) | Source | Pareto |
|---|---:|---:|---|---|
| TinyMLP | 92.54 | 0.0760 | Measured | True |
| EfficientCNN | 96.80 | 0.42 | Demo | True |
| SmallCNN | 97.80 | 0.62 | Demo | True |
| MediumCNN | 97.20 | 0.95 | Demo | False |
| LargeCNN | 99.00 | 1.65 | Demo | True |

> SmallCNN/MediumCNN/LargeCNN/EfficientCNN 값은 지침서와 동일한 **가상 예시값**이다.

## 해석

- TinyMLP는 가장 빠르지만(0.076 ms) 정확도가 가장 낮다. 그래도 "그보다 빠르면서 더 정확한" 모델이
  없으므로 Pareto Front의 최좌하단 끝점으로 남는다.
- MediumCNN은 SmallCNN보다 느리고(0.95 vs 0.62) 덜 정확해서(97.2 vs 97.8) 유일한 dominated 모델이다.
- 나머지 네 모델은 서로 trade-off 관계라 모두 Front 위에 있다. 즉 "지연시간 예산"이 정해져야
  하나를 고를 수 있다.
- TinyMLP latency(0.076 ms)가 ToyCNN(0.137 ms)보다 짧은 이유는 연산량 차이라기보다
  커널 실행 횟수 차이가 크다. 이 규모에서는 GPU 연산 시간보다 kernel launch overhead가 지배적이다.
  → 다음 주 FLOPs/MACs 비교에서 "FLOPs가 적다 = 빠르다"가 아닌 이유의 근거가 된다.

---

# 확장 실습: ResNet / DenseNet / EfficientNet 실제 학습

지침서의 demo 값(SmallCNN / MediumCNN / LargeCNN / EfficientCNN)을 **전부 실측값으로 교체**했다.

## 추가 파일

```
models.py              # TinyMLP + ResNet18 / DenseNet121 / EfficientNetB0 (MNIST용 수정)
train_models.py        # 세 모델 학습 → weights/*.pth, train_log.json
pareto_full.py         # 네 모델 전부 실측 → pareto_full_results.csv / pareto_full.png
```

## ImageNet 모델을 MNIST에 맞춘 방법

torchvision 구현은 3채널 224x224 / 1000 class 기준이라 두 곳만 고쳤다.

1. 첫 conv의 `in_channels` 3 → 1
2. 분류기 `num_classes` 1000 → 10

추가로 ResNet18 / DenseNet121은 28x28 입력에 stem의 stride-2 conv + maxpool이 너무 공격적이라
(28 → 7로 바로 줄어듦) CIFAR 관례대로 **3x3 stride 1 conv + maxpool 제거**로 바꿨다.
EfficientNetB0은 구조상 stride 설정을 그대로 두었다.

학습 조건은 TinyMLP와 동일하게 맞췄다 (Adam 1e-3, batch 128, 5 epoch, 같은 전처리).
조건이 같아야 Accuracy 차이를 모델 구조 차이로 읽을 수 있다.

## 실행

```bash
python3 train_models.py                      # 세 모델 순차 학습
python3 train_models.py --models ResNet18    # 특정 모델만
python3 pareto_full.py                       # 네 모델 실측 + Pareto
```

GPU 여러 장이면 병렬로 돌릴 수 있다(이번에 실제로 그렇게 했다).

```bash
CUDA_VISIBLE_DEVICES=1 LOG_PATH=train_log_effnet.json \
    python3 train_models.py --models EfficientNetB0 &
```

## 결과 (전부 실측)

| Model | Params | Accuracy (%) | Latency (ms) | Std | Train time | Pareto |
|---|---:|---:|---:|---:|---:|---|
| TinyMLP | 25,450 | 92.54 | 0.0755 | 0.0049 | - | True |
| ResNet18 | 11,172,810 | 99.22 | 1.8617 | 0.0536 | 53.9 s | True |
| DenseNet121 | 6,955,274 | 99.37 | 11.6806 | 0.1668 | 190.5 s | True |
| EfficientNetB0 | 4,019,782 | 97.72 | 5.8479 | 0.2912 | 111.7 s | **False** |

그래프: `pareto_full.png`

## 해석

- **Pareto Front = TinyMLP → ResNet18 → DenseNet121.** 왼쪽 끝은 압도적으로 빠르고,
  오른쪽 끝은 가장 정확하며, ResNet18이 그 사이의 실질적인 균형점이다.
- **EfficientNetB0만 dominated.** ResNet18보다 정확도가 1.5%p 낮으면서 3.1배 느리다.
  파라미터는 가장 적은데(4.02M vs 11.17M) 결과는 가장 나쁜, 교과서적인 반례다.
- **왜 파라미터가 적은데 느린가.** EfficientNet은 depthwise separable conv와 SE 블록으로
  이론 연산량(FLOPs)을 줄인 구조다. 그런데 이 연산들은 채널당 작은 커널을 수없이 띄우므로
  **kernel launch 횟수와 메모리 접근**이 늘어난다. batch=1 추론에서는 GPU 연산 자체보다
  이 오버헤드가 지배적이라, FLOPs 이득이 실제 latency로 이어지지 않는다.
  DenseNet121이 파라미터가 ResNet18보다 적은데도 6.3배 느린 것도 같은 이유
  (dense connectivity → concat과 레이어 수가 많음).
- 즉 지침서 Part H의 질문 "FLOPs가 적은 모델은 Jetson에서도 반드시 latency가 짧을까?"에 대한
  답은 **이 표 안에 실측으로 이미 들어 있다. 아니다.**
- 주의: 이 latency는 RTX 3090 / batch 1 기준이다. Jetson에서는 절대값이 커지고,
  GPU가 약할수록 연산량 비중이 커지므로 모델 간 **순서 자체가 바뀔 수도 있다.**
  같은 모델이라도 하드웨어가 바뀌면 Pareto Front를 다시 그려야 한다는 뜻이다.
