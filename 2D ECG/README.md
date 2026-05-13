# A Multimodal Fusion Method for Non-invasive CO Prediction in ICU

> **🏆 삼성융합의과학원장상 최우수상** — 2025 디지털 바이오 인력양성 학술대회

ICU 환자의 ECG와 PPG 신호, 환자 정보를 결합하여 심박출량(Cardiac Output, CO)을 비침습적으로 예측하는 멀티모달 딥러닝 모델입니다.  
학위논문의 최종 아키텍처 도출 과정에서 시도한 접근법으로, 학술대회 발표 자료를 기반으로 정리하였습니다.

---

## Data Preprocessing

ECG 신호는 학습 전에 **Stockwell Transform (S-Transform)** 을 적용하여 2D 시간-주파수 표현으로 변환합니다.  
S-Transform은 주파수에 따라 시간-주파수 해상도가 가변적으로 조정되어, 급격한 파형 변화와 미세한 리듬 변화를 동시에 포착할 수 있습니다.

전처리 과정은 [`S-Transform_test.ipynb`](./S-Transform_test.ipynb)에 구현되어 있으며, 학습 전 해당 노트북으로 ECG를 미리 변환한 뒤 pickle 파일로 저장합니다.

**변환 예시 (1D ECG → 2D S-Transform):**

![S-Transform Output](./output.png)

---

## Model Architecture

세 가지 독립 브랜치에서 특징을 추출한 뒤 Feature-level Concatenation으로 결합하고, MLP로 최종 CO를 예측합니다.

| Branch | 입력 | 모델 | 출력 차원 |
|--------|------|------|-----------|
| PPG | Raw 1D signal `(B, 2500, 1)` | 1D CNN → BiLSTM | 256 |
| ECG | S-Transform image `(B, 1, 599, 2500)` | 2D ResNet + Spectral Attention + Freq-Time Fusion | 256 |
| Patient Info | Demographics `(B, 4)` | MLP | 64 |
| Fusion | Concatenated `(B, 576)` | MLP → CO | 1 |

![Architecture](./Architecture.png)

**ECG branch 상세:**  
`2D ResNet Encoder` → `Spectral Attention (주파수/시간/채널)` → `Frequency-Time Fusion` → `Global Average Pool`

**PPG branch 상세:**  
`1D CNN (4-stage)` → `BiLSTM (2-layer, bidirectional)` → last hidden state

---

## Results

| Metric | Value |
|--------|-------|
| RMSE | 0.832 L/min |
| MAE | 0.780 L/min |

**Bland-Altman Plot (Test Set):**

![Bland-Altman](./bland-altman.png)

- Mean bias: ~1.22%
- 95% Limits of Agreement: −31.42% ~ +33.62%
- Within limits: 95.2%

---

## Usage

### 1. 데이터 전처리

```bash
jupyter notebook S-Transform_test.ipynb
```

ECG 신호에 S-Transform을 적용하여 `train.pkl`, `val.pkl`, `test.pkl` 파일로 저장합니다.  
각 pkl 파일은 다음 컬럼을 포함해야 합니다:

| 컬럼 | 형태 | 설명 |
|------|------|------|
| `ppg` | 1D array (~2500) | Raw PPG signal |
| `ecg_s_transform` | `(599, 2500)` array | S-Transform 변환된 ECG |
| `Sex`, `Age`, `Ht`, `Wt` | scalar | 환자 인구통계 정보 |
| `co` | scalar | Cardiac Output (L/min) |

### 2. 모델 shape 검증

```bash
python multimodal_co_prediction.py
```

데이터 없이 dummy tensor로 입출력 shape 및 파라미터 수를 확인할 수 있습니다.

### 3. 학습 (데이터 준비 후)

```python
from multimodal_co_prediction import MultimodalCOPredictor, build_dataloaders

train_loader, val_loader, test_loader = build_dataloaders(data_dir='./')
model = MultimodalCOPredictor()
```

---

## Requirements

```
torch
numpy
pandas
scikit-learn
stockwell
```

---

## Stack

- **Framework**: PyTorch
- **Signal processing**: Stockwell (S-Transform)
- **OS**: Linux
