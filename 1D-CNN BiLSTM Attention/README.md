# Non-Invasive Cardiac Output Estimation

PPG·ECG 신호와 환자 정보(성별, 나이, 신장, 체중)를 입력받아 심박출량(CO, L/min)을 비침습적으로 추정하는 딥러닝 모델 실험 저장소입니다.  
현재는 **1D CNN + BiLSTM + Self-Attention** 모델이 포함되어 있으며, 이후 다른 모델도 순차적으로 추가될 예정입니다.

---

## 모델 목록

| 모델 | 디렉토리 | 설명 |
|------|----------|------|
| 1D CNN + BiLSTM + Self-Attention | `models/cnn_bilstm_attention.py` | PPG/ECG 각각 CNN→BiLSTM→Self-Attention, 환자 정보 MLP, FusionAttention으로 CO 예측 |
| _(추가 예정)_ | `models/` | — |

---

## 모델 구조 (1D CNN + BiLSTM + Self-Attention)

```
PPG (B, T, 1)  ──► CNN1D ──► BiLSTM ──► SignalAttention ──► (B, H) ─┐
ECG (B, T, 1)  ──► CNN1D ──► BiLSTM ──► SignalAttention ──► (B, H) ─┼─► FusionAttention ──► Linear ──► CO (B,)
info (B, 4)    ──────────────────────► MLP ─────────────► (B, H) ─┘
```

| 컴포넌트 | 역할 |
|----------|------|
| `CNN1D` | 4-block 1D Conv로 시계열에서 로컬 특징 추출 |
| `BiLSTM` | 양방향 LSTM으로 장기 시간 의존성 학습 |
| `SignalAttention` | 로컬 윈도우 기반 셀프 어텐션으로 신호별 컨텍스트 벡터 생성 |
| `MLP` | 환자 정보(Sex, Age, Ht, Wt) 인코딩 |
| `FusionAttention` | PPG / ECG / 환자 정보 3개 모달리티 간 크로스 어텐션 융합 |

---

## 프로젝트 구조

```
.
├── train.py                      # 학습 진입점
├── evaluate.py                   # 평가 및 시각화
├── config.py                     # 하이퍼파라미터 설정
├── requirements.txt
├── data/
│   └── dataset.py                # PPGECGDataset, 전처리, DataLoader 빌더
├── models/
│   └── cnn_bilstm_attention.py   # 모델 컴포넌트 및 SignalProcessingModel
└── utils/
    ├── metrics.py                # MSE / RMSE / MAE / R2 / MAPE 계산
    ├── early_stopping.py         # EarlyStopping
    └── bland_altman.py           # Bland-Altman 플롯
```

---

## 데이터 형식

`data_dir`(기본값: `.`) 아래에 아래 3개 파일이 있어야 합니다.

| 파일 | 설명 |
|------|------|
| `Train.pkl` | 학습 데이터 |
| `Val_df_final.pkl` | 검증 데이터 |
| `Test.pkl` | 테스트 데이터 |

각 DataFrame은 다음 컬럼을 포함해야 합니다.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `ppg` | array-like | PPG 시계열 신호 |
| `ecg` | array-like | ECG 시계열 신호 |
| `Sex` | float | 성별 |
| `Age` | float | 나이 |
| `Ht` | float | 신장 (cm) |
| `Wt` | float | 체중 (kg) |
| `co` | float | 심박출량 레이블 (L/min) |
| `pid` | int | 환자 ID |

---

## 설치

```bash
pip install -r requirements.txt
```

GPU 사용 시 PyTorch 공식 사이트에서 CUDA 버전에 맞는 torch를 별도 설치하세요.  
https://pytorch.org/get-started/locally/

---

## 실행 방법

### 학습

```bash
python train.py
```

하이퍼파라미터는 `config.py`의 `CONFIG` 딕셔너리에서 수정합니다.

```python
# config.py
CONFIG = {
    'data_dir':   '.',        # pkl 파일 위치
    'save_dir':   './results', # 체크포인트 및 결과 저장 경로
    'epochs':     200,
    'learning_rate': 8e-4,
    ...
}
```

학습이 완료되면 `results/` 디렉토리에 다음 파일이 생성됩니다.

```
results/
├── best_model.pth           # 최적 체크포인트
├── evaluation_results.csv   # 샘플별 예측 결과
├── patient_summary.csv      # 환자별 지표 요약
├── scatter_plot.png         # 실제 vs 예측 산점도
├── error_distribution.png   # 예측 오차 분포
└── bland_altman_plot.png    # Bland-Altman 플롯
```

### 평가만 별도 실행

저장된 체크포인트로 평가만 수행할 경우 `evaluate.py`를 직접 import해서 사용합니다.

```python
import torch
from models.cnn_bilstm_attention import SignalProcessingModel
from evaluate import evaluate_and_visualize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SignalProcessingModel().to(device)
model.load_state_dict(torch.load('results/best_model.pth', map_location=device))

results = evaluate_and_visualize(model, test_loader, device, save_path='./results')
```

---

## 새 모델 추가 방법

1. `models/` 아래에 새 파일 추가 (예: `models/transformer.py`)
2. 모델 클래스가 `forward(ppg, ecg, patient_info) -> (B,)` 인터페이스를 따르면 `train.py`의 모델 초기화 부분만 교체하면 됩니다.

```python
# train.py main() 내부 모델 초기화 부분만 변경
from models.transformer import TransformerModel   # 새 모델 import

model = TransformerModel(...).to(device)
```
