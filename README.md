# Multimodal deep learning for non-invasive cardiac output estimation in ICU patients

**Samsung Medical Center collaborative research · Master's thesis (2023–2025) · IRB/DRB approved**

---

## Overview

Cardiac output matters in the ICU because it tells you how well the heart is actually perfusing the body. The current reference standard, intermittent thermodilution via a pulmonary artery catheter (Swan-Ganz), works, but it requires threading a catheter into the pulmonary artery, which carries infection risk, can trigger arrhythmias on insertion, and delivers point-in-time measurements rather than a continuous trend.

This research tests whether ECG, PPG, and basic demographic data already collected on ICU patients contain enough information to estimate CO without a catheter. On a cohort of 642 ICU patients across two institutions, the proposed multimodal model achieves **PE = 31.24%** on internal test data and **PE = 39%** on an external validation set collected with different monitoring hardware. Both fall within the clinical acceptability range for non-invasive CO methods.

![Hemodynamic relationships](figures/hemodynamic.png)

> CO = HR × SV. Preload, afterload, and contractility converge on stroke volume, which combined with heart rate gives cardiac output. Combined with systemic vascular resistance, this sets mean arterial pressure.

---

## Clinical background

The reference standard is thermodilution via pulmonary artery catheterization (Stewart-Hamilton equation). The clinical interchangeability criterion, PE < 30%, comes from Critchley & Critchley (1999), who derived it from thermodilution's own precision. Thermodilution carries roughly ±20% precision; two methods each at 20% precision produce a combined PE of about 28.3%, rounded to 30% as the interchangeability cutoff.

Existing non-invasive techniques fall short of that:

![CO measurement methods comparison](figures/co_methods_comparison.png)

| Method | PE (%) | Core limitation |
|--------|--------|-----------------|
| Partial CO₂ rebreathing | 40% | Requires mechanical ventilation |
| Bioimpedance (TEB) | 42% | Sensitive to electrical noise and fluid shifts |
| Pulse contour (niPCA) | 45% | Degrades during peripheral vasoconstriction |
| Pulse wave transit time (PWTT) | 62% | High variability; needs frequent recalibration |

Several of these also have patient-selection constraints that limit where they can be used. The study target was set at **PE < 42%**, matching bioimpedance accuracy without its restriction to ventilated patients.

---

## Dataset and cohort

This research used data collected under IRB/DRB approval in collaboration with **Samsung Medical Center (SMC)** and Seoul National University Bundang Hospital (SNUBH).

**Raw data are not publicly available** due to patient privacy regulations and institutional data agreements. The schema below documents the expected structure for anyone replicating this with their own institutional dataset.

### Cohort construction

3,733 ICU patients were screened across both institutions. Patients on ECMO or LVAD support were excluded because mechanical circulatory assist devices distort ECG and PPG morphology in ways that break the physiological assumptions the model depends on. Cases with missing CO labels or signal dropout were also removed, leaving 642 patients.

| Split | Institution | N | Role |
|-------|-------------|---|------|
| Train / val / test (70/15/15) | Samsung Medical Center (SMC) | 501 | Model development |
| External validation | SNUBH | 141 | Generalizability test |

All splits are at the patient level. Every segment from a given patient lands in exactly one split, which prevents leakage from the strong within-patient correlation that shows up in any sliding-window dataset.

### Signal structure

| Field | Type | Description |
|-------|------|-------------|
| `ppg` | array (2500,) | PPG time-series, 20 s at 125 Hz |
| `ecg` | array (2500,) | ECG time-series, 20 s at 125 Hz |
| `Sex` | float | Biological sex |
| `Age` | float | Age in years |
| `Ht` | float | Height (cm) |
| `Wt` | float | Weight (kg) |
| `co` | float | Cardiac output (L/min), thermodilution reference |
| `pid` | int | Anonymized patient ID |

Signals were segmented with a non-overlapping 20-second window at 125 Hz (2,500 samples per segment). Non-overlapping windows preserve temporal continuity and avoid artificial autocorrelation between adjacent segments from the same patient.

---

## Model architecture

Three branches extract features from the input in parallel — time-domain, frequency-domain, and patient demographics — and their outputs are fused through a cross-attention layer conditioned on the patient context.

![Full multimodal architecture](figures/multimodal_architecture.png)

### Time domain module

Raw ECG and PPG feed a U-Net style autoencoder. Three stages of 1D convolution and pooling compress the 20-second signal to a 256-dimensional latent vector. Skip connections between encoder and decoder preserve morphological features at multiple scales while the bottleneck learns a compact summary. This module trains jointly with a reconstruction objective that prevents the encoder from discarding information that is not immediately useful for regression but still carries physiological content.

![Autoencoder architecture](figures/autoencoder.png)

### Frequency domain module

The same raw signals pass independently through an FFT, then through a 1D CNN with batch normalization, ReLU, and max pooling. Heart rate variability, respiratory modulation of PPG amplitude, and low-frequency power shifts tied to sympathetic tone are more accessible from the spectral representation than from the raw waveform.

### Patient information module

A two-layer MLP encodes sex, age, height, and weight. Body size and composition have a real effect on stroke volume, and a model that ignores demographics will systematically mis-estimate CO for patients at the distribution extremes. This branch makes that correction data-driven rather than rule-based.

### Fusion module: dynamic cross-attention

Time-domain, frequency-domain, and patient-information features are merged through a cross-attention layer where the patient information vector is the query. Attention weights are computed per-inference, so the model can shift weight toward ECG when the PPG is degraded by poor peripheral perfusion, rather than applying a fixed mixture for every patient.

![Cross-attention fusion](figures/cross_attention.png)

---

## Training: multi-task learning

Two objectives run simultaneously during training.

**Task 1 — signal reconstruction (L_recon):** The autoencoder reconstructs the original ECG and PPG from the latent representation, measured by Percent Root-mean-square Difference (PRD) averaged across both signals. This keeps the encoder from collapsing to features that only serve the regression head.

**Task 2 — CO prediction (L_pred):** The fused features map to a scalar CO estimate. Huber loss instead of MSE: ICU CO measurements include genuine physiological extremes, and MSE would let those dominate training. Huber limits the gradient contribution of large errors without ignoring them.

![Loss functions](figures/loss_function.png)

```
L_total = 0.8 × L_pred  +  0.2 × L_recon
```

Prediction drives training. Reconstruction keeps the encoder quality up.

### Hyperparameters

![Hyperparameter table](figures/hyperparameters.png)

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| LR scheduler | ReduceLROnPlateau |
| Max epochs | 300 |
| Early stopping patience | 30 |
| Latent dimension | 256 |

---

## Results

### Model comparison

![Performance comparison](figures/performance_table.png)

| Model | Input modalities | RMSE | MAE | R² | Mean bias (%) | PE (%) |
|-------|-----------------|------|-----|----|---------------|--------|
| 1D CNN-GRU | Time ECG, PPG, demographics | 1.325 | 1.12 | 0.234 | 4.2 | 42.1 |
| 1D CNN-LSTM + self-attention | Time PPG, demographics | 1.18 | 1.03 | 0.35 | 2.2 | 38.46 |
| XGBoost (engineered features) | ECG, PPG, demographics | 0.93 | 0.86 | 0.45 | 1.38 | 35.2 |
| 1D CNN-LSTM + self-attention (STFT) | STFT-ECG, time PPG, demographics | 0.91 | 0.89 | 0.36 | 1.7 | 35.5 |
| 1D CNN-2D ResNet + cross-attention | S-Transform ECG, time PPG, demographics | 0.832 | 0.78 | 0.41 | 0.7 | 32.62 |
| **Proposed multimodal** | **ECG+PPG (time & FFT), demographics** | **0.79** | **0.67** | **0.59** | **0.01** | **31.24** |

### Bland-Altman analysis

RMSE alone does not establish clinical interchangeability. A method can have low RMSE while still showing systematic bias at specific CO ranges that would matter in practice. Bland-Altman analysis is the standard approach for assessing whether two measurement methods agree well enough to be used interchangeably.

![Bland-Altman plot](figures/bland_altman.png)

| Metric | Value |
|--------|-------|
| Mean bias | 0.01% |
| +1.96 SD | +31.25% |
| -1.96 SD | -31.23% |
| PE | 31.24% |

Mean bias near zero means the model is not systematically over- or underestimating CO at any part of the measurement range. The ±1.96 SD limits fall within the clinical acceptability band.

### External validation

On the SNUBH dataset, recorded with HemoSphere equipment not seen during training, the model achieved **PE = 39%**. It stays within the clinical acceptability range and held up across a different recording platform, which is some evidence that the model is responding to physiological signal properties rather than recording-specific artifacts from a single institution.

---

## Limitations

The cohort is entirely ICU patients on invasive hemodynamic monitoring. How the model performs in general wards, the emergency department, or perioperative settings is unknown.

Arrhythmia segments were excluded from the sliding-window extraction. The model's behavior on atrial fibrillation and other dysrhythmias was not tested.

All data are retrospective. Prospective validation is needed before any clinical use.

---

## Future work

- Arrhythmia segment evaluation with rhythm-stratified reporting
- Additional modalities: medication history, surgical context
- Prospective multi-site validation in ward and emergency settings
- Real-time inference feasibility on edge hardware

---

## Repository structure

```
Paper_CO/
├── data/
│   └── dataset.py                    # Shared data loading: PPGECGDataset, normalization, DataLoader
├── figures/                          # Architecture diagrams and result plots
├── 1D-CNN BiLSTM Attention/          # Baseline: 1D CNN + BiLSTM + self-attention
│   ├── models/
│   │   └── cnn_bilstm_attention.py
│   ├── train.py
│   ├── evaluate.py
│   └── config.py
├── 2D ECG/                           # 2D spectrogram-based model experiments
└── MTL/                              # Multi-task learning implementation iterations
```

`data/dataset.py` is shared across all model variants. It loads `.pkl` files, applies per-segment min-max normalization to [-1, 1], standardizes patient demographics with StandardScaler, and builds train/val/test DataLoaders with patient-level splits enforced.

**Data are not included in this repository.** The schema above documents the expected format for replication with institutional data.

---

## Acknowledgements

This research was a collaboration with **Samsung Medical Center (SMC)**, which provided the clinical data, IRB/DRB approval, and institutional research support. External validation data came from **Seoul National University Bundang Hospital (SNUBH)**. Both institutions approved the study under their respective IRB/DRB protocols.

---

## Citation

```bibtex
@mastersthesis{seol2025multimodal,
  title  = {Multimodal Deep Learning for Non-Invasive Cardiac Output Estimation in Critically Ill Patients},
  author = {Seol, Heeseung},
  year   = {2025},
  note   = {Samsung Medical Center Collaborative Research, IRB/DRB Approved}
}
```

---

*Model weights and the full multimodal implementation are not publicly released. The baseline model in `1D-CNN BiLSTM Attention/` is available for reference. Questions can go through GitHub Issues.*
