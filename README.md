# A Multimodal Deep Learning Study for Noninvasive Cardiac Output Prediction in ICU Patients
## — Fusion of Biosignals and Patient Information —

**Samsung Medical Center collaborative research · Master's thesis (2023–2025) · IRB/DRB approved**

---

## Overview

Cardiac output matters in the ICU because it tells you how well the heart is actually perfusing the body. The current reference standard, intermittent thermodilution via a pulmonary artery catheter (Swan-Ganz), works well enough, but threading a catheter into the pulmonary artery carries infection risk, can trigger arrhythmias on insertion, and gives you a point-in-time number rather than a continuous trend.

This study tests whether ECG, PPG, and basic demographic data already collected on most ICU patients contain enough information to estimate CO without a catheter. On 642 ICU patients across two institutions, the proposed multimodal model achieves **PE = 31.24%** on internal test data and **PE = 39%** on an external validation set collected with different monitoring hardware, both within the clinical acceptability range for non-invasive CO methods.

<p align="center">
  <img src="figures/hemodynamic.png" alt="Hemodynamic relationships" width="420">
</p>

> CO = HR × SV. Preload, afterload, and contractility converge on stroke volume; heart rate and stroke volume together give cardiac output, which, combined with systemic vascular resistance, sets mean arterial pressure.

---

## Clinical background

The reference standard is thermodilution via pulmonary artery catheterization (Stewart-Hamilton equation). Critchley & Critchley (1999) put the clinical interchangeability threshold at PE < 30%: if thermodilution carries ±20% precision, combining two methods each at ±20% produces a PE of about 28.3%, rounded to 30%. That criterion assumes thermodilution is genuinely that precise, which doesn't always hold in practice.

Peyton & Chong (2010), in a meta-analysis of minimally invasive CO monitoring, argued the 30% bar is too strict given thermodilution's real-world variability, and proposed **PE < 45%** as a more realistic threshold for non-invasive techniques. Most existing methods fall somewhere between the two benchmarks:

<div align="center">

| Invasiveness | Technology | PE (%) | Methodology | Limitations |
|---|---|---|---|---|
| Invasive (standard) | Thermodilution (PAC) | Ref | Temperature change curve after coolant injection (Stewart-Hamilton) | Risk of infection, arrhythmia; requires specialized procedure |
| Non-invasive | Partial CO₂ rebreathing | 40% | End-tidal CO₂ partial pressure changes (Fick's principle) | Not applicable to patients with lung disease or shunt effects |
| Non-invasive | Bioimpedance (TEB) | 42% | Blood flow estimation via thoracic electrical resistance changes | Sensitive to electrical noise and body fluid conditions |
| Non-invasive | Pulse contour (niPCA) | 45% | Finger blood pressure waveform area and shape (volume clamp) | Errors during peripheral vasoconstriction; affected by vascular elasticity |
| Non-invasive | Pulse wave transit time (PWTT) | 62% | Pulse transit time measured at ECG R-peak | Highest error rate; frequent recalibration required |

</div>

This study uses **PE < 42%** as its acceptance criterion — tighter than Peyton & Chong's 45%, and free from the applicability constraints that restrict partial CO₂ rebreathing to ventilated patients or make pulse contour unreliable during vasoconstriction.

---

## Dataset and cohort

This research used clinical data collected under IRB/DRB approval in collaboration with **Samsung Medical Center (SMC)** and Seoul National University Bundang Hospital (SNUBH).

**Raw data are not publicly available** due to patient privacy regulations and institutional data use agreements. The schema below documents the expected structure for independent replication with appropriately formatted institutional data.

### Cohort construction

3,733 ICU patients were screened across both institutions. Patients on ECMO or LVAD support were excluded because mechanical circulatory assist devices distort ECG and PPG morphology in ways that conflict with the physiological assumptions underlying the model. Cases with missing CO labels or signal dropout were also removed, leaving 642 patients.

<div align="center">

| Split | Institution | N | Role |
|---|---|---|---|
| Train / val / test (70/15/15) | Samsung Medical Center (SMC) | 501 | Model development |
| External validation | SNUBH | 141 | Generalizability test |

</div>

Splits are at the patient level. Every segment from a given patient lands in exactly one split, preventing leakage from the strong within-patient correlation in a sliding-window dataset.

### Signal structure

<div align="center">

| Field | Type | Description |
|---|---|---|
| `ppg` | array (2500,) | PPG time-series, 20 s at 125 Hz |
| `ecg` | array (2500,) | ECG time-series, 20 s at 125 Hz |
| `Sex` | float | De-identified biological sex |
| `Age` | float | De-identified age in years |
| `Ht` | float | De-identified height (cm) |
| `Wt` | float | De-identified weight (kg) |
| `co` | float | Cardiac output (L/min), thermodilution reference |
| `pid` | int | Randomized integer assigned during cohort construction; no link to any hospital record or personal identifier |

</div>

All personally identifiable information was removed during cohort construction per IRB/DRB de-identification requirements. The demographic fields (`Sex`, `Age`, `Ht`, `Wt`) are retained as physiological covariates only; no combination of these fields can re-identify any individual in the dataset.

Signals were segmented with a non-overlapping 20-second window at 125 Hz (2,500 samples per segment). The non-overlapping design preserves temporal continuity and avoids artificial autocorrelation between adjacent segments from the same patient.

---

## Model architecture

Three branches extract features in parallel — time-domain, frequency-domain, and patient demographics — then merge through a patient-conditioned attention mechanism.

<p align="center">
  <img src="figures/multimodal_architecture.png" alt="Full multimodal architecture" width="780">
</p>

### Time domain module

Raw ECG and PPG go through a U-Net style autoencoder. Three stages of 1D convolution and pooling compress the 20-second signal to a 256-dimensional latent vector. Skip connections between encoder and decoder stages preserve morphological features at multiple scales, while the bottleneck learns a compact summary. This branch trains with a reconstruction objective alongside CO prediction, which keeps the encoder from dropping physiologically relevant information that doesn't immediately help regression.

<p align="center">
  <img src="figures/autoencoder.png" alt="Autoencoder architecture" width="700">
</p>

### Frequency domain module

The same raw signals pass independently through an FFT, then a 1D CNN with batch normalization, ReLU, and max pooling. Heart rate variability, respiratory modulation of PPG amplitude, and low-frequency autonomic shifts are more readable in the spectral domain than from the raw waveform.

### Patient information module

A two-layer MLP encodes sex, age, height, and weight. Body size and composition have a real effect on stroke volume, and a model that ignores demographics will systematically mis-estimate CO for patients at the distribution extremes.

### Fusion module

The five modality representations — ECG (time and frequency), PPG (time and frequency), and patient information — are projected to a common 128-dimensional token space and fused through a 2-layer multi-head Transformer encoder. Self-attention across all five tokens lets each modality attend to every other, including the patient context, so signal branch weighting adapts to each patient's hemodynamic situation at inference time.

<p align="center">
  <img src="figures/cross_attention.png" alt="Patient-conditioned fusion" width="680">
</p>

---

## Training: multi-task learning

Two objectives train simultaneously.

**Task 1 — signal reconstruction.** The autoencoder reconstructs the original ECG and PPG from the latent representation. Quality is measured by Percent Root-mean-square Difference (PRD):

$$\text{PRD} = \sqrt{\frac{\displaystyle\sum_{i=1}^{M}(X_i - X_{\text{recon},i})^2}{\displaystyle\sum_{i=1}^{N}X_i^2}} \times 100$$

$$L_{\text{recon}} = \frac{\text{PRD}_{\text{ECG}} + \text{PRD}_{\text{PPG}}}{2}$$

This prevents the encoder from discarding information that is physiologically meaningful but not immediately predictive.

**Task 2 — CO prediction.** Fused features map to a scalar CO estimate. Huber loss (SmoothL1) is used instead of MSE because ICU CO measurements include genuine physiological extremes that MSE would let dominate training:

$$L_{\text{pred}} = \begin{cases} \dfrac{1}{2}(X_{\text{true}} - X_{\text{pred}})^2 & \text{if } |X_{\text{true}} - X_{\text{pred}}| < 1 \\ |X_{\text{true}} - X_{\text{pred}}| - \dfrac{1}{2} & \text{otherwise} \end{cases}$$

$$L_{\text{total}} = 0.8 \times L_{\text{pred}} \;+\; 0.2 \times L_{\text{recon}}$$

The 0.8/0.2 split keeps CO prediction as the primary objective; reconstruction acts as a regularizer on the encoder.

### Evaluation metrics

$$\text{RMSE} = \sqrt{\frac{1}{M}\sum_{i=1}^{M}(\text{CO}_{\text{true},i} - \text{CO}_{\text{pred},i})^2}$$

$$\text{MAE} = \frac{1}{M}\sum_{i=1}^{M}|\text{CO}_{\text{true},i} - \text{CO}_{\text{pred},i}|$$

$$R^2 = 1 - \frac{\displaystyle\sum_{i=1}^{M}(\text{CO}_{\text{true},i} - \text{CO}_{\text{pred},i})^2}{\displaystyle\sum_{i=1}^{M}(\text{CO}_{\text{true},i} - \overline{\text{CO}_{\text{true}}})^2}$$

### Hyperparameters

<div align="center">

| Parameter | Value |
|---|---|
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| LR scheduler | ReduceLROnPlateau |
| Max epochs | 300 |
| Early stopping patience | 30 |
| Latent dimension | 256 |
| Loss function | 0.8 × L_pred + 0.2 × L_recon |

</div>

---

## Results

### Model comparison

<div align="center">

| Model | Input modalities | RMSE | MAE | Mean bias (%) | PE (%) |
|---|---|---|---|---|---|
| 1D CNN-GRU | Time ECG, PPG, demographics | 1.325 | 1.12 | 4.2 | 42.1 |
| 1D CNN-LSTM + self-attention | Time PPG, demographics | 1.18 | 1.03 | 2.2 | 38.46 |
| XGBoost (engineered features) | ECG, PPG, demographics | 0.93 | 0.86 | 1.38 | 35.2 |
| 1D CNN-LSTM + self-attention (STFT) | STFT-ECG, time PPG, demographics | 0.91 | 0.89 | 1.7 | 35.5 |
| 1D CNN-2D ResNet + cross-attention | S-Transform ECG, time PPG, demographics | 0.832 | 0.78 | 0.7 | 32.62 |
| **Proposed multimodal** | **ECG+PPG (time & FFT), demographics** | **0.79** | **0.67** | **0.01** | **31.24** |

</div>

All results are from the SMC internal test set (patient-level split, N=501). External validation results are reported separately below.

### Bland-Altman analysis

RMSE alone doesn't tell you whether a method can replace another in clinical practice. A model can have low RMSE while still systematically underestimating high CO values or overestimating low ones. Bland-Altman analysis checks whether the two methods agree well enough across the full measurement range.

<p align="center">
  <img src="figures/bland_altman.png" alt="Bland-Altman plot" width="560">
</p>

<div align="center">

| Metric | Value |
|---|---|
| Mean bias | 0.01% |
| +1.96 SD | +31.25% |
| −1.96 SD | −31.23% |
| PE | 31.24% |

</div>

Mean bias near zero means no systematic over- or underestimation across the measured range. The ±1.96 SD limits fall within the PE < 42% acceptance criterion.

### External validation

On the SNUBH dataset, collected with HemoSphere equipment not seen during training, the model achieved **PE = 39%**, within the PE < 42% criterion. A different site, a different monitoring platform — and the error profile stayed consistent. That suggests the model is picking up on physiology rather than hardware-specific recording characteristics.

---

## Limitations

The cohort is entirely ICU patients on invasive hemodynamic monitoring. How the model performs in general wards, the emergency department, or perioperative settings is unknown.

Arrhythmia segments were excluded from the sliding-window extraction. The model's behavior on atrial fibrillation and other dysrhythmias was not tested.

All data are retrospective. Prospective validation is needed before any clinical application.

---

## Future work

- Arrhythmia segment evaluation with rhythm-stratified reporting
- Integration of clinical context (medications, surgical history) as additional modalities
- Prospective multi-site validation in ward and emergency settings
- Real-time inference feasibility on edge hardware

---

## Repository structure

```
Paper_CO/
├── data/                             # Shared data loading module
│   ├── __init__.py
│   └── dataset.py                    # PPGECGDataset, normalization, DataLoader builder
├── figures/                          # Architecture diagrams and result images
├── 1D-CNN BiLSTM Attention/          # Baseline model: 1D CNN + BiLSTM + self-attention
│   ├── models/
│   │   └── cnn_bilstm_attention.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── bland_altman.py
│   │   └── early_stopping.py
│   ├── train.py
│   ├── evaluate.py
│   ├── config.py
│   └── requirements.txt
├── 2D ECG/                           # Ablation: S-Transform ECG + 2D ResNet + PPG BiLSTM
│   ├── multimodal_co_prediction.py   # Model definition and data loader
│   └── S-Transform_test.ipynb        # ECG → S-Transform preprocessing
└── MTL/                              # Proposed multimodal model (COPredNet)
    ├── MTL_ver1.py                   # Architecture: autoencoder + freq encoder + Transformer fusion
    ├── training_ver1.py              # End-to-end joint training
    └── training_ver2.py              # Staged training: AE pretrain → CO fine-tune
```

**Code availability:**

- `1D-CNN BiLSTM Attention/` is fully self-contained and reproducible with appropriate data files.
- `2D ECG/` contains the S-Transform ECG ablation variant.
- `MTL/` contains the architecture and training code for the proposed multimodal model (COPredNet). Trained weights are not publicly released.
- `data/dataset.py` at root is the shared data loading module for the baseline. It is also present inside `1D-CNN BiLSTM Attention/data/` where the baseline training scripts expect it.

**Data are not included in this repository.** The schema in the Dataset section documents the expected format for replication with institutional data.

---

## Acknowledgements

This research was a collaboration with **Samsung Medical Center (SMC)**, which provided the clinical data, IRB/DRB approval, and institutional research support. External validation data came from **Seoul National University Bundang Hospital (SNUBH)**. Both institutions approved the study under their respective IRB/DRB protocols.

---

## Citation

```bibtex
@mastersthesis{seol2025multimodal,
  title  = {A Multimodal Deep Learning Study for Noninvasive Cardiac Output Prediction in ICU Patients},
  author = {Seol, Heeseung},
  year   = {2025},
  note   = {Samsung Medical Center Collaborative Research, IRB/DRB Approved}
}
```

---

*Trained weights for the proposed model are not publicly released. The baseline model in `1D-CNN BiLSTM Attention/` and the proposed model architecture in `MTL/` are available for reference. Questions can go through GitHub Issues.*
