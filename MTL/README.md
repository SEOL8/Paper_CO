# MTL — Proposed multimodal model (COPredNet)

This directory contains the architecture and training code for the proposed multimodal CO prediction model described in the paper. **Trained weights are not publicly released.**

---

## Files

| File | Description |
|---|---|
| `MTL_ver1.py` | `COPredNet` — proposed model architecture. ECG and PPG autoencoders for time-domain feature extraction with reconstruction objective. FFT-based frequency encoders (ConvFormer). Patient demographic MLP. 5-token Transformer fusion (ModalFusion) across ECG/PPG time and frequency tokens plus patient token. |
| `training_ver1.py` | End-to-end joint training: CO prediction (SmoothL1) + reconstruction (PRD), weighted as `0.8 × L_pred + 0.2 × L_recon`. |
| `training_ver2.py` | Staged training: Stage 1 pre-trains the ECG and PPG autoencoders with reconstruction loss only; Stage 2 freezes the autoencoders and trains the remaining modules on CO prediction. |

---

## Architecture (`MTL_ver1.py` — COPredNet)

`COPredNet` takes five inputs:

| Input | Shape | Description |
|---|---|---|
| `ecg_time` | (B, 2500, 1) | Raw ECG, 20 s at 125 Hz |
| `ppg_time` | (B, 2500, 1) | Raw PPG, 20 s at 125 Hz |
| `ecg_freq` | (B, F, 1) | FFT magnitude spectrum of ECG |
| `ppg_freq` | (B, F, 1) | FFT magnitude spectrum of PPG |
| `patient_info` | (B, 4) | [Sex, Age, Ht, Wt], StandardScaler normalized |

Each input is encoded into a 128-dimensional token:
- Time signals: `SignalAutoEncoder` (1D CNN with skip connections, latent dim 512) → projected to 128
- Frequency signals: `FreqEncoder` (ConvFormer, 3 blocks) → 128
- Patient info: `PatientEncoder` (2-layer MLP) → 64 → projected to 128

The five tokens are fused by `ModalFusion` (2-layer Transformer encoder, 4 attention heads). The fused output is mean-pooled and passed to the prediction head.

Outputs: `pred_co (B, 1)`, `ecg_recon (B, 2500, 1)`, `ppg_recon (B, 2500, 1)`

---

## Results (internal test set, SMC)

| Metric | Value |
|---|---|
| RMSE | 0.79 L/min |
| MAE | 0.67 L/min |
| Mean bias | 0.01% |
| PE | 31.24% |

External validation (SNUBH): PE = 39%.

---

## Data format

The training scripts expect batches as dicts with keys:

| Key | Shape | Description |
|---|---|---|
| `ecg_time` | (B, 2500, 1) | Raw ECG |
| `ppg_time` | (B, 2500, 1) | Raw PPG |
| `ecg_freq` | (B, F, 1) | FFT of ECG |
| `ppg_freq` | (B, F, 1) | FFT of PPG |
| `patient_info` | (B, 4) | Standardized demographics |
| `co_label` | (B,) | CO target in L/min |
| `pid` | (B,) | Randomized patient ID |

A dataset loader for this format can be built by extending `data/dataset.py` at the repository root to compute FFT features on the fly or load them from pre-computed files.

---

## Notes

- `training_ver1.py` and `training_ver2.py` provide training loop functions to be called from an orchestrating training script; they are not standalone runnable scripts.
- `run_validation` in `training_ver2.py` uses CO loss only (no reconstruction term), suited for Stage 2 evaluation.
- The `return_attn=True` flag on `COPredNet.forward()` returns per-layer attention weights from `ModalFusion`, useful for inspecting which modality tokens the model attends to for a given patient.
