# MTL — Proposed multimodal model development

This directory contains the architecture and training code for the proposed multimodal CO prediction model described in the paper. It is development-stage code showing the iterative progression toward the final model.

**The final trained weights are not publicly released.**

---

## Files

| File | Description |
|---|---|
| `MTL_ver1.py` | `COPredNet` — full model architecture. 5-token Transformer fusion over ECG (time + freq), PPG (time + freq), and patient info. Includes ECG/PPG autoencoders for multi-task reconstruction. |
| `training_ver1.py` | End-to-end joint training: CO prediction (SmoothL1) + reconstruction (PRD), combined as `0.8 × L_pred + 0.2 × L_recon`. |
| `training_ver2.py` | Staged training alternative: Stage 1 pre-trains autoencoders with reconstruction loss only; Stage 2 freezes the autoencoders and trains the remaining modules on CO prediction. |

---

## Model overview (`MTL_ver1.py`)

`COPredNet` takes five inputs:

| Input | Shape | Description |
|---|---|---|
| `ecg_time` | (B, 2500, 1) | Raw ECG, 20 s at 125 Hz |
| `ppg_time` | (B, 2500, 1) | Raw PPG, 20 s at 125 Hz |
| `ecg_freq` | (B, F, 1) | FFT magnitude of ECG |
| `ppg_freq` | (B, F, 1) | FFT magnitude of PPG |
| `patient_info` | (B, 4) | [Sex, Age, Ht, Wt], StandardScaler normalized |

Each input is encoded independently into a 128-dimensional token. The five tokens are fused by a 2-layer Transformer encoder (`ModalFusion`). The mean-pooled output goes to the prediction head.

Outputs: `pred_co (B, 1)`, `ecg_recon (B, 2500, 1)`, `ppg_recon (B, 2500, 1)`

---

## Data format

The training scripts expect batches as dicts with keys: `ecg_time`, `ppg_time`, `ecg_freq`, `ppg_freq`, `patient_info`, `co_label`, `pid`. A dataset loader for this format is not included here; it can be built by extending `data/dataset.py` at the repository root to compute FFT on the fly or load pre-computed frequency features.

---

## Notes

- `training_ver1.py` and `training_ver2.py` are not standalone scripts; they provide training loop functions to be called from an orchestrating training script.
- The `run_validation` function in `training_ver2.py` differs from the one in `training_ver1.py`: it uses CO loss only (no reconstruction term), suited for Stage 2 validation.
- Results for this model variant (internal test set, SMC): RMSE=0.79, MAE=0.67, R²=0.59, PE=31.24%, Mean Bias=0.01%.
