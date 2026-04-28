"""
training_ver1.py — Loss + End-to-End Training v1

Loss
    COLoss = 0.8 * SmoothL1(CO) + 0.2 * PRD(ECG_recon + PPG_recon)

End-to-End flow
    raw signal → COPredNet → (pred_co, ecg_recon, ppg_recon) → COLoss → backprop
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from MTL_ver1 import COPredNet


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def prd_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Percentage Root-mean-square Difference."""
    if pred.dim() == 3:   pred   = pred.squeeze(-1)
    if target.dim() == 3: target = target.squeeze(-1)
    return torch.mean(torch.sqrt(
        torch.mean((pred - target) ** 2, dim=1) / (torch.mean(target ** 2, dim=1) + eps)
    ))


class COLoss(nn.Module):
    """
    total = co_weight * SmoothL1(CO) + recon_weight * 0.5 * (PRD_ECG + PRD_PPG)

    Default weights: co=0.8, recon=0.2
    """
    def __init__(self, co_weight: float = 0.8, recon_weight: float = 0.2):
        super().__init__()
        self.co_weight    = co_weight
        self.recon_weight = recon_weight
        self.co_loss_fn   = nn.SmoothL1Loss()

    def forward(self, pred_co, true_co, ecg_recon, ecg_orig, ppg_recon, ppg_orig):
        co_loss    = self.co_loss_fn(pred_co.view(-1), true_co.view(-1))
        recon_loss = 0.5 * (prd_loss(ecg_recon, ecg_orig) + prd_loss(ppg_recon, ppg_orig))
        total      = self.co_weight * co_loss + self.recon_weight * recon_loss
        return total, {'total': total.item(), 'co': co_loss.item(), 'recon': recon_loss.item()}


# ---------------------------------------------------------------------------
# End-to-End training / validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip: float = 0.5) -> dict:
    model.train()
    running = {'total': [], 'co': [], 'recon': []}

    for batch in tqdm(loader, desc='Train', leave=False):
        ecg_time = batch['ecg_time'].to(device).contiguous()
        ppg_time = batch['ppg_time'].to(device).contiguous()
        ecg_freq = batch['ecg_freq'].to(device).contiguous()
        ppg_freq = batch['ppg_freq'].to(device).contiguous()
        pat_info = batch['patient_info'].to(device).contiguous()
        co_true  = batch['co_label'].to(device).contiguous()

        optimizer.zero_grad()
        pred_co, ecg_recon, ppg_recon = model(ecg_time, ppg_time, ecg_freq, ppg_freq, pat_info)
        loss, loss_parts = criterion(pred_co, co_true, ecg_recon, ecg_time, ppg_recon, ppg_time)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k in running:
            running[k].append(loss_parts[k])

    return {k: float(np.mean(v)) for k, v in running.items()}


@torch.no_grad()
def run_validation(model, loader, criterion, device) -> tuple:
    model.eval()
    running = {'total': [], 'co': [], 'recon': []}
    all_preds, all_targets, all_pids = [], [], []

    for batch in tqdm(loader, desc='Val', leave=False):
        ecg_time = batch['ecg_time'].to(device).contiguous()
        ppg_time = batch['ppg_time'].to(device).contiguous()
        ecg_freq = batch['ecg_freq'].to(device).contiguous()
        ppg_freq = batch['ppg_freq'].to(device).contiguous()
        pat_info = batch['patient_info'].to(device).contiguous()
        co_true  = batch['co_label'].to(device).contiguous()

        pred_co, ecg_recon, ppg_recon = model(ecg_time, ppg_time, ecg_freq, ppg_freq, pat_info)
        _, loss_parts = criterion(pred_co, co_true, ecg_recon, ecg_time, ppg_recon, ppg_time)

        for k in running:
            running[k].append(loss_parts[k])
        all_preds.extend(pred_co.cpu().numpy())
        all_targets.extend(co_true.cpu().numpy())
        all_pids.extend(batch['pid'])

    avg_loss = {k: float(np.mean(v)) for k, v in running.items()}
    y_pred   = np.array(all_preds).flatten()
    y_true   = np.array(all_targets).flatten()
    rmse     = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return avg_loss, {'rmse': rmse, 'preds': y_pred, 'targets': y_true, 'pids': np.array(all_pids)}
