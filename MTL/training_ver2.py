"""
training_ver2.py — Staged Training v2

Stage 1 : AutoEncoder pre-training
    ecg_time_enc + ppg_time_enc 만 학습
    Loss: PRD(ecg_recon, ecg_orig) + PRD(ppg_recon, ppg_orig)

Stage 2 : CO prediction (AE frozen)
    AE 동결 후 FreqEncoder + PatientEncoder + ModalFusion + pred_head 학습
    Loss: SmoothL1(pred_co, true_co)
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from MTL_ver1 import COPredNet
from training_ver1 import prd_loss


# ---------------------------------------------------------------------------
# AE parameter helpers
# ---------------------------------------------------------------------------

def _base(model) -> COPredNet:
    return model.module if isinstance(model, nn.DataParallel) else model


def freeze_autoencoders(model):
    for enc in (_base(model).ecg_time_enc, _base(model).ppg_time_enc):
        for p in enc.parameters():
            p.requires_grad = False


def unfreeze_autoencoders(model):
    for enc in (_base(model).ecg_time_enc, _base(model).ppg_time_enc):
        for p in enc.parameters():
            p.requires_grad = True


def ae_parameters(model):
    b = _base(model)
    return list(b.ecg_time_enc.parameters()) + list(b.ppg_time_enc.parameters())


def non_ae_parameters(model):
    ae_ids = {id(p) for p in ae_parameters(model)}
    return [p for p in model.parameters() if id(p) not in ae_ids]


# ---------------------------------------------------------------------------
# Stage 1 — AE pre-training
# ---------------------------------------------------------------------------

def pretrain_ae_epoch(model, loader, optimizer, device) -> dict:
    """Train only ecg_time_enc / ppg_time_enc with reconstruction loss."""
    model.train()
    running = {'ecg': [], 'ppg': [], 'total': []}

    for batch in tqdm(loader, desc='AE pretrain', leave=False):
        ecg_time = batch['ecg_time'].to(device).contiguous()
        ppg_time = batch['ppg_time'].to(device).contiguous()

        optimizer.zero_grad()
        _, ecg_recon, ppg_recon = model(
            ecg_time, ppg_time,
            batch['ecg_freq'].to(device).contiguous(),
            batch['ppg_freq'].to(device).contiguous(),
            batch['patient_info'].to(device).contiguous(),
        )
        l_ecg  = prd_loss(ecg_recon, ecg_time)
        l_ppg  = prd_loss(ppg_recon, ppg_time)
        loss   = 0.5 * (l_ecg + l_ppg)
        loss.backward()
        optimizer.step()

        running['ecg'].append(l_ecg.item())
        running['ppg'].append(l_ppg.item())
        running['total'].append(loss.item())

    return {k: float(np.mean(v)) for k, v in running.items()}


# ---------------------------------------------------------------------------
# Stage 2 — CO prediction (AE frozen)
# ---------------------------------------------------------------------------

def train_co_epoch(model, loader, optimizer, device, grad_clip: float = 0.5) -> dict:
    """Train all non-AE modules with SmoothL1 CO loss. AE must be frozen beforehand."""
    model.train()
    co_loss_fn = nn.SmoothL1Loss()
    losses = []

    for batch in tqdm(loader, desc='CO train', leave=False):
        ecg_time = batch['ecg_time'].to(device).contiguous()
        ppg_time = batch['ppg_time'].to(device).contiguous()
        ecg_freq = batch['ecg_freq'].to(device).contiguous()
        ppg_freq = batch['ppg_freq'].to(device).contiguous()
        pat_info = batch['patient_info'].to(device).contiguous()
        co_true  = batch['co_label'].to(device).contiguous()

        optimizer.zero_grad()
        pred_co, _, _ = model(ecg_time, ppg_time, ecg_freq, ppg_freq, pat_info)
        loss = co_loss_fn(pred_co.view(-1), co_true.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(loss.item())

    return {'co': float(np.mean(losses))}


# ---------------------------------------------------------------------------
# Validation (shared with ver1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(model, loader, device) -> dict:
    model.eval()
    co_loss_fn = nn.SmoothL1Loss()
    losses, all_preds, all_targets, all_pids = [], [], [], []

    for batch in tqdm(loader, desc='Val', leave=False):
        ecg_time = batch['ecg_time'].to(device).contiguous()
        ppg_time = batch['ppg_time'].to(device).contiguous()
        ecg_freq = batch['ecg_freq'].to(device).contiguous()
        ppg_freq = batch['ppg_freq'].to(device).contiguous()
        pat_info = batch['patient_info'].to(device).contiguous()
        co_true  = batch['co_label'].to(device).contiguous()

        pred_co, _, _ = model(ecg_time, ppg_time, ecg_freq, ppg_freq, pat_info)
        losses.append(co_loss_fn(pred_co.view(-1), co_true.view(-1)).item())
        all_preds.extend(pred_co.cpu().numpy())
        all_targets.extend(co_true.cpu().numpy())
        all_pids.extend(batch['pid'])

    y_pred = np.array(all_preds).flatten()
    y_true = np.array(all_targets).flatten()
    rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        'co_loss': float(np.mean(losses)),
        'rmse'   : rmse,
        'preds'  : y_pred,
        'targets': y_true,
        'pids'   : np.array(all_pids),
    }
