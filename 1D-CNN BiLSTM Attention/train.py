import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import CONFIG
from data.dataset import build_loaders
from models.cnn_bilstm_attention import SignalProcessingModel
from utils.metrics import calculate_metrics, print_metrics
from utils.early_stopping import EarlyStopping
from evaluate import evaluate_and_visualize


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, config, device, save_path):
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'], verbose=True)

    for epoch in range(config['epochs']):

        # ── Train
        model.train()
        train_loss, train_preds, train_targets = 0.0, [], []
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')

        for ppg, ecg, patient_info, targets, _ in train_pbar:
            ppg, ecg, patient_info, targets = (
                ppg.to(device), ecg.to(device),
                patient_info.to(device), targets.to(device)
            )
            optimizer.zero_grad()
            outputs = model(ppg, ecg, patient_info)
            loss    = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # ── Validate
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []

        with torch.no_grad():
            for ppg, ecg, patient_info, targets, _ in val_loader:
                ppg, ecg, patient_info, targets = (
                    ppg.to(device), ecg.to(device),
                    patient_info.to(device), targets.to(device)
                )
                outputs  = model(ppg, ecg, patient_info)
                loss     = criterion(outputs, targets)
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        train_metrics = calculate_metrics(np.array(train_targets), np.array(train_preds),
                                          train_loss / len(train_loader))
        val_metrics   = calculate_metrics(np.array(val_targets),   np.array(val_preds),
                                          val_loss   / len(val_loader))

        scheduler.step(val_metrics['loss'])

        print('\n' + '=' * 100)
        print(f'Epoch {epoch+1}/{config["epochs"]}')
        print('-' * 100)
        print('Train :'); print_metrics(train_metrics)
        print('Val   :'); print_metrics(val_metrics)
        print(f'LR    : {optimizer.param_groups[0]["lr"]:.6f}')
        print('=' * 100)

        early_stopping(val_metrics['loss'], model, save_path)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    return model


def main():
    config = CONFIG

    # ── Load data
    train_df = pd.read_pickle(os.path.join(config['data_dir'], 'Train.pkl'))
    val_df   = pd.read_pickle(os.path.join(config['data_dir'], 'Val_df_final.pkl'))
    test_df  = pd.read_pickle(os.path.join(config['data_dir'], 'Test.pkl'))

    print("Train PIDs:", sorted(train_df['pid'].unique()))
    print("Val   PIDs:", sorted(val_df['pid'].unique()))
    print("Test  PIDs:", sorted(test_df['pid'].unique()))

    train_pids, val_pids, test_pids = (
        set(train_df['pid'].unique()),
        set(val_df['pid'].unique()),
        set(test_df['pid'].unique()),
    )
    print("\n[Overlap Check]")
    print("Train ∩ Val :", train_pids & val_pids)
    print("Train ∩ Test:", train_pids & test_pids)
    print("Val   ∩ Test:", val_pids   & test_pids)

    # ── Build loaders
    train_loader, val_loader, test_loader = build_loaders(train_df, val_df, test_df, config)

    # ── Setup
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Model
    model = SignalProcessingModel(
        hidden_size=config['hidden_size'],
        lstm_layers=config['lstm_layers'],
        dropout=config['dropout'],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        min_lr=config['scheduler_min_lr'],
    )

    # ── Train
    model_save_path = os.path.join(save_dir, 'best_model.pth')
    model = train_model(model, train_loader, val_loader, criterion, optimizer,
                        scheduler, config, device, model_save_path)

    # ── Evaluate (best checkpoint)
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    results = evaluate_and_visualize(model, test_loader, device, save_dir)
    return results


if __name__ == '__main__':
    main()
