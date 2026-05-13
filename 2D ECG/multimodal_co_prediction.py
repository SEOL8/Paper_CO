import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class PPGECGDataset(Dataset):
    """
    Each pickle must contain:
      - ppg             : 1-D array, raw signal (~2500 samples at 250 Hz)
      - ecg_s_transform : 2-D array (599, 2500), pre-computed in S-Transform_test.ipynb
      - Sex, Age, Ht, Wt: scalar demographics
      - co              : cardiac output label (L/min)
    """
    def __init__(self, data_path, patient_scaler=None, max_ppg_len=2500):
        self.data = pd.read_pickle(data_path)
        self.max_ppg_len = max_ppg_len

        patient_info = self.data[['Sex', 'Age', 'Ht', 'Wt']].values.astype(np.float32)
        if patient_scaler is None:
            self.scaler = StandardScaler()
            self.patient_info = self.scaler.fit_transform(patient_info)
        else:
            self.scaler = patient_scaler
            self.patient_info = self.scaler.transform(patient_info)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        ppg = np.zeros(self.max_ppg_len, dtype=np.float32)
        sig = row['ppg']
        n = min(self.max_ppg_len, len(sig))
        ppg[:n] = sig[:n]
        ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
        ppg_tensor = torch.from_numpy(ppg).unsqueeze(-1)         # (2500, 1)

        ecg_st = row['ecg_s_transform']
        if not isinstance(ecg_st, np.ndarray):
            ecg_st = np.zeros((1, 599, 2500), dtype=np.float32)
        if ecg_st.ndim == 2:
            ecg_st = ecg_st[np.newaxis]
        ecg_tensor = torch.from_numpy(ecg_st.astype(np.float32))  # (1, 599, 2500)

        patient_tensor = torch.from_numpy(self.patient_info[idx])  # (4,)
        co_tensor = torch.tensor(float(row['co']), dtype=torch.float32)

        return ppg_tensor, ecg_tensor, patient_tensor, co_tensor


def build_dataloaders(data_dir='./', batch_size=64):
    train_set = PPGECGDataset(os.path.join(data_dir, 'train.pkl'))
    val_set   = PPGECGDataset(os.path.join(data_dir, 'val.pkl'),  patient_scaler=train_set.scaler)
    test_set  = PPGECGDataset(os.path.join(data_dir, 'test.pkl'), patient_scaler=train_set.scaler)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader


# Recalibrates activations along frequency, time, and channel axes of the S-Transform map
class SpectralAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 8, 1)
        self.freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, mid, 1), nn.ReLU(),
            nn.Conv2d(mid, channels, 1), nn.Sigmoid(),
        )
        self.time_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, mid, 1), nn.ReLU(),
            nn.Conv2d(mid, channels, 1), nn.Sigmoid(),
        )
        self.chan_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 1), 1), nn.ReLU(),
            nn.Conv2d(max(channels // 4, 1), channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.freq_attn(x) * self.time_attn(x) * self.chan_attn(x)


# Applies separate frequency-axis and time-axis convolutions, then merges via residual
class FrequencyTimeFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )
        self.time_conv = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )

    def forward(self, x):
        merged = torch.cat([self.freq_conv(x), self.time_conv(x)], dim=1)
        return x + self.fusion(merged)


class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x))


class PPGModule(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),    # 2500 → 625
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),   #  625 → 156
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4), #  156 → 39
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),                 #   39
        )
        # bidirectional: hidden 128 × 2 = 256
        self.bilstm = nn.LSTM(256, 128, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.3)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, 2500, 1)
        x = self.cnn(x.transpose(1, 2))        # (B, 256, 39)
        x, _ = self.bilstm(x.transpose(1, 2))  # (B, 39, 256)
        return self.drop(x[:, -1, :])           # (B, 256)


class ECGSTransformModule(nn.Module):
    # S-Transform produces a 2D matrix (freq × time) that encodes both morphological
    # and spectral information simultaneously. Passing it through a 2D ResNet lets the
    # network learn fine spatial patterns across both axes — which a plain 1D conv cannot do.
    def __init__(self, out_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(                                       # (B, 1, 599, 2500)
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),           # → (B, 64, 300, 1250)
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),                           # → (B, 64, 150, 625)
        )
        self.layer1 = self._make_layer(64, 64, blocks=2)                    # → (B, 64, 150, 625)
        self.spectral_attn1 = SpectralAttentionBlock(64)

        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)         # → (B, 128, 75, 313)
        self.freq_time_fusion = FrequencyTimeFusion(128)

        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)        # → (B, 256, 38, 157)
        self.spectral_attn2 = SpectralAttentionBlock(256)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, out_dim), nn.ReLU(), nn.Dropout(0.3),
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = [ResidualBlock2D(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock2D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.spectral_attn1(self.layer1(x))
        x = self.freq_time_fusion(self.layer2(x))
        x = self.spectral_attn2(self.layer3(x))
        return self.fc(torch.flatten(self.global_pool(x), 1))  # (B, 256)


class PatientInfoModule(nn.Module):
    def __init__(self, in_dim=4, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class MultimodalCOPredictor(nn.Module):
    # Feature dims: PPG 256 + ECG 256 + Patient 64 = 576
    def __init__(self):
        super().__init__()
        self.ppg_module     = PPGModule(out_dim=256)
        self.ecg_module     = ECGSTransformModule(out_dim=256)
        self.patient_module = PatientInfoModule(in_dim=4, out_dim=64)

        self.fusion = nn.Sequential(
            nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, ppg, ecg_st, patient_info):
        """
        ppg         : (B, 2500, 1)       raw PPG waveform
        ecg_st      : (B, 1, 599, 2500)  S-Transform of ECG
        patient_info: (B, 4)             [Sex, Age, Ht, Wt], StandardScaler applied
        returns     : (B,)               predicted CO in L/min
        """
        ppg_feat     = self.ppg_module(ppg)
        ecg_feat     = self.ecg_module(ecg_st)
        patient_feat = self.patient_module(patient_info)
        combined     = torch.cat([ppg_feat, ecg_feat, patient_feat], dim=1)
        return self.fusion(combined).squeeze(1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalCOPredictor().to(device)
    model.eval()

    B = 4
    ppg          = torch.randn(B, 2500, 1).to(device)
    ecg_st       = torch.randn(B, 1, 599, 2500).to(device)
    patient_info = torch.randn(B, 4).to(device)

    with torch.no_grad():
        out = model(ppg, ecg_st, patient_info)

    print(f'PPG     : {tuple(ppg.shape)}')
    print(f'ECG ST  : {tuple(ecg_st.shape)}')
    print(f'Patient : {tuple(patient_info.shape)}')
    print(f'Output  : {tuple(out.shape)}')
