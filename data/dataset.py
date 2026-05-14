import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class PPGECGDataset(Dataset):
    def __init__(self, ppg, ecg, patient_info, targets, pids):
        self.ppg          = ppg
        self.ecg          = ecg
        self.patient_info = patient_info
        self.targets      = targets
        self.pids         = pids

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.ppg[idx], self.ecg[idx], self.patient_info[idx], self.targets[idx], self.pids[idx]


def normalize_per_segment(data: np.ndarray) -> np.ndarray:
    """Per-row min-max normalization to [-1, 1]."""
    sig_min = data.min(axis=1, keepdims=True)
    sig_max = data.max(axis=1, keepdims=True)
    denom   = np.where(sig_max - sig_min == 0, 1.0, sig_max - sig_min)
    return 2.0 * (data - sig_min) / denom - 1.0


def load_and_preprocess(df, max_len: int = 2500):
    n = len(df)

    ppg_raw = np.zeros((n, max_len))
    for i, sig in tqdm(enumerate(df['ppg']), total=n, desc="Loading PPG"):
        length = min(max_len, len(sig))
        ppg_raw[i, :length] = sig[:length]

    ecg_raw = np.zeros((n, max_len))
    for i, sig in tqdm(enumerate(df['ecg']), total=n, desc="Loading ECG"):
        length = min(max_len, len(sig))
        ecg_raw[i, :length] = sig[:length]

    ppg_norm = normalize_per_segment(ppg_raw)[:, :, np.newaxis]   # (N, T, 1)
    ecg_norm = normalize_per_segment(ecg_raw)[:, :, np.newaxis]   # (N, T, 1)

    patient_info = df[['Sex', 'Age', 'Ht', 'Wt']].values
    targets      = df['co'].values
    pids         = df['pid'].astype(np.int64).values

    return ppg_norm, ecg_norm, patient_info, targets, pids


def build_loaders(train_df, val_df, test_df, config):
    max_len    = config['max_len']
    batch_size = config['batch_size']

    ppg_tr, ecg_tr, info_tr, y_tr, pid_tr = load_and_preprocess(train_df, max_len)
    ppg_va, ecg_va, info_va, y_va, pid_va = load_and_preprocess(val_df,   max_len)
    ppg_te, ecg_te, info_te, y_te, pid_te = load_and_preprocess(test_df,  max_len)

    scaler  = StandardScaler()
    info_tr = scaler.fit_transform(info_tr)
    info_va = scaler.transform(info_va)
    info_te = scaler.transform(info_te)

    def t(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype)

    train_ds = PPGECGDataset(t(ppg_tr), t(ecg_tr), t(info_tr), t(y_tr), t(pid_tr, torch.int64))
    val_ds   = PPGECGDataset(t(ppg_va), t(ecg_va), t(info_va), t(y_va), t(pid_va, torch.int64))
    test_ds  = PPGECGDataset(t(ppg_te), t(ecg_te), t(info_te), t(y_te), t(pid_te, torch.int64))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader
