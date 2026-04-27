import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    """
    1D CNN for temporal feature extraction from a single biosignal.
    Input  : (B, 1, T)
    Output : (B, T', 256)
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(2, 2), nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32,  64,  kernel_size=7,  stride=2, padding=3),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2, 2), nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64,  128, kernel_size=5,  stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Dropout(dropout),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3,  stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return x.transpose(1, 2)   # (B, T', 256)


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM that returns the full output sequence.
    Input  : (B, T', 256)
    Output : (B, T', hidden_size * 2)
    """
    def __init__(self, input_size: int = 256, hidden_size: int = 256,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out   # (B, T', hidden_size * 2)


class SignalAttention(nn.Module):
    """
    Local windowed self-attention over a biosignal sequence.
    Returns a single context vector per sample via mean-pooling.

    Input  : (B, T', in_features)
    Output : (B, hidden_features)
    """
    def __init__(self, in_features: int, hidden_features: int, window_size: int = 32):
        super().__init__()
        self.W_q = nn.Linear(in_features, hidden_features)
        self.W_k = nn.Linear(in_features, hidden_features)
        self.W_v = nn.Linear(in_features, hidden_features)

        self.norm_q = nn.InstanceNorm1d(hidden_features)
        self.norm_k = nn.InstanceNorm1d(hidden_features)
        self.norm_v = nn.InstanceNorm1d(hidden_features)

        self.window_size = window_size
        self.scale       = hidden_features ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.W_q(x).transpose(1, 2)).transpose(1, 2)
        k = self.norm_k(self.W_k(x).transpose(1, 2)).transpose(1, 2)
        v = self.norm_v(self.W_v(x).transpose(1, 2)).transpose(1, 2)

        seq_len = x.size(1)
        idx  = torch.arange(seq_len, device=x.device)
        mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.window_size // 2

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)
        return out.mean(dim=1)   # (B, H)


class MLP(nn.Module):
    """
    Patient metadata encoder.
    Input  : (B, 4)
    Output : (B, out_features)
    """
    def __init__(self, in_features: int = 4, out_features: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout),
            nn.Linear(64, out_features),
            nn.ReLU(), nn.BatchNorm1d(out_features), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionAttention(nn.Module):
    """
    Self-attention across modality feature vectors (PPG, ECG, patient info).

    Input  : (B, N, embed_dim)   N = 3 modalities
    Output : (B, N, embed_dim)
    """
    def __init__(self, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.W_q   = nn.Linear(embed_dim, embed_dim)
        self.W_k   = nn.Linear(embed_dim, embed_dim)
        self.W_v   = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5
        self.drop  = nn.Dropout(dropout)
        self.norm  = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.bmm(attn, v)
        return self.norm(out + x)   # residual + LayerNorm


class SignalProcessingModel(nn.Module):

    def __init__(self, hidden_size: int = 256, lstm_layers: int = 3, dropout: float = 0.3):
        super().__init__()

        self.ppg_cnn       = CNN1D(dropout=0.1)
        self.ecg_cnn       = CNN1D(dropout=0.1)

        self.ppg_bilstm    = BiLSTM(input_size=256, hidden_size=hidden_size,
                                    num_layers=lstm_layers, dropout=dropout)
        self.ecg_bilstm    = BiLSTM(input_size=256, hidden_size=hidden_size,
                                    num_layers=lstm_layers, dropout=dropout)

        self.ppg_attention = SignalAttention(in_features=hidden_size * 2, hidden_features=hidden_size)
        self.ecg_attention = SignalAttention(in_features=hidden_size * 2, hidden_features=hidden_size)

        self.patient_mlp   = MLP(in_features=4, out_features=hidden_size, dropout=dropout)

        self.fusion_attention = FusionAttention(embed_dim=hidden_size)

        self.output_head   = nn.Linear(hidden_size * 3, 1)

    def forward(self, ppg: torch.Tensor, ecg: torch.Tensor,
                patient_info: torch.Tensor) -> torch.Tensor:

        ppg_feat     = self.ppg_attention(
                           self.ppg_bilstm(
                               self.ppg_cnn(ppg.squeeze(-1).unsqueeze(1))))

        ecg_feat     = self.ecg_attention(
                           self.ecg_bilstm(
                               self.ecg_cnn(ecg.squeeze(-1).unsqueeze(1))))

        patient_feat = self.patient_mlp(patient_info)

        modalities = torch.stack([ppg_feat, ecg_feat, patient_feat], dim=1)   # (B, 3, H)
        fused      = self.fusion_attention(modalities)                         # (B, 3, H)

        out = self.output_head(fused.flatten(1))   # (B, 1)
        return out.squeeze(-1)                      # (B,) — squeeze(-1) to avoid scalar when B=1
