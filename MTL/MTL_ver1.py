"""
MTL_ver1.py — Model Architecture v1

COPredNet
    ECG/PPG time-domain  : SignalAutoEncoder  (CNN-AE with skip connections)
    ECG/PPG freq-domain  : FreqEncoder        (ConvFormer)
    Patient demographics : PatientEncoder     (MLP)
    Fusion               : ModalFusion        (Transformer, 5 tokens)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalAutoEncoder(nn.Module):
    """
    1D CNN autoencoder with skip connections.
    Input  : (B, T, 1)
    Output : z (B, latent_dim),  recon (B, T, 1)
    """
    def __init__(self, signal_length: int = 2500, latent_dim: int = 512):
        super().__init__()
        self.enc_conv1 = nn.Conv1d(1,   64,  kernel_size=15, padding=7)
        self.enc_bn1   = nn.BatchNorm1d(64)
        self.enc_conv2 = nn.Conv1d(64,  64,  kernel_size=7,  padding=3)
        self.enc_bn2   = nn.BatchNorm1d(64)
        self.enc_pool1 = nn.MaxPool1d(4)
        self.enc_conv3 = nn.Conv1d(64,  128, kernel_size=7,  padding=3)
        self.enc_bn3   = nn.BatchNorm1d(128)
        self.enc_conv4 = nn.Conv1d(128, 128, kernel_size=5,  padding=2)
        self.enc_bn4   = nn.BatchNorm1d(128)
        self.enc_pool2 = nn.MaxPool1d(5)
        self.enc_conv5 = nn.Conv1d(128, 256, kernel_size=5,  padding=2)
        self.enc_bn5   = nn.BatchNorm1d(256)
        self.enc_conv6 = nn.Conv1d(256, 256, kernel_size=3,  padding=1)
        self.enc_bn6   = nn.BatchNorm1d(256)
        self.enc_pool3 = nn.MaxPool1d(5)
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.enc_fc    = nn.Sequential(
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.2),     nn.Linear(512, latent_dim),
        )
        self.dec_fc    = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.2),            nn.Linear(512, 256 * 25),
        )
        self.dec_conv1 = nn.ConvTranspose1d(256, 256, kernel_size=3, padding=1)
        self.dec_bn1   = nn.BatchNorm1d(256)
        self.dec_conv2 = nn.ConvTranspose1d(512, 128, kernel_size=5, stride=5)
        self.dec_bn2   = nn.BatchNorm1d(128)
        self.dec_conv3 = nn.ConvTranspose1d(128, 128, kernel_size=5, padding=2)
        self.dec_bn3   = nn.BatchNorm1d(128)
        self.dec_conv4 = nn.ConvTranspose1d(256, 64,  kernel_size=5, stride=5)
        self.dec_bn4   = nn.BatchNorm1d(64)
        self.dec_conv5 = nn.ConvTranspose1d(64,  64,  kernel_size=7, padding=3)
        self.dec_bn5   = nn.BatchNorm1d(64)
        self.dec_conv6 = nn.ConvTranspose1d(128, 1,   kernel_size=4, stride=4)

    def encode(self, x):
        x  = x.transpose(1, 2).contiguous()
        s1 = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x  = self.enc_pool1(F.relu(self.enc_bn2(self.enc_conv2(s1))))
        s2 = F.relu(self.enc_bn3(self.enc_conv3(x)))
        x  = self.enc_pool2(F.relu(self.enc_bn4(self.enc_conv4(s2))))
        s3 = F.relu(self.enc_bn5(self.enc_conv5(x)))
        x  = self.enc_pool3(F.relu(self.enc_bn6(self.enc_conv6(s3))))
        z  = self.enc_fc(self.global_avg(x).squeeze(-1))
        return z, s1, s2, s3

    def decode(self, z, skip1, skip2, skip3):
        x = self.dec_fc(z).view(-1, 256, 25)
        x = F.relu(self.dec_bn1(self.dec_conv1(x)))
        x = torch.cat([x, F.interpolate(skip3, size=x.size(2), mode='linear', align_corners=False)], dim=1)
        x = F.dropout(F.relu(self.dec_bn2(self.dec_conv2(x))), p=0.1, training=self.training)
        x = F.relu(self.dec_bn3(self.dec_conv3(x)))
        x = torch.cat([x, F.interpolate(skip2, size=x.size(2), mode='linear', align_corners=False)], dim=1)
        x = F.dropout(F.relu(self.dec_bn4(self.dec_conv4(x))), p=0.1, training=self.training)
        x = F.relu(self.dec_bn5(self.dec_conv5(x)))
        x = torch.cat([x, F.interpolate(skip1, size=x.size(2), mode='linear', align_corners=False)], dim=1)
        return self.dec_conv6(x).transpose(1, 2).contiguous()

    def forward(self, x):
        z, s1, s2, s3 = self.encode(x)
        return z, self.decode(z, s1, s2, s3)


class ConvFormerBlock(nn.Module):
    """Depthwise conv + multi-head self-attention + FFN."""
    def __init__(self, dim: int, kernel_size: int = 9, dropout: float = 0.2):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=kernel_size // 2),
            nn.BatchNorm1d(dim), nn.SiLU(), nn.Conv1d(dim, dim, 1),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.depthwise_conv(x.transpose(1, 2)).transpose(1, 2)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class FreqEncoder(nn.Module):
    """
    Frequency-domain encoder: stem conv → ConvFormerBlocks → global avg pool.
    Input  : (B, F, 1)
    Output : (B, 128)
    """
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(),
        )
        self.blocks     = nn.ModuleList([ConvFormerBlock(hidden_dim) for _ in range(num_layers)])
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.out_proj   = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        x = self.stem(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(self.global_avg(x.transpose(1, 2)).squeeze(-1))


class PatientEncoder(nn.Module):
    """
    Input  : (B, 4)  [Age, Sex, Ht, Wt]
    Output : (B, 64)
    """
    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64),        nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ModalFusion(nn.Module):
    """
    Transformer encoder over 5 modality tokens.
    Input  : (B, 5, 128)
    Output : (B, 5, 128)
    """
    def __init__(self, token_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        single_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads, dim_feedforward=512,
            dropout=0.1, activation='relu', batch_first=True,
        )
        self.layers = nn.ModuleList([copy.deepcopy(single_layer) for _ in range(num_layers)])
        self.norm   = nn.LayerNorm(token_dim)

    def forward(self, tokens, return_attn: bool = False):
        x, attn_weights = tokens, []
        for layer in self.layers:
            attn_out, weight = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
            x = layer.norm1(x + layer.dropout1(attn_out))
            x = layer.norm2(x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))))
            if return_attn:
                attn_weights.append(weight.detach().cpu())
        x = self.norm(x)
        return (x, attn_weights) if return_attn else x


class COPredNet(nn.Module):
    """
    End-to-End multi-modal multi-task CO prediction network.

    Inputs
        ecg_time, ppg_time : (B, T, 1)
        ecg_freq, ppg_freq : (B, F, 1)
        patient_info       : (B, 4)
    Outputs
        pred_co   : (B, 1)
        ecg_recon : (B, T, 1)
        ppg_recon : (B, T, 1)
    """
    TOKEN_DIM = 128

    def __init__(self, signal_length: int = 2500, latent_dim: int = 512):
        super().__init__()
        self.ecg_time_enc  = SignalAutoEncoder(signal_length, latent_dim)
        self.ppg_time_enc  = SignalAutoEncoder(signal_length, latent_dim)
        self.ecg_freq_enc  = FreqEncoder(hidden_dim=128, num_layers=3)
        self.ppg_freq_enc  = FreqEncoder(hidden_dim=128, num_layers=3)
        self.patient_enc   = PatientEncoder(input_dim=4)

        self.ecg_time_proj = nn.Linear(latent_dim, self.TOKEN_DIM)
        self.ppg_time_proj = nn.Linear(latent_dim, self.TOKEN_DIM)
        self.patient_proj  = nn.Linear(64, self.TOKEN_DIM)

        self.ecg_time_norm = nn.LayerNorm(self.TOKEN_DIM)
        self.ppg_time_norm = nn.LayerNorm(self.TOKEN_DIM)
        self.ecg_freq_norm = nn.LayerNorm(self.TOKEN_DIM)
        self.ppg_freq_norm = nn.LayerNorm(self.TOKEN_DIM)
        self.patient_norm  = nn.LayerNorm(self.TOKEN_DIM)

        self.fusion    = ModalFusion(token_dim=self.TOKEN_DIM, num_heads=4, num_layers=2)
        self.pred_head = nn.Sequential(
            nn.Linear(self.TOKEN_DIM, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),             nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, ecg_time, ppg_time, ecg_freq, ppg_freq, patient_info, return_attn=False):
        ecg_z, ecg_recon = self.ecg_time_enc(ecg_time)
        ppg_z, ppg_recon = self.ppg_time_enc(ppg_time)
        ecg_freq_feat    = self.ecg_freq_enc(ecg_freq)
        ppg_freq_feat    = self.ppg_freq_enc(ppg_freq)
        patient_feat     = self.patient_enc(patient_info)

        tok_ecg_time = self.ecg_time_norm(self.ecg_time_proj(ecg_z))
        tok_ppg_time = self.ppg_time_norm(self.ppg_time_proj(ppg_z))
        tok_ecg_freq = self.ecg_freq_norm(ecg_freq_feat)
        tok_ppg_freq = self.ppg_freq_norm(ppg_freq_feat)
        tok_patient  = self.patient_norm(self.patient_proj(patient_feat))

        tokens = torch.stack(
            [tok_ecg_time, tok_ppg_time, tok_ecg_freq, tok_ppg_freq, tok_patient], dim=1
        )

        if return_attn:
            fused, attn_weights = self.fusion(tokens, return_attn=True)
            return self.pred_head(fused.mean(dim=1)), ecg_recon, ppg_recon, attn_weights

        fused = self.fusion(tokens)
        return self.pred_head(fused.mean(dim=1)), ecg_recon, ppg_recon
