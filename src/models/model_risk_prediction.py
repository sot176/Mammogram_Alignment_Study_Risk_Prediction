import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousPosEncoding(nn.Module):
    def __init__(self, dim, drop=0.1, maxtime=5, num_steps=100):
        """
        Continuous sinusoidal positional encoding with linear interpolation over time.

        Args:
            dim (int): Dimension of the encoding.
            drop (float): Dropout rate.
            maxtime (float): Maximum time value for normalization.
            num_steps (int): Number of discrete time steps for encoding table.
        """
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.maxtime = maxtime
        self.num_steps = num_steps

        # Precompute sinusoidal encodings
        position = torch.linspace(0, maxtime, steps=num_steps).unsqueeze(1)  # (S, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe = torch.zeros(num_steps, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, xs, times):
        """
        Args:
            xs (Tensor): Input tensor of shape (N, B, C).
            times (Tensor): Time values of shape (B,).

        Returns:
            Tensor: Time-encoded input of shape (N, B, C).
        """
        times = torch.clamp(times, 0, self.maxtime) * (self.num_steps - 1) / self.maxtime
        t_floor = torch.floor(times).long()
        t_ceil = torch.ceil(times).long()
        alpha = (times - t_floor).unsqueeze(1)  # (B, 1)

        # Linear interpolation
        pe_floor = self.pe[t_floor]  # (B, C)
        pe_ceil = self.pe[t_ceil]    # (B, C)
        pe_interp = (1 - alpha) * pe_floor + alpha * pe_ceil  # (B, C)

        return self.dropout(xs + pe_interp.unsqueeze(0))  # (N, B, C)

class CumulativeProbabilityLayer(nn.Module):
    def __init__(self, num_features, max_followup):
        """
        Predict cumulative cancer probabilities via time-dependent hazard estimation.

        Args:
            num_features (int): Feature size from the model.
            max_followup (int): Number of follow-up years (prediction steps).
        """
        super().__init__()
        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)

        # Lower-triangular mask (T x T)
        mask = torch.tril(torch.ones(max_followup, max_followup)).T
        self.register_buffer("upper_triangular_mask", mask)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input features of shape (B, C)

        Returns:
            Tensor: Cumulative probability over time (B, T)
        """
        B = x.size(0)
        raw_hazards = self.relu(self.hazard_fc(x))  # (B, T)
        base_hazard = self.base_hazard_fc(x)        # (B, 1)

        expanded = raw_hazards.unsqueeze(-1).expand(B, -1, raw_hazards.size(1))  # (B, T, T)
        masked = expanded * self.upper_triangular_mask  # (B, T, T)

        cum_probs = masked.sum(dim=1) + base_hazard  # (B, T)
        return cum_probs


class TemporalAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Multi-head self-attention with residual connection and feedforward
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + attn_output)
        return self.norm(x + self.ff(x))


class RiskModelWithAttention(nn.Module):
    def __init__(self, num_years=5, time_encoding_dim=512):
        super().__init__()
        self.positional_encoding = ContinuousPosEncoding(dim=time_encoding_dim)
        self.attention_layer = TemporalAttentionLayer(dim=512, num_heads=8)
        self.feature_projection = nn.Linear(512, 512)

        # Prediction heads
        self.cumulative_prob_layer_fused = CumulativeProbabilityLayer(512, num_years)
        self.cumulative_prob_layer_cur = CumulativeProbabilityLayer(512, num_years)
        self.cumulative_prob_layer_pri = CumulativeProbabilityLayer(512, num_years)

    def forward(self, f_cur, f_pri, f_pri_aligned, f_dif, time_gap):
        """
        Args:
            f_cur: Current image features (B, C, H, W)
            f_pri: Prior image features (B, C, H, W)
            f_pri_aligned: Spatially aligned prior features (B, C, H, W)
            f_dif: Difference map (B, C, H, W)
            time_gap: Time gap (tensor) (B, 1)
        """
        B, C, H, W = f_dif.shape

        # Time-aware encoding of difference map
        flattened_feats = f_dif.flatten(start_dim=2).permute(2, 0, 1)  # [N, B, C]
        fdif_with_time = self.positional_encoding(flattened_feats, time_gap)
        fdif_with_time = fdif_with_time.permute(1, 2, 0).view(B, C, H, W)

        # Global average pooling
        f_cur_pooled = F.adaptive_avg_pool2d(f_cur, (1, 1)).view(B, C)
        f_pri_pooled = F.adaptive_avg_pool2d(f_pri, (1, 1)).view(B, C)
        f_pri_aligned_pooled = F.adaptive_avg_pool2d(f_pri_aligned, (1, 1)).view(B, C)
        fdif_pooled = F.adaptive_avg_pool2d(fdif_with_time, (1, 1)).view(B, C)

        # Temporal attention
        stacked = torch.stack([f_pri_aligned_pooled, fdif_pooled, f_cur_pooled], dim=1)  # [B, 3, C]
        attended = self.attention_layer(stacked.permute(1, 0, 2))  # [3, B, C]
        attended = attended.permute(1, 0, 2).mean(dim=1)  # [B, C]
        fused_feat = self.feature_projection(attended)

        return {
            "pred_fused": self.cumulative_prob_layer_fused(fused_feat),
            "pred_cur": self.cumulative_prob_layer_cur(f_cur_pooled),
            "pred_pri": self.cumulative_prob_layer_pri(f_pri_pooled),
        }


class RiskModelWithAttentionNoAlignment(nn.Module):
    def __init__(self, num_years=5, time_encoding_dim=512):
        super().__init__()
        self.positional_encoding = ContinuousPosEncoding(dim=time_encoding_dim)
        self.attention_layer = TemporalAttentionLayer(dim=512, num_heads=8)
        self.feature_projection = nn.Linear(512, 512)

        self.cumulative_prob_layer_fused = CumulativeProbabilityLayer(512, num_years)
        self.cumulative_prob_layer_cur = CumulativeProbabilityLayer(512, num_years)
        self.cumulative_prob_layer_pri = CumulativeProbabilityLayer(512, num_years)

    def forward(self, f_cur, f_pri, time_gap):
        """
        Args:
            f_cur: Current image features (B, C, H, W)
            f_pri: Prior image features (B, C, H, W)
            time_gap: Time gap (unused here)
        """
        B, C, H, W = f_pri.shape

        f_cur_pooled = F.adaptive_avg_pool2d(f_cur, (1, 1)).view(B, C)
        f_pri_pooled = F.adaptive_avg_pool2d(f_pri, (1, 1)).view(B, C)

        stacked = torch.stack([f_pri_pooled, f_cur_pooled], dim=1)  # [B, 2, C]
        attended = self.attention_layer(stacked.permute(1, 0, 2))  # [2, B, C]
        attended = attended.permute(1, 0, 2).mean(dim=1)  # [B, C]
        fused_feat = self.feature_projection(attended)

        return {
            "pred_fused": self.cumulative_prob_layer_fused(fused_feat),
            "pred_cur": self.cumulative_prob_layer_cur(f_cur_pooled),
            "pred_pri": self.cumulative_prob_layer_pri(f_pri_pooled),
        }


class RiskModelWithImplicitAlignment(nn.Module):
    def __init__(self, num_years=5, time_encoding_dim=512):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.attention_layer = TemporalAttentionLayer(dim=512, num_heads=8)
        self.feature_projection = nn.Linear(512, 512)

        self.cumulative_prob_layer_fused = CumulativeProbabilityLayer(512, num_years)
        self.cumulative_prob_layer_cur = CumulativeProbabilityLayer(512, num_years)
        self.cumulative_prob_layer_pri = CumulativeProbabilityLayer(512, num_years)

    def forward(self, f_cur, f_pri, time_gap):
        """
        Args:
            f_cur: Current features (B, C, H, W)
            f_pri: Prior features (B, C, H, W)
            time_gap: Time gap (unused)
        """
        B, _, _, _ = f_pri.shape

        # Combine and extract features
        concatenated = torch.cat([f_cur, f_pri], dim=1)
        features = self.feature_extractor(concatenated)

        # Temporal attention
        attended = self.attention_layer(features.unsqueeze(0)).squeeze(0)
        fused_feat = self.feature_projection(attended)

        # Global pooling
        cur_flat = F.adaptive_avg_pool2d(f_cur, (1, 1)).view(B, -1)
        pri_flat = F.adaptive_avg_pool2d(f_pri, (1, 1)).view(B, -1)

        return {
            "pred_fused": self.cumulative_prob_layer_fused(fused_feat),
            "pred_cur": self.cumulative_prob_layer_cur(cur_flat),
            "pred_pri": self.cumulative_prob_layer_pri(pri_flat),
        }