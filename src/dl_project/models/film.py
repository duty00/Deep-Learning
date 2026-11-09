"""Metadata encoder and FiLM layer (PLAN_BTL.md section 5.6).

FiLM (Perez et al., AAAI 2018): F' = F * (1 + gamma(h)) + beta(h), with gamma/beta predicted per channel
from the metadata embedding h. The projection is zero-initialised, so at initialisation FiLM is the
identity and the pretrained backbone's features are unchanged; the modulation is learned from there.
Implemented by the team (no external code reused).
"""
import torch
from torch import nn


class MetaEncoder(nn.Module):
    """[numeric features ; localization embedding] -> MLP -> h (dim ``out_dim``)."""

    def __init__(self, num_numeric, num_locations, loc_dim=8, hidden_dim=64, out_dim=64, dropout=0.1):
        super().__init__()
        self.location = nn.Embedding(num_locations, loc_dim)
        self.mlp = nn.Sequential(
            nn.Linear(num_numeric + loc_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, meta_num, meta_loc):
        return self.mlp(torch.cat([meta_num, self.location(meta_loc)], dim=1))


class FiLM(nn.Module):
    """Channel-wise affine modulation of an NCHW feature map conditioned on a vector."""

    def __init__(self, cond_dim, channels):
        super().__init__()
        self.channels = channels
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features, cond):
        if features.dim() != 4 or features.shape[1] != self.channels:
            raise ValueError(f"FiLM expects NCHW with C={self.channels}, got {tuple(features.shape)}.")
        gamma, beta = self.proj(cond).to(features.dtype).chunk(2, dim=1)
        return features * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
