# Local-Global Transformer backbone (with optional global tokens). Heaviest variant, but better for larger files/global contexts.
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from utils.masking import build_local_global_mask


@dataclass
class TransformerBackboneConfig:
    d_in: int
    d_model: int
    n_layers: int = 4
    n_heads: int = 4
    radius: int = 32
    n_global_tokens: int = 4
    dropout: float = 0.1


class LocalGlobalBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, n_global_tokens, radius):
        super().__init__()
        self.G = n_global_tokens
        self.radius = radius
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.global_tokens = (
            nn.Parameter(torch.randn(self.G, d_model) / (d_model**0.5))
            if self.G > 0
            else None
        )

    def forward(self, x, h):
        B, h_max, d = x.shape
        tok = x[:, :h, :]
        if self.G > 0:
            g = self.global_tokens.unsqueeze(0).expand(B, -1, -1)
            seq = torch.cat([tok, g], dim=1)
        else:
            seq = tok
        L = h + self.G
        attn_mask = build_local_global_mask(h, self.G, self.radius, device=x.device)[
            :L, :L
        ]
        res = seq
        out, _ = self.attn(seq, seq, seq, attn_mask=attn_mask)
        seq = self.ln1(res + out)
        res = seq
        seq = self.ln2(res + self.ff(seq))
        if self.G > 0:
            return seq[:, :h, :], seq[:, h:, :]
        return seq, None


class LocalGlobalTransformer(nn.Module):
    def __init__(self, cfg: TransformerBackboneConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.d_in, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                LocalGlobalBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.dropout,
                    cfg.n_global_tokens,
                    cfg.radius,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, x, h):
        x = self.ln(self.proj(x))
        g = None
        for blk in self.blocks:
            x, g = blk(x, h)
        return x[:, :h, :], g
