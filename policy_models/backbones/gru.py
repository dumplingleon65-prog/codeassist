# BiGRU backbone with local mixing layers before and after (for per-line features). Lighter variant of LSTM backbone.
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lstm import SurroundBlock  # Reused. Same as LSTM backbone's.


@dataclass
class GRUBackboneConfig:
    d_in: int
    d_model: int
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.1
    surround_layers: int = 2
    kernel_size: int = 3


class BiGRUBackbone(nn.Module):
    """Lighter variant with GRU instead of LSTM; same surround blocks."""

    def __init__(self, cfg: GRUBackboneConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden
        self.in_proj = nn.Linear(cfg.d_in, H)
        self.in_norm = nn.LayerNorm(H)
        self.pre = nn.ModuleList(
            [
                SurroundBlock(H, cfg.kernel_size, cfg.dropout)
                for _ in range(cfg.surround_layers)
            ]
        )
        hidden_gru = H // 2
        self.gru = nn.GRU(
            input_size=H,
            hidden_size=hidden_gru,
            num_layers=cfg.layers,
            bidirectional=True,
            batch_first=True,
            dropout=(cfg.dropout if cfg.layers > 1 else 0.0),
        )
        self.post = nn.ModuleList(
            [
                SurroundBlock(H, cfg.kernel_size, cfg.dropout)
                for _ in range(cfg.surround_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(H)
        self.drop = nn.Dropout(cfg.dropout)
        self.out_proj = nn.Linear(H, cfg.d_model)

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, h_used: int | None = None) -> torch.Tensor:
        B, Htot, _ = x.shape

        # Create mask for padded elements
        if isinstance(h_used, int) or h_used is None:
            L = Htot if h_used is None else int(h_used)
            m = torch.arange(Htot, device=x.device).unsqueeze(0) < L  # (1,H)
            mask = m.expand(B, Htot).unsqueeze(-1)  # (B,H,1)
        else:
            raise ValueError("h_used must be int or None")
        # Project, normalize, and mask
        y = self.in_proj(x)
        y = self.in_norm(y)
        y = y * mask
        # Pre GRU local mixing with masking per mixing stage
        for blk in self.pre:
            y = blk(y)
            y = y * mask
        # GRU with masking after
        y, _ = self.gru(y)
        y = y * mask
        # Post GRU local mixing with masking per mixing stage
        for blk in self.post:
            y = blk(y)
            y = y * mask
        # Final norm, dropout, projection, and masking
        y = self.out_norm(y)
        y = self.drop(y)
        y = self.out_proj(y)
        y = y * mask
        # Return only valid lines if h_used is given
        if h_used is not None and h_used < Htot:
            return y[:, :h_used, :], None
        return y, None  # (B, H, d_model), None because no global tokens
