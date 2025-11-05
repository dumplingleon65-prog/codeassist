# LSTM backbone with local mixing layers before and after (for per-line features). Inspired by AssistanceZero's architecture.
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LSTMBackboneConfig:
    d_in: int  # per-line input dim from featurizer
    d_model: int  # per-line output dim to heads
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.1
    surround_layers: int = 2  # local mixing layers before/after RNN
    kernel_size: int = 3  # odd; mixes across nearby lines


class SurroundBlock(nn.Module):
    """Local mixing across lines: depthwise-separable conv over line axis + residual."""

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, groups=channels
        )
        self.pw = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_bhk: torch.Tensor) -> torch.Tensor:
        y = x_bhk.transpose(1, 2)  # (B,K,H)
        y = self.dw(y)
        y = F.relu_(y)
        y = self.pw(y)
        y = y.transpose(1, 2)  # (B,H,K)
        y = self.norm(y)
        y = self.drop(y)
        return x_bhk + y


class LSTMBackbone(nn.Module):
    """
    AssistanceZero-inspired: in_proj -> [Surround]* -> BiLSTM(stack) -> [Surround]* -> out_proj
    """

    def __init__(self, cfg: LSTMBackboneConfig):
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

        hidden_lstm = H // 2
        self.lstm = nn.LSTM(
            input_size=H,
            hidden_size=hidden_lstm,
            num_layers=cfg.layers,
            batch_first=True,
            bidirectional=True,
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
        """
        x: (B, H, d_in) with H==h_max (padded if needed)
        returns (B, H, d_model), None
        """
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
        # Pre LSTM local mixing with masking per mixing stage
        for blk in self.pre:
            y = blk(y)
            y = y * mask
        # LSTM with masking after
        y, _ = self.lstm(y)
        y = y * mask
        # Post LSTM local mixing with masking per mixing stage
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
