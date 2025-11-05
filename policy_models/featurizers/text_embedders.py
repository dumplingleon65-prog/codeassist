from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    OllamaEmbedderConfig,
    MLPEmbedderConfig,
    TrainableMLPConfig,
    CharCNNConfig,
)
import ollama

logger = logging.getLogger(__name__)


# --- Frozen text embedders served via Ollama API ---
class OllamaEmbedder(nn.Module):
    """Ollama-based text embedder with a trainable projection head."""

    def __init__(self, cfg: OllamaEmbedderConfig = OllamaEmbedderConfig()):
        super().__init__()
        self.model = cfg.model
        self.model_embed_dim = cfg.model_embed_dim
        self.output_dim = cfg.output_dim

        logger.info(f"Initialized Ollama embedder with model: {self.model}")

        self.text_mlp = nn.Sequential(
            nn.Linear(self.model_embed_dim, self.output_dim), nn.GELU()
        )

    @torch.no_grad()
    def _fetch_ollama_embedding(self, text: str, device: torch.device) -> torch.Tensor:
        resp = ollama.embed(model=self.model, input=text)
        vec = resp.embeddings[0]
        return torch.tensor(vec, dtype=torch.float32, device=device)

    def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        try:
            device = next(self.parameters()).device
            base = self._fetch_ollama_embedding(text, device=device)
            out = self.text_mlp(base)
            return out
        except Exception as e:
            logger.error(f"Error getting embedding from Ollama: {e}")
            raise e

    forward = get_embedding


# --- Frozen MLP-based text embedder ---
class MLPEmbedder(nn.Module):
    """MLP-based text embedder using character-level embeddings."""

    def __init__(self, cfg: MLPEmbedderConfig = MLPEmbedderConfig()):
        super().__init__()
        self.cfg = cfg
        self.model_embed_dim = cfg.model_embed_dim
        self.output_dim = cfg.output_dim
        self.max_length = cfg.max_length

        self.char_embed = nn.Embedding(256, self.model_embed_dim)
        self.text_mlp = nn.Sequential(
            nn.Linear(self.model_embed_dim, self.output_dim), nn.GELU()
        )
        logger.info(
            f"Initialized MLP embedder with char_dim: {self.model_embed_dim}, output_dim: {self.output_dim}"
        )

    def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Get embedding using character-level MLP approach."""
        try:
            text_bytes = text.encode("utf-8")[: self.max_length] or b" "
            ids = torch.tensor(list(text_bytes), dtype=torch.long).clamp(0, 255)

            char_embeddings = self.char_embed(ids)
            pooled_embedding = char_embeddings.mean(dim=0)
            text_embedding = self.text_mlp(pooled_embedding)

            return text_embedding
        except Exception as e:
            logger.error(f"Error getting MLP embedding: {e}")
            raise e

    forward = get_embedding


# --- Trainable MLP text embedder ---
class TrainableMLPTextEmbedder(nn.Module):
    """
    Line-level embedder:
      - one-hot / count over ASCII chars (length capped at w_max)
      - 2-layer MLP -> output_dim
    """

    def __init__(self, cfg: TrainableMLPConfig = TrainableMLPConfig()):
        super().__init__()
        self.cfg = cfg
        D, H = cfg.vocab_size, cfg.hidden
        self.lin1 = nn.Linear(D, H)
        self.lin2 = nn.Linear(H, cfg.output_dim)
        self.norm = nn.LayerNorm(H)
        self.drop = nn.Dropout(cfg.dropout)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    @torch.no_grad()
    def _counts(self, lines: list[str], w_max: int) -> torch.Tensor:
        # counts per ASCII char (B, 128)
        B = len(lines)
        M = torch.zeros(B, self.cfg.vocab_size)
        for i, s in enumerate(lines):
            for ch in s[:w_max]:
                c = ord(ch)
                if 0 <= c < self.cfg.vocab_size:
                    M[i, c] += 1.0
        return M

    def forward(
        self, lines: list[str], w_max: int, device: torch.device
    ) -> torch.Tensor:
        X = self._counts(lines, w_max).to(device)
        h = self.lin1(X)
        h = F.relu_(h)
        h = self.norm(h)
        h = self.drop(h)
        z = self.lin2(h)
        return z  # (B, output_dim)

    def get_embedding(self, text: str, w_max: int) -> torch.Tensor:
        try:
            return self.forward(
                [text], w_max=w_max, device=next(self.parameters()).device
            )
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise e


# --- Trainable Char CNN ---
class CharCNNTextEmbedder(nn.Module):
    """
    Line-level char CNN:
      - Embedding over ASCII chars
      - Multi-kernel 1D conv + global max pool
      - Concatenate and project to output_dim
    """

    def __init__(self, cfg: CharCNNConfig = CharCNNConfig()):
        super().__init__()
        self.cfg = cfg
        V = cfg.vocab_size
        E = cfg.model_embed_dim
        C = cfg.channels
        self.emb = nn.Embedding(V, E)
        self.cnns = nn.ModuleList(
            [nn.Conv1d(E, C, k, padding=k // 2) for k in cfg.kernels]
        )
        self.proj = nn.Linear(C * len(cfg.kernels), cfg.output_dim)
        self.drop = nn.Dropout(cfg.dropout)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @torch.no_grad()
    def _encode_ascii(self, lines: list[str], w_max: int) -> torch.Tensor:
        B = len(lines)
        X = torch.zeros(B, w_max, dtype=torch.long)
        for i, s in enumerate(lines):
            for j, ch in enumerate(s[:w_max]):
                X[i, j] = min(max(ord(ch), 0), self.cfg.vocab_size - 1)
        return X  # (B, w_max)

    def forward(
        self, lines: list[str], w_max: int, device: torch.device
    ) -> torch.Tensor:
        X = self._encode_ascii(lines, w_max).to(device)  # (B, L)
        T = self.emb(X).transpose(1, 2)  # (B, E, L)
        feats = []
        for conv in self.cnns:
            y = torch.tanh(conv(T))  # (B, C, L)
            y = torch.max(y, dim=-1).values  # (B, C)
            feats.append(y)
        H = torch.cat(feats, dim=-1)  # (B, C*#kernels)
        H = self.drop(H)
        Z = self.proj(H)  # (B, output_dim)
        return Z

    def get_embedding(self, text: str, w_max: int) -> torch.Tensor:
        try:
            return self.forward(
                [text], w_max=w_max, device=next(self.parameters()).device
            )
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise e
