# Line‑by‑line featurizer turning the canonical state dict into (1, h_max, D_in) with valid h.
from config import FeaturizerConfig
from dataclasses import dataclass
from typing import Dict, Any
import torch
import torch.nn as nn
from .text_embedders import (
    CharCNNTextEmbedder,
    OllamaEmbedder,
    MLPEmbedder,
    TrainableMLPTextEmbedder,
)


class LineFeaturizer(nn.Module):
    """
    Converts canonical line-by-line state into per-line features of size d_in.
    Text embedding: configurable embedder (MLP or Ollama) -> 128-D.
    Additional scalars: length, indent, attribution (H/A), cursor features.
    """

    def __init__(self, cfg: FeaturizerConfig):
        super().__init__()
        self.cfg = cfg

        # Initialize text embedder based on config type
        if cfg.text_embedder_type == "ollama":
            self.text_embedder = OllamaEmbedder()
        elif cfg.text_embedder_type == "mlp_trainable":
            self.text_embedder = TrainableMLPTextEmbedder()
        elif cfg.text_embedder_type == "char_cnn":
            self.text_embedder = CharCNNTextEmbedder()
        else:
            self.text_embedder = MLPEmbedder()

        # Gather additional params needed for certain line embedders
        if cfg.text_embedder_type in ["mlp_trainable", "char_cnn"]:
            self.line_limit = self.cfg.w_max
        else:
            self.line_limit = None  # No limit for Ollama or other embedders

        # Final projection to model d_in if needed
        self.proj = nn.Linear(148, cfg.d_in)

        # MLP is not trainable. Forcing here for consistency until we choose to remove it.
        if cfg.text_embedder_type == "mlp":
            self.cfg.train_text_embedder = False
            self.cfg.train_featurizer_projector = False

        # Determine if featurizer is trainable based on config
        if self.cfg.train_text_embedder or self.cfg.train_featurizer_projector:
            self.trainable = True
        else:
            self.trainable = False

        # Get trainable params for optimizer if needed
        self.trainable_params = self.trainable_parameters()

    def forward(self, state: Dict[str, Any], agent: str | None = None):
        lines = state.get("lines_text", [])
        h = state.get("h", len(lines))
        cur = state.get("cursor", {"on": False, "line": -1, "char": 0, "last_t": -1})
        cur_line = int(cur.get("line", -1))
        cur_char = int(cur.get("char", 0))
        t_now = int(state.get("t", 0))
        feats = []
        for i in range(h):
            txt = lines[i] if i < len(lines) else ""

            # Get text embedding using configured embedder
            if self.line_limit:
                embedding = self.text_embedder.get_embedding(txt, self.line_limit)[
                    0
                ]  # (1, D)
            else:
                embedding = self.text_embedder.get_embedding(txt)
            text_vec = embedding.to(torch.float32)

            line_len = min(len(txt.encode("utf-8")), self.cfg.w_max) / float(
                self.cfg.w_max
            )
            indent = 0
            for ch in txt:
                if ch == " ":
                    indent += 1
                elif ch == "\t":
                    indent += 4
                else:
                    break
            indent_norm = min(indent, self.cfg.w_max) / float(self.cfg.w_max)

            def pack(agent_key: str):
                la = (
                    state["line_attribs"][agent_key][i]
                    if i < len(state["line_attribs"][agent_key])
                    else {"t_last": -1, "span": (0, 0), "flags": (0, 0, 0)}
                )
                t_last = int(la.get("t_last", -1))
                t_last_norm = max(t_last, 0) / 1000.0
                decay = float(
                    torch.exp(
                        torch.tensor(-(max(t_now - t_last, 0)) / self.cfg.tau_steps)
                    )
                )
                s, e = la.get("span", (0, 0))
                s_norm = max(min(s, self.cfg.w_max), 0) / float(self.cfg.w_max)
                e_norm = max(min(e, self.cfg.w_max), 0) / float(self.cfg.w_max)
                add, dele, rep = [
                    1.0 if int(x) != 0 else 0.0 for x in la.get("flags", (0, 0, 0))
                ]
                return t_last_norm, decay, s_norm, e_norm, add, dele, rep

            tH, dH, sH, eH, addH, delH, repH = pack("H")
            tA, dA, sA, eA, addA, delA, repA = pack("A")
            cursor_on = 1.0 if cur.get("on", False) and cur_line == i else 0.0
            cursor_char_norm = (cur_char / float(self.cfg.w_max)) if cursor_on else 0.0
            cursor_last_t_norm = max(cur.get("last_t", -1), 0) / 1000.0
            dist = (
                float(torch.tanh(torch.tensor((i - cur_line) / self.cfg.cursor_dist_c)))
                if cur_line >= 0
                else 0.0
            )
            scal = torch.tensor(
                [
                    line_len,
                    indent_norm,
                    tH,
                    dH,
                    sH,
                    eH,
                    addH,
                    delH,
                    repH,
                    tA,
                    dA,
                    sA,
                    eA,
                    addA,
                    delA,
                    repA,
                    cursor_on,
                    cursor_char_norm,
                    cursor_last_t_norm,
                    dist,
                ],
                dtype=torch.float32,
            )
            feats.append(torch.cat([text_vec, scal], dim=0))
        X = torch.zeros(self.cfg.h_max, 148, dtype=torch.float32)
        if h > 0 and feats:
            X[:h] = torch.stack(feats, dim=0)
        X = self.proj(X).unsqueeze(0)  # (1,h_max,d_in)
        return X, h, None

    def featurize(self, state: Dict[str, Any], agent: str | None = None):
        if self.trainable:
            x, h, _ = self.forward(state, agent)
        else:
            with torch.no_grad():
                x, h, _ = self.forward(state, agent)
            x = x.detach()  # Detach features when featurizer is not trainable
        return x, h, _

    def to(self, device):
        self.device_ = device
        return super().to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.cfg.train_text_embedder:
            self.text_embedder.eval()
            for p in self.text_embedder.parameters():
                p.requires_grad_(False)
        if (not self.cfg.train_featurizer_projector) and not isinstance(
            self.proj, nn.Identity
        ):
            self.proj.eval()
            for p in self.proj.parameters():
                p.requires_grad_(False)
        return self

    def trainable_parameters(self):
        """Parameters to include in optimizer if training the featurizer."""
        params = []
        if self.cfg.train_text_embedder:
            params += list(self.text_embedder.parameters())
        if self.cfg.train_featurizer_projector and not isinstance(
            self.proj, nn.Identity
        ):
            params += list(self.proj.parameters())
        return params
