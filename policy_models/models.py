# Backbones + heads + PolicyNet.
# NOTE: Ensure n_actions matches our mapping; ensure mask is (B, A, h).
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from backbones.lstm import LSTMBackbone, LSTMBackboneConfig
from backbones.gru import BiGRUBackbone, GRUBackboneConfig
from backbones.lg_transformer import LocalGlobalTransformer, TransformerBackboneConfig

# ----- Policy Heads -----


class ActionHead(nn.Module):
    def __init__(self, d_model, n_actions, hidden=128):
        super().__init__()
        self.line_mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, n_actions)
        )

    def forward(self, per_line, line_mask_per_action=None):
        B, h, d = per_line.shape
        line_logits = self.line_mlp(per_line).permute(0, 2, 1)  # (B, A, h)
        if line_mask_per_action is not None:
            # Expect mask shape (B, A, h)
            if line_mask_per_action.shape != line_logits.shape:
                raise ValueError(
                    f"ActionHead mask shape {tuple(line_mask_per_action.shape)} "
                    f"!= line_logits shape {tuple(line_logits.shape)}"
                )
            line_logits = line_logits.masked_fill(~line_mask_per_action, float("-inf"))
        action_logits = torch.logsumexp(line_logits, dim=-1)  # (B, A)
        return action_logits, line_logits


class HumanActionPredictionHead(ActionHead):
    pass


class ValueHead(nn.Module):
    def __init__(self, d_model, use_globals=True, hidden=128):
        super().__init__()
        in_dim = d_model * 2 if use_globals else d_model
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.use_globals = use_globals

    def forward(self, per_line, globals_tok=None):
        mean_pool = per_line.mean(dim=1)
        max_pool, _ = per_line.max(dim=1)
        feat = torch.cat([mean_pool, max_pool], dim=-1)
        if self.use_globals and globals_tok is not None:
            g_mean = globals_tok.mean(dim=1)
            feat = 0.5 * (feat + torch.cat([g_mean, g_mean], dim=-1))
        return self.mlp(feat)


class GoalHead(nn.Module):
    def __init__(self, d_model, goal_dim, use_globals=True, hidden=128):
        super().__init__()
        in_dim = d_model * 2 if use_globals else d_model
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, goal_dim)
        )
        self.use_globals = use_globals

    def forward(self, per_line, globals_tok=None):
        mean_pool = per_line.mean(dim=1)
        max_pool, _ = per_line.max(dim=1)
        feat = torch.cat([mean_pool, max_pool], dim=-1)
        if self.use_globals and globals_tok is not None:
            g_mean = globals_tok.mean(dim=1)
            feat = 0.5 * (feat + torch.cat([g_mean, g_mean], dim=-1))
        return self.mlp(feat)


# ----- Policy -----


@dataclass
class PolicyOutputs:
    action_logits: torch.Tensor
    line_logits: torch.Tensor
    human_pred_action_logits: torch.Tensor
    human_pred_line_logits: torch.Tensor
    value: torch.Tensor
    goal_logits: torch.Tensor
    per_line: torch.Tensor


class PolicyNet(nn.Module):
    def __init__(self, h_max: int, cfg):
        super().__init__()
        self.h_max = h_max
        self.cfg = cfg
        if cfg.backbone == "lstm":
            self.backbone = LSTMBackbone(
                LSTMBackboneConfig(
                    cfg.d_in,
                    cfg.d_model,
                    cfg.hidden,
                    cfg.layers,
                    cfg.dropout,
                    cfg.surround_layers,
                    cfg.kernel_size,
                )
            )
        elif cfg.backbone == "bigru":
            self.backbone = BiGRUBackbone(
                GRUBackboneConfig(
                    cfg.d_in,
                    cfg.d_model,
                    cfg.hidden,
                    cfg.layers,
                    cfg.dropout,
                    cfg.surround_layers,
                    cfg.kernel_size,
                )
            )
        else:  # Default to LG Transformer
            self.backbone = LocalGlobalTransformer(
                TransformerBackboneConfig(
                    cfg.d_in,
                    cfg.d_model,
                    cfg.n_layers,
                    cfg.n_heads,
                    cfg.radius,
                    cfg.n_global_tokens,
                    cfg.dropout,
                )
            )

        self.action_head = ActionHead(cfg.d_model, cfg.n_actions)
        self.human_pred_head = HumanActionPredictionHead(cfg.d_model, cfg.n_actions)
        self.value_head = ValueHead(cfg.d_model)
        self.goal_head = GoalHead(cfg.d_model, cfg.goal_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        line_mask_per_action: Optional[torch.Tensor] = None,
    ) -> PolicyOutputs:
        # If a mask is provided, trust its last dimension as authoritative h.
        h_used = (
            line_mask_per_action.shape[-1] if line_mask_per_action is not None else h
        )

        # Ensure mask (if provided) matches (B,A,h_used)
        if (
            line_mask_per_action is not None
            and line_mask_per_action.shape[-1] != h_used
        ):
            raise ValueError(
                f"Mask last dim {line_mask_per_action.shape[-1]} != h_used {h_used}"
            )

        # Route through backbone(s)
        per_line, globals_tok = self.backbone(x, h_used)

        # Route through heads
        action_logits, line_logits = self.action_head(per_line, line_mask_per_action)
        hpred_action_logits, hpred_line_logits = self.human_pred_head(
            per_line, line_mask_per_action
        )
        value = self.value_head(per_line, globals_tok)
        goal_logits = self.goal_head(per_line, globals_tok)
        return PolicyOutputs(
            action_logits,
            line_logits,
            hpred_action_logits,
            hpred_line_logits,
            value,
            goal_logits,
            per_line,
        )
