# Local-global attention mask for the transformer backbone
import torch


def build_local_global_mask(h: int, G: int, radius: int, device=None):
    L = h + G
    M = torch.full((L, L), float("-inf"), device=device)
    if G > 0:
        M[h:, :] = 0.0  # globals attend everywhere
    for i in range(h):
        left = max(0, i - radius)
        right = min(h - 1, i + radius)
        M[i, left : right + 1] = 0.0
        if G > 0:
            M[i, h:] = 0.0
    return M
