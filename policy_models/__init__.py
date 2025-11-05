"""
CodeAssistZero Policy Core (focused):
- Single-state inference: 4-tuple line states -> embeddings -> assistant policy -> (action_idx, line_idx).
- Batch training: BC (human), PPO-from-episodes (assistant), shallow zero-style self-play (assistant vs human).
"""
