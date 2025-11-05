# Overview
CodeAssist policy components and training utilities.

Implements:
- State adaptation and action inference: state-service output → policy-ready state representations.
- Single-state inference: 4-tuple line states → embeddings → assistant policy → (action_idx, line_idx).
- Batch training: BC (human), PPO-from-episodes (assistant), shallow & deep zero-style self-play.

## Recommended Workflow
For the majority of users, the intended entry point is the repository root `run.py` script:

1. `run.py` bootstraps containers/services (state service, solution tester, UI, etc.).
2. After environment setup and episode recording sessions, it automatically launches the main training loop implemented in `policy_models/training_loop.py`.
3. The training loop performs a two-phase run: initial BC/PPO training on collected episodes, optional zero-style recordings with a follow-up fine-tune.

Running `python run.py` therefore executes the full system workflow end-to-end, culminating in policy training without requiring manual CLI invocation. Use it unless you have a specific need to call the lower-level tools yourself.

## Key Components

- `cli/run_tasks.py`: Lightweight entrypoints for inference and direct training-from-episodes. Exposes the `infer_action` and `train_from_episodes` subcommands used by the training loop.
- `training/train_from_episodes.py`: Core training routine that wires together the featurizer, policy networks, BC trainer, PPO trainer, reward mixer, and persistence utilities.
- `training/telemetry.py`: Collects run metadata and pushes structured telemetry events after each training invocation.
- `training/training_loop.py`: Orchestrates the multi-phase training pipeline (pre-training, zero-style recording, post-training) and is invoked by `run.py`.
- `config.py`: Dataclasses describing model, PPO, search, and training hyperparameters.
- `featurizers/`: Feature extraction logic for converting textual states into tensor representations.
- `backbones/` and `models/`: Model architectures for policies and related heads.
- `rewards.py`: Reward shaping and mixer utilities used during PPO.
- `verify/`: Sanity checks and toy environments to validate training components end-to-end.

## Direct Usage (Advanced)
The modules above can be used standalone when development requires isolating individual components.

### Single-state inference
Create a JSON file with your 4-tuple per line state, e.g., `single_state.json`:
```json
{
  "t": 0,
  "line_tuples": [
    ["def two_sum(a,b):", {"t_last":-1,"span":[0,0],"flags":[0,0,0]}, {"t_last":-1,"span":[0,0],"flags":[0,0,0]}, {"on": false, "line": 0, "char": 0, "last_t": -1}],
    ["    s = a + b",     {"t_last":0,"span":[4,10],"flags":[1,0,0]}, {"t_last":-1,"span":[0,0],"flags":[0,0,0]}, {"on": false, "line": 1, "char": 0, "last_t": -1}],
    ["    return s",      {"t_last":0,"span":[4,14],"flags":[1,0,0]}, {"t_last":-1,"span":[0,0],"flags":[0,0,0]}, {"on": true,  "line": 2, "char": 12, "last_t": 0}]
  ]
}
```

Run:
```bash
python -m cli.run_tasks infer_action --state_json single_state.json --h_max 300 --device cpu
```

### Inference Arguments
- `--state_json`: Path to JSON file containing the 4-tuple per line state
- `--h_max`: Maximum number of lines to process (default: 300)
- `--device`: Device to use for inference (default: cpu)

Expected output:
```json
{
  "action_idx": <int>,
  "line_idx": <int or -1>,
  "debug": {
    "action_logits": [...],
    "line_logits_for_action": [...],
    "h": 3
  }
}
```

### Train from episodes
Prepare `episodes.json` as a list. Each episode has:
- "states": array of either 4‑tuple per line states or already canonical states (we auto‑convert 4‑tuples via state_adapter).
- Optionally "actions": array of dicts per time with "H":{"type":..., "line": ...}, "A":{...}. If missing, we infer actions from last‑action attribution deltas.

Example episode as input:
```json
[
  {
    "states": [
      [
        ["def f(x):", {"t_last":-1,"span":[0,0],"flags":[0,0,0]}, {"t_last":-1,"span":[0,0],"flags":[0,0,0]}, {"on": true, "line":1,"char":8,"last_t":-1}],
        ["    pass",   {"t_last":-1,"span":[0,0],"flags":[0,0,0]}, {"t_last":-1,"span":[0,0],"flags":[0,0,0]}, {"on": false,"line":1,"char":8,"last_t":-1}]
      ],
      [
        ["def f(x):", {"t_last":-1,"span":[0,0],"flags":[0,0,0]}, {"t_last":-1,"span":[0,0],"flags":[0,0,0]}, {"on": false, "line":1,"char":8,"last_t":0}],
        ["    return x", {"t_last":0,"span":[4,14],"flags":[1,0,0]}, {"t_last":-1,"span":[0,0],"flags":[0,0,0]}, {"on": false,"line":1,"char":8,"last_t":0}]
      ]
    ]
  }
]
```

Run:
```bash
python -m cli.run_tasks train_from_episodes --episodes_dir ../persistent-data/state-service/episodes/ --h_max 300 --w_max 160 --checkpoint_dir ./_artifacts --device cpu --backbone lg_transformer --bc_epochs 1 --ppo_epochs 1 --zero_style_epochs 1 --ppo_steps 2048 --zero_roots 32 --zero_horizon 6
```

### Backbone Options
The `--backbone` argument supports three different architectures:

- **`lg_transformer`** (default): Local-Global Transformer with attention mechanisms. Best for larger contexts and files requiring global understanding.
- **`lstm`**: Bidirectional LSTM with local mixing layers. Good balance of performance and efficiency.
- **`bigru`**: Bidirectional GRU backbone. Lighter and faster variant of LSTM.

Each backbone has specific hyperparameters that are automatically configured via the respective model config classes.

### Training Arguments
- `--episodes_dir`: Directory containing episode JSON files
- `--h_max`: Maximum number of lines (default: 300)
- `--w_max`: Maximum line width (default: 160)  
- `--checkpoint_dir`: Output directory for artifacts (default: ./_artifacts_v2)
- `--pv_dir`: Persistent volume directory for models (default: ../persistent-data/trainer/models)
- `--device`: Device to use for training (default: cpu)
- `--backbone`: Model backbone architecture (default: lg_transformer)
- `--bc_epochs`: Behavioral cloning epochs (default: 1)
- `--ppo_epochs`: PPO training epochs (default: 1)
- `--zero_style_epochs`: "Anchored" zero-style self-play epochs (default: 1)
- `--ppo_steps`: PPO steps per epoch (default: 2048)
- `--zero_roots`: "Anchored" zero-style roots per epoch (default: 32)
- `--zero_horizon`: "Anchored" zero-style horizon (default: 6)

Expected console/log artifacts:
- JSONL logs written under ./_artifacts/train.jsonl with phases bc, ppo_from_episodes, zero_style.
- Checkpoints assistant_policy_e1.pt, human_policy_e1.pt saved to ./_artifacts/.

## Sanity checks
We currently check the following:
-	Common wiring mistakes (mask length vs. h_max, illegal mass leakage) don’t exist.
- Heads and backbone can learn what they see when the mapping is trivial.
- PPO loop, logprob plumbing, GAE, and optimizer actually move a policy in the right direction.

To run all test run
```bash
python -m verify.run_all --device cpu
```

Example of what "good" results look like:
```json
{
  "overall_pass": true,
  "thresholds": {
    "illegal_mass_mean_lt": 1e-6,
    "bc_action_acc_ge": 0.95,
    "bc_line_acc_ge": 0.90,
    "value_rmse_le": 0.05,
    "goal_mse_le": 1e-3,
    "ppo_post_reward_ge": 0.4
  },
  "results": {
    "shapes_masks": {"illegal_mass_mean": 0.0},
    "bc_overfit": {"action_acc": 0.99, "line_acc": 0.98},
    "value_regression": {"value_rmse": 0.01},
    "goal_regression": {"goal_mse": 0.0003},
    "ppo_toy": {"pre_reward": 0.02, "post_reward": 0.62, "passed": true}
  }
}
```
