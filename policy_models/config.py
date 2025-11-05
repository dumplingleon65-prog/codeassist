from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Base model configuration with common parameters."""

    # Common backbone parameters
    d_in: int = 192  # per-line input dim from featurizer
    d_model: int = 128  # final per-line feature dim, this is what heads ingest
    backbone: str = "lg_transformer"  # "lg_transformer" | "lstm" | "bigru"
    dropout: float = 0.1
    # Head parameters
    goal_dim: int = 8
    n_actions: int = 7
    predict_reward_delta: bool = False  # keep off by default here

    # Backbone-specific parameters (keeping here for backward compatibility) TODO: Remove when subconfigs fully in place
    # LSTM / BiGRU parameters
    hidden: int = 128
    layers: int = 2
    surround_layers: int = 2
    kernel_size: int = 3
    # Transformer parameters
    n_layers: int = 4
    n_heads: int = 4
    radius: int = 32
    n_global_tokens: int = 4


@dataclass
class LSTMModelConfig(ModelConfig):
    """LSTM-specific model configuration."""

    backbone: str = "lstm"
    # LSTM-specific parameters
    hidden: int = 128
    layers: int = 2
    surround_layers: int = 2
    kernel_size: int = 3


@dataclass
class BiGRUModelConfig(ModelConfig):
    """BiGRU-specific model configuration."""

    backbone: str = "bigru"
    # BiGRU-specific parameters
    hidden: int = 128
    layers: int = 2
    surround_layers: int = 2
    kernel_size: int = 3


@dataclass
class TransformerModelConfig(ModelConfig):
    """Transformer-specific model configuration."""

    backbone: str = "lg_transformer"
    # Transformer-specific parameters
    n_layers: int = 4
    n_heads: int = 4
    radius: int = 32
    n_global_tokens: int = 4


@dataclass
class PPOConfig:
    epochs: int = 2
    minibatch_size: int = 64
    lr: float = 3e-4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    pikl_beta: float = 0.0  # piKL vs. human-action-pred head (anchor)


@dataclass
class SearchConfig:
    depth: int = 2
    n_sims: int = 32
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    gamma: float = 0.99
    topk_lines_per_action: int = 8


@dataclass
class TrainConfig:
    h_max: int = 300
    w_max: int = 160
    device: str = "cpu"
    checkpoint_dir: str = "./_artifacts"
    pv_dir: str = "./persistent-data/trainer/models"
    tb_dir: str = ""  # if "", default to out_dir + "/tb"
    run_label: str = ""
    bc_epochs: int = 10
    ppo_epochs: int = 50
    zero_style_epochs: int = 1
    ppo_steps_per_epoch: int = 2048
    zero_style_roots_per_epoch: int = 32
    zero_style_horizon: int = 6
    save_every_epochs: int = 10
    init_from_pv: bool = False


@dataclass
class FeaturizerConfig:
    h_max: int = 300
    w_max: int = 160
    states_max: int = 1000  # TODO: Arbitrary number, needs to be adjusted later
    d_in: int = 192
    char_dim: int = 64
    tau_steps: float = 20.0
    cursor_dist_c: float = 20.0
    text_embedder_type: str = "mlp"  # "mlp" | "ollama" | "mlp_trainable" | "char_cnn"
    train_text_embedder: bool = (
        False  # if True, include featurizer params in optimizers.
    )
    train_featurizer_projector: bool = (
        False  # if True, include final linear proj in optimizers.
    )


@dataclass
class OllamaEmbedderConfig:
    model: str = "qwen2.5-coder:0.5b-base"
    model_embed_dim: int = 896
    output_dim: int = 128


@dataclass
class MLPEmbedderConfig:
    output_dim: int = 128
    model_embed_dim: int = 64
    max_length: int = 160


@dataclass
class TrainableMLPConfig:
    output_dim: int = 128
    vocab_size: int = 128  # ASCII
    hidden: int = 128
    dropout: float = 0.1


@dataclass
class CharCNNConfig:
    output_dim: int = 128
    vocab_size: int = 128  # ASCII
    model_embed_dim: int = 32
    channels: int = 64
    kernels: tuple[int, ...] = (3, 5, 7)
    dropout: float = 0.1
