"""Configuration and constants for nanochat Dagster pipeline."""

import dagster as dg

from dagster_nanochat.tasks.arc import ARC
from dagster_nanochat.tasks.gsm8k import GSM8K
from dagster_nanochat.tasks.mmlu import MMLU
from dagster_nanochat.tasks.smoltalk import SmolTalk
from dagster_nanochat.tasks.spellingbee import SimpleSpelling, SpellingBee

# =============================================================================
# Directory Constants
# =============================================================================

# Staging directories
FILE_DIRECTORY = "data/raw"
VALIDATION_DIRECTORY = "data/validation"
CHECKPOINT_DIRECTORY = "data/checkpoints"
MIDTRAIN_CHECKPOINT_DIRECTORY = "data/mid_checkpoints"
SFT_CHECKPOINT_DIRECTORY = "data/sft_checkpoints"
MIDTRAIN_DATASETS_CACHE = "data/mid_datasets"
SFT_DATASETS_CACHE = "data/sft_datasets"
HF_DATASETS_CACHE = "data/hf_datasets_cache"
S3_BUCKET_NAME = "dagster-nanochat"

# =============================================================================
# Dataset Constants
# =============================================================================

# Hugging Face Karpathy datasets
MAX_SHARDS = 1823
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle"
SHARDS = [f"{BASE_URL}/resolve/main/shard_{i:05d}.parquet" for i in range(MAX_SHARDS)]
TRAINING_SET = SHARDS[:-1]
VALIDATION_SET = SHARDS[-1:]

# Special tokens for conversation rendering
CONVERSATION_SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

# =============================================================================
# Midtraining Dataset Configurations
# =============================================================================

# Define midtraining dataset partitions with configurations
MIDTRAINING_DATASET_CONFIGS = {
    "smoltalk_train": {
        "class": SmolTalk,
        "quick_mode_kwargs": {"split": "train", "stop": 1000},
        "full_mode_kwargs": {"split": "train"},
    },
    "smoltalk_val": {
        "class": SmolTalk,
        "quick_mode_kwargs": {"split": "test", "stop": 100},
        "full_mode_kwargs": {"split": "test"},
    },
    "mmlu_train": {
        "class": MMLU,
        "quick_mode_kwargs": {
            "subset": "auxiliary_train",
            "split": "train",
            "stop": 500,
        },
        "full_mode_kwargs": {"subset": "auxiliary_train", "split": "train"},
    },
    "mmlu_val": {
        "class": MMLU,
        "quick_mode_kwargs": {"subset": "all", "split": "test", "stop": 100},
        "full_mode_kwargs": {"subset": "all", "split": "test", "stop": 5200},
    },
    "gsm8k_train": {
        "class": GSM8K,
        "quick_mode_kwargs": None,  # Skip in quick mode
        "full_mode_kwargs": {"subset": "main", "split": "train"},
    },
    "gsm8k_val": {
        "class": GSM8K,
        "quick_mode_kwargs": None,  # Skip in quick mode
        "full_mode_kwargs": {"subset": "main", "split": "test", "stop": 420},
    },
    "simple_spelling_train": {
        "class": SimpleSpelling,
        "quick_mode_kwargs": None,  # Skip in quick mode
        "full_mode_kwargs": {"size": 200000, "split": "train"},
    },
    "spelling_bee_train": {
        "class": SpellingBee,
        "quick_mode_kwargs": None,  # Skip in quick mode
        "full_mode_kwargs": {"size": 80000, "split": "train"},
    },
}

# Partition keys for Dagster
MIDTRAINING_DATASETS = list(MIDTRAINING_DATASET_CONFIGS.keys())


# =============================================================================
# Unified Configuration
# =============================================================================


class NanochatConfig(dg.Config):
    """
    Unified configuration for all nanochat training and inference assets.

    Different assets use different subsets of these parameters.
    All parameters have sensible defaults for quick testing.

    Asset usage guide:
    - midtraining_checkpoint, sft_datasets: quick_mode
    - base_model_checkpoint: quick_mode, depth, max_seq_len, num_iterations,
      target_param_data_ratio, device_batch_size, total_batch_size, embedding_lr,
      unembedding_lr, weight_decay, matrix_lr, grad_clip, warmup_ratio,
      warmdown_ratio, final_lr_frac, eval_every, eval_tokens, sample_every, model_tag
    - midtraining_checkpoint: quick_mode, model_tag, num_iterations, max_seq_len,
      device_batch_size, total_batch_size, unembedding_lr, embedding_lr, matrix_lr,
      init_lr_frac, weight_decay, eval_every, eval_tokens
    - sft_checkpoint: model_tag, quick_mode, num_epochs, device_batch_size,
      target_examples_per_step, unembedding_lr, embedding_lr, matrix_lr,
      weight_decay, init_lr_frac, eval_every, eval_steps
    """

    # === Global Settings ===
    quick_mode: bool = True  # Fast GPU training mode: 4-layer model, reduced iterations (~5-10min on GPU)
    model_tag: str = "d4"  # Model identifier (e.g., "d4", "d12", "d20")

    # === Model Architecture (base training only) ===
    depth: int = 12  # Transformer depth (set to 4 when quick_mode=True, 12 when quick_mode=False)
    max_seq_len: int = 2048  # Maximum sequence length

    # === Training Horizon ===
    num_iterations: int = -1  # Number of iterations (-1 = auto-calculate)
    target_param_data_ratio: int = 20  # Chinchilla optimal ratio
    num_epochs: int = 1  # Number of epochs (SFT only)

    # === Batch Sizes ===
    device_batch_size: int = 32  # Per-device batch size
    total_batch_size: int = 524288  # Total batch size in tokens
    target_examples_per_step: int = 32  # Target examples per step (SFT only)

    # === Learning Rates ===
    embedding_lr: float = 0.2  # Learning rate for embeddings (Adam)
    unembedding_lr: float = 0.004  # Learning rate for unembedding (Adam)
    matrix_lr: float = 0.02  # Learning rate for matrices (Muon)
    init_lr_frac: float = 1.0  # Initial LR fraction (mid/SFT only)

    # === Regularization ===
    weight_decay: float = 0.0  # Weight decay (Adam)
    grad_clip: float = 1.0  # Gradient clipping (base training only)

    # === Learning Rate Schedule (base training only) ===
    warmup_ratio: float = 0.0  # LR warmup ratio
    warmdown_ratio: float = 0.2  # LR warmdown ratio
    final_lr_frac: float = 0.0  # Final LR fraction

    # === Evaluation ===
    eval_every: int = 250  # Evaluate every N steps
    eval_tokens: int = 20 * 524288  # Tokens to use for validation
    eval_steps: int = 100  # Eval steps (SFT only)
    sample_every: int = 2000  # Sample from model every N steps (base only)


class ChatInferenceConfig(dg.Config):
    """
    Configuration for the chat_inference asset.

    This config allows you to call your deployed RunPod Serverless endpoint
    for inference with your trained model. The endpoint ID comes from the
    serverless_endpoint asset.
    """

    question: str  # User question/prompt (required)

    # === Inference Parameters ===
    max_tokens: int = 256  # Maximum tokens to generate
    temperature: float = 1.0  # Sampling temperature
    top_k: int = 50  # Top-k sampling parameter
