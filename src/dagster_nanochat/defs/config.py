"""Configuration and constants for nanochat Dagster pipeline."""

import dagster as dg

# =============================================================================
# Directory Constants
# =============================================================================

# Staging directories
FILE_DIRECTORY = "data/raw"
CHECKPOINT_DIRECTORY = "data/checkpoints"
SFT_CHECKPOINT_DIRECTORY = "data/sft_checkpoints"
SFT_DATASETS_CACHE = "data/sft_datasets"
HF_DATASETS_CACHE = "data/hf_datasets_cache"
S3_BUCKET_NAME = "dagster-nanochat"
TOKENIZER_FILE = "data/tokenizer/tokenizer.json"


# =============================================================================
# Dataset Constants
# =============================================================================

# Hugging Face Karpathy datasets
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle"
SHARDS = [f"{BASE_URL}/resolve/main/shard_{i:05d}.parquet" for i in range(1823)]
TRAINING_SET = SHARDS[:-1]

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
# Stage-Specific Configurations
# =============================================================================


class BaseTrainingConfig(dg.Config):
    """
    Configuration for base model pretraining.

    Used by: model_run_config, base_model_checkpoint
    """

    model_tag: str = "d4"
    depth: int = 4
    max_seq_len: int = 1024
    num_iterations: int = 50
    target_param_data_ratio: int = 1
    device_batch_size: int = 16
    total_batch_size: int = 262144
    embedding_lr: float = 0.2
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.02
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.2
    final_lr_frac: float = 0.0
    eval_every: int = 25
    eval_tokens: int = 5 * 262144
    sample_every: int = 50


class MidtrainingConfig(dg.Config):
    """
    Configuration for midtraining.

    Used by: midtraining_run_config, midtraining_checkpoint
    """

    model_tag: str = "d4"
    max_seq_len: int = 512
    num_iterations: int = 50
    device_batch_size: int = 1
    total_batch_size: int = 512
    embedding_lr: float = 0.2
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.02
    init_lr_frac: float = 1.0
    weight_decay: float = 0.0
    eval_every: int = 25
    eval_tokens: int = 512


class SFTConfig(dg.Config):
    """
    Configuration for supervised fine-tuning.

    Used by: sft_datasets, sft_run_config, sft_checkpoint
    """

    model_tag: str = "d4"
    max_seq_len: int = 512
    num_epochs: int = 1
    device_batch_size: int = 16
    target_examples_per_step: int = 32
    embedding_lr: float = 0.2
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.02
    init_lr_frac: float = 1.0
    weight_decay: float = 0.0
    eval_every: int = 25
    eval_steps: int = 20


# Backwards compatibility alias (for any existing code)
NanochatConfig = BaseTrainingConfig


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
