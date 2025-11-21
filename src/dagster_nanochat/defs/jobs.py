import dagster as dg

from dagster_nanochat.defs.assets import (
    base_model_checkpoint,
    midtraining_checkpoint,
    midtraining_run_config,
    model_run_config,
    sft_checkpoint,
    sft_datasets,
    sft_run_config,
    tokenizer_training,
    training_files,
)

# =============================================================================
# Configuration Presets
# =============================================================================

# $1 TIER: E2E Testing (~10 minutes total on 2×A40)
# Cost: Base (5 min) + Mid (3 min) + SFT (2 min) ≈ $0.80
CONFIG_1_DOLLAR = {
    "quick_mode": True,
    "model_tag": "d4",
    "depth": 4,
    "max_seq_len": 1024,
    "num_iterations": 50,
    "target_param_data_ratio": 1,
    "device_batch_size": 16,
    "total_batch_size": 262144,
    "target_examples_per_step": 32,
    "num_epochs": 1,
    "embedding_lr": 0.2,
    "unembedding_lr": 0.004,
    "matrix_lr": 0.02,
    "init_lr_frac": 1.0,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "warmup_ratio": 0.0,
    "warmdown_ratio": 0.2,
    "final_lr_frac": 0.0,
    "eval_every": 25,
    "eval_tokens": 5 * 262144,
    "sample_every": 50,
    "eval_steps": 20,
}

# $10 TIER: Basic Training (~1 hour total on 2×A40)
# Cost: Base (35 min) + Mid (15 min) + SFT (10 min) ≈ $9.60
CONFIG_10_DOLLAR = {
    "quick_mode": False,
    "model_tag": "d12",
    "depth": 12,
    "max_seq_len": 2048,
    "num_iterations": -1,
    "target_param_data_ratio": 5,
    "device_batch_size": 24,
    "total_batch_size": 393216,
    "target_examples_per_step": 32,
    "num_epochs": 1,
    "embedding_lr": 0.2,
    "unembedding_lr": 0.004,
    "matrix_lr": 0.02,
    "init_lr_frac": 1.0,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "warmup_ratio": 0.0,
    "warmdown_ratio": 0.2,
    "final_lr_frac": 0.0,
    "eval_every": 200,
    "eval_tokens": 10 * 524288,
    "sample_every": 1000,
    "eval_steps": 50,
}

# $100 TIER: Full Training (~8 hours total on 2×A40)
# Cost: Base (5 hrs) + Mid (2 hrs) + SFT (1 hr) ≈ $96
CONFIG_100_DOLLAR = {
    "quick_mode": False,
    "model_tag": "d20",
    "depth": 20,
    "max_seq_len": 2048,
    "num_iterations": -1,
    "target_param_data_ratio": 20,
    "device_batch_size": 32,
    "total_batch_size": 524288,
    "target_examples_per_step": 32,
    "num_epochs": 1,
    "embedding_lr": 0.2,
    "unembedding_lr": 0.004,
    "matrix_lr": 0.02,
    "init_lr_frac": 1.0,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "warmup_ratio": 0.0,
    "warmdown_ratio": 0.2,
    "final_lr_frac": 0.0,
    "eval_every": 250,
    "eval_tokens": 20 * 524288,
    "sample_every": 2000,
    "eval_steps": 100,
}


# =============================================================================
# Training Jobs
# =============================================================================

nanochat_1_dollar = dg.define_asset_job(
    name="nanochat_1_dollar",
    description="$1 Tier: E2E testing pipeline (~10 min, 4-layer model)",
    selection=[
        training_files,
        tokenizer_training,
        model_run_config,
        base_model_checkpoint,
        midtraining_run_config,
        midtraining_checkpoint,
        sft_datasets,
        sft_run_config,
        sft_checkpoint,
    ],
    config={
        "ops": {
            "model_run_config": {"config": CONFIG_1_DOLLAR},
            "base_model_checkpoint": {"config": CONFIG_1_DOLLAR},
            "midtraining_run_config": {"config": CONFIG_1_DOLLAR},
            "midtraining_checkpoint": {"config": CONFIG_1_DOLLAR},
            "sft_datasets": {"config": CONFIG_1_DOLLAR},
            "sft_run_config": {"config": CONFIG_1_DOLLAR},
            "sft_checkpoint": {"config": CONFIG_1_DOLLAR},
        }
    },
)

nanochat_10_dollar = dg.define_asset_job(
    name="nanochat_10_dollar",
    description="$10 Tier: Basic training pipeline (~1 hr, 12-layer model)",
    selection=[
        training_files,
        tokenizer_training,
        model_run_config,
        base_model_checkpoint,
        midtraining_run_config,
        midtraining_checkpoint,
        sft_datasets,
        sft_run_config,
        sft_checkpoint,
    ],
    config={
        "ops": {
            "model_run_config": {"config": CONFIG_10_DOLLAR},
            "base_model_checkpoint": {"config": CONFIG_10_DOLLAR},
            "midtraining_run_config": {"config": CONFIG_10_DOLLAR},
            "midtraining_checkpoint": {"config": CONFIG_10_DOLLAR},
            "sft_datasets": {"config": CONFIG_10_DOLLAR},
            "sft_run_config": {"config": CONFIG_10_DOLLAR},
            "sft_checkpoint": {"config": CONFIG_10_DOLLAR},
        }
    },
)

nanochat_100_dollar = dg.define_asset_job(
    name="nanochat_100_dollar",
    description="$100 Tier: Full training pipeline (~8 hrs, 20-layer model)",
    selection=[
        training_files,
        tokenizer_training,
        model_run_config,
        base_model_checkpoint,
        midtraining_run_config,
        midtraining_checkpoint,
        sft_datasets,
        sft_run_config,
        sft_checkpoint,
    ],
    config={
        "ops": {
            "model_run_config": {"config": CONFIG_100_DOLLAR},
            "base_model_checkpoint": {"config": CONFIG_100_DOLLAR},
            "midtraining_run_config": {"config": CONFIG_100_DOLLAR},
            "midtraining_checkpoint": {"config": CONFIG_100_DOLLAR},
            "sft_datasets": {"config": CONFIG_100_DOLLAR},
            "sft_run_config": {"config": CONFIG_100_DOLLAR},
            "sft_checkpoint": {"config": CONFIG_100_DOLLAR},
        }
    },
)
