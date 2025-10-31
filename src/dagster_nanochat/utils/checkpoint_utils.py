"""Checkpoint loading utilities."""

import json
import os

import torch

from dagster_nanochat.nanochat.gpt import GPT, GPTConfig


def load_checkpoint(checkpoint_dir: str, device: torch.device) -> tuple[GPT, dict]:
    """
    Load model and configuration from a checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory containing:
            - checkpoint.json: Model configuration and metadata
            - model.pt: Model state dict
        device: Device to load model onto

    Returns:
        Tuple of (model, checkpoint_data) where:
        - model: Loaded GPT model on specified device
        - checkpoint_data: Full checkpoint metadata dictionary

    Raises:
        FileNotFoundError: If checkpoint files don't exist
    """
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.json")
    model_path = os.path.join(checkpoint_dir, "model.pt")

    # Load checkpoint metadata
    with open(checkpoint_path) as f:
        checkpoint_data = json.load(f)

    # Create model from config
    model_config = GPTConfig(**checkpoint_data["model_config"])
    model = GPT(model_config)

    # Load weights
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    return model, checkpoint_data
