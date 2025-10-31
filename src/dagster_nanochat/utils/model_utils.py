"""Model utilities for training."""

import dagster as dg
import torch
import torch.nn as nn

from dagster_nanochat.defs.config import CONVERSATION_SPECIAL_TOKENS
from dagster_nanochat.nanochat.gpt import GPT


def extend_model_vocab_for_special_tokens(
    model: GPT,
    base_vocab_size: int,
    device: torch.device,
    context: dg.AssetExecutionContext,
) -> int:
    """
    Extend model embeddings to accommodate conversation special tokens.

    This dynamically resizes the model's token embedding and output layers
    to support additional special tokens beyond the base vocabulary.
    New token embeddings are initialized with small random values.

    Args:
        model: The GPT model to extend
        base_vocab_size: Original vocabulary size
        device: Device to create new embeddings on
        context: Dagster context for logging

    Returns:
        Extended vocabulary size
    """
    num_special_tokens = len(CONVERSATION_SPECIAL_TOKENS)
    extended_vocab_size = base_vocab_size + num_special_tokens

    context.log.info(f"📚 Base vocab size: {base_vocab_size:,}")
    context.log.info(f"📚 Extending vocab for {num_special_tokens} special tokens...")

    # Resize token embeddings
    old_embeddings = model.transformer.wte.weight.data
    new_embeddings = nn.Embedding(
        extended_vocab_size, model.config.n_embd, device=device
    )
    new_embeddings.weight.data[:base_vocab_size] = old_embeddings
    new_embeddings.weight.data[base_vocab_size:].normal_(mean=0.0, std=0.02)
    model.transformer.wte = new_embeddings

    # Resize lm_head
    old_lm_head = model.lm_head.weight.data
    new_lm_head = nn.Linear(
        model.config.n_embd, extended_vocab_size, bias=False, device=device
    )
    new_lm_head.weight.data[:base_vocab_size] = old_lm_head
    new_lm_head.weight.data[base_vocab_size:].normal_(mean=0.0, std=0.02)
    model.lm_head = new_lm_head

    # Update model config
    model.config.vocab_size = extended_vocab_size

    context.log.info(f"✅ Extended vocab size to: {extended_vocab_size:,}")

    return extended_vocab_size
