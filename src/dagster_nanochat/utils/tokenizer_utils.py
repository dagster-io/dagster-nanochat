"""Tokenizer utilities for training."""

import tiktoken

from dagster_nanochat.defs.config import CONVERSATION_SPECIAL_TOKENS
from dagster_nanochat.nanochat.tokenizer import RustBPETokenizer


def create_tokenizer_with_special_tokens(
    tokenizer_data: dict, base_vocab_size: int
) -> RustBPETokenizer:
    """
    Create a RustBPETokenizer with conversation special tokens.

    This extends the base tokenizer vocabulary with special tokens used
    for conversation rendering (user/assistant markers, etc.).

    Args:
        tokenizer_data: Dictionary with 'mergeable_ranks' and 'pattern'
        base_vocab_size: Base vocabulary size for special token offset

    Returns:
        RustBPETokenizer configured with special tokens
    """
    mergeable_ranks = {
        bytes(item["bytes"]): item["token_id"]
        for item in tokenizer_data["mergeable_ranks"]
    }
    pattern = tokenizer_data["pattern"]

    # Map special tokens to IDs starting after base vocab
    special_tokens = {
        token: base_vocab_size + i
        for i, token in enumerate(CONVERSATION_SPECIAL_TOKENS)
    }

    tokenizer_enc = tiktoken.Encoding(
        name="training_tokenizer",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    return RustBPETokenizer(tokenizer_enc, "<|bos|>")
