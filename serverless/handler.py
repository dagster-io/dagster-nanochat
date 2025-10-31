"""
RunPod Serverless Handler for NanoChat Inference

This handler loads the trained SFT model and processes inference requests.
Optimized for fast cold starts and efficient inference.
"""

import json
from pathlib import Path

import runpod
import torch

from dagster_nanochat.nanochat.engine import Engine
from dagster_nanochat.nanochat.gpt import GPT, GPTConfig
from dagster_nanochat.utils.tokenizer_utils import create_tokenizer_with_special_tokens

# Global variables for model caching
MODEL = None
TOKENIZER = None
ENGINE = None
DEVICE = None


def load_model():
    """Load model on first request (lazy loading for faster cold start)."""
    global MODEL, TOKENIZER, ENGINE, DEVICE

    if MODEL is not None:
        return MODEL, TOKENIZER, ENGINE, DEVICE

    print("Loading model...")

    # Setup device
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.cuda.set_device(0)
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU")

    # Load checkpoint
    checkpoint_dir = Path("/workspace/checkpoint")
    checkpoint_path = checkpoint_dir / "checkpoint.json"

    with open(checkpoint_path, "r") as f:
        checkpoint_data = json.load(f)

    model_config_dict = checkpoint_data["model_config"]
    model_config = GPTConfig(**model_config_dict)

    # Initialize model
    MODEL = GPT(model_config)

    # Load weights
    model_path = checkpoint_dir / "model.pt"
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    MODEL.load_state_dict(state_dict)
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    print(
        f"Model loaded: {model_config.n_layer} layers, "
        f"{model_config.n_head} heads, {model_config.n_embd} dim, "
        f"vocab_size: {model_config.vocab_size}"
    )

    # Load tokenizer
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    with open(tokenizer_path, "r") as f:
        tokenizer_data = json.load(f)

    base_vocab_size = tokenizer_data["vocab_size"]
    TOKENIZER = create_tokenizer_with_special_tokens(tokenizer_data, base_vocab_size)

    # Create engine
    ENGINE = Engine(MODEL, TOKENIZER)

    print("Model, tokenizer, and engine ready")

    return MODEL, TOKENIZER, ENGINE, DEVICE


def handler(event):
    """
    Handle inference requests.

    Expected input format:
    {
        "input": {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ],
            "max_tokens": 100,  # Optional, default 256
            "temperature": 1.0,  # Optional, default 1.0
            "top_k": 50  # Optional, default 50
        }
    }

    Returns:
    {
        "response": "I'm doing well, thank you!",
        "tokens_generated": 8,
        "finish_reason": "eos"  # or "max_tokens"
    }
    """
    try:
        # Load model if not already loaded
        model, tokenizer, engine, device = load_model()

        # Parse input
        job_input = event.get("input", {})
        messages = job_input.get("messages", [])
        max_tokens = job_input.get("max_tokens", 256)
        temperature = job_input.get("temperature", 1.0)
        top_k = job_input.get("top_k", 50)

        if not messages:
            return {"error": "No messages provided"}

        # Validate messages format
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                return {"error": "Each message must have 'role' and 'content'"}
            if msg["role"] not in ["user", "assistant"]:
                return {
                    "error": f"Invalid role: {msg['role']}. Must be 'user' or 'assistant'"
                }

        # Ensure conversation ends with user message
        if messages[-1]["role"] != "user":
            return {"error": "Last message must be from 'user'"}

        # Render conversation to tokens
        # render_conversation expects {"messages": [...]} format and returns (tokens, mask)
        conversation = {"messages": messages}
        conversation_tokens, _ = tokenizer.render_conversation(conversation)

        # Generate response using the engine
        with torch.no_grad():
            result_tokens, _ = engine.generate_batch(
                tokens=conversation_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        # Decode the generated tokens (result is a list of token sequences)
        # generate_batch returns the FULL sequence (input + generated)
        full_tokens = result_tokens[0]  # Get first (and only) sample

        # Extract only the newly generated tokens (after the input)
        new_tokens = full_tokens[len(conversation_tokens) :]

        # Debug logging
        print(f"DEBUG: Input length: {len(conversation_tokens)}")
        print(f"DEBUG: Full output length: {len(full_tokens)}")
        print(f"DEBUG: New tokens length: {len(new_tokens)}")
        print(f"DEBUG: First 10 new tokens: {new_tokens[:10]}")
        print(
            f"DEBUG: Token types: {type(new_tokens)}, {type(new_tokens[0]) if new_tokens else 'empty'}"
        )

        # Decode the new tokens to text
        # Ensure new_tokens is a list of integers
        if not isinstance(new_tokens, list):
            new_tokens = list(new_tokens)

        response_text = tokenizer.decode(new_tokens)

        print(f"DEBUG: Decoded text length: {len(response_text)}")
        print(f"DEBUG: First 100 chars: {response_text[:100]}")

        # Clean up special tokens from the response
        response_text = response_text.replace("<|assistant_end|>", "")
        response_text = response_text.replace("<|assistant_start|>", "")
        response_text = response_text.strip()

        tokens_generated = len(new_tokens)

        # Determine finish reason
        assistant_end_token = tokenizer.encode_special("<|assistant_end|>")
        if tokens_generated >= max_tokens:
            finish_reason = "max_tokens"
        elif response_tokens and response_tokens[-1] == assistant_end_token:
            finish_reason = "eos"
        else:
            finish_reason = "stop"

        return {
            "response": response_text,
            "tokens_generated": tokens_generated,
            "finish_reason": finish_reason,
            "model_info": {
                "vocab_size": model.config.vocab_size,
                "n_layers": model.config.n_layer,
            },
        }

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    # Start the serverless worker
    print("Starting RunPod Serverless Handler")
    runpod.serverless.start({"handler": handler})
