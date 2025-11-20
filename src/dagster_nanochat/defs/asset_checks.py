import json
import os
import time

import dagster as dg
import tiktoken
import torch

from dagster_nanochat.defs.assets import (
    SFT_CHECKPOINT_DIRECTORY,
    tokenizer_training,
)
from dagster_nanochat.defs.config import TOKENIZER_FILE
from dagster_nanochat.defs.serverless_resource import ServerlessResource
from dagster_nanochat.nanochat.chat_eval import run_chat_eval
from dagster_nanochat.nanochat.engine import Engine
from dagster_nanochat.nanochat.gpt import GPT, GPTConfig
from dagster_nanochat.utils.tokenizer_utils import create_tokenizer_with_special_tokens

# Canonical tokenizer file path


# =============================================================================
# Helper Functions
# =============================================================================


def _get_canonical_tokenizer_file() -> str | None:
    """
    Get the canonical tokenizer file path.

    Returns:
        Path to tokenizer.json, or None if not found
    """
    tokenizer_path = os.path.abspath(TOKENIZER_FILE)
    if os.path.exists(tokenizer_path):
        return tokenizer_path
    return None


# =============================================================================
# Evaluation Texts
# =============================================================================

EVAL_TEXTS = {
    "news": """(Washington, D.C., July 9, 2025)- Yesterday, Mexico's National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid.""",
    "korean": """정직한 사실 위에, 공정한 시선을 더하다
Herald Korea Times

헤럴드코리아타임즈는 정치, 경제, 사회, 문화 등 한국 사회 전반의 주요 이슈를 심도 있게 다루는 종합 온라인 신문사입니다.""",
    "code": """class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256""",
    "science": """Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment–protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center.""",
}


@dg.asset_check(asset=tokenizer_training, blocking=True)
def tokenizer_file_valid(
    context: dg.AssetCheckExecutionContext,
) -> dg.AssetCheckResult:
    """Check that the canonical tokenizer file exists and is valid."""

    # Get the canonical tokenizer file
    tokenizer_path = _get_canonical_tokenizer_file()

    if tokenizer_path is None:
        return dg.AssetCheckResult(
            passed=False,
            description="Canonical tokenizer file not found at data/tokenizer/tokenizer.json",
        )

    try:
        with open(tokenizer_path, "r") as f:
            data = json.load(f)

        # Check required fields
        required_fields = ["vocab_size", "pattern", "mergeable_ranks"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return dg.AssetCheckResult(
                passed=False,
                description=f"Missing required fields: {missing_fields}",
            )

        # Check vocab_size matches actual number of mergeable_ranks
        vocab_size = data["vocab_size"]
        actual_merges = len(data["mergeable_ranks"])

        if vocab_size != actual_merges:
            return dg.AssetCheckResult(
                passed=False,
                description=f"Vocab size mismatch: declared {vocab_size}, found {actual_merges} merge rules",
                metadata={
                    "declared_vocab_size": vocab_size,
                    "actual_merges": actual_merges,
                },
            )

        # Check that vocab_size is reasonable (at least 256 for byte tokens)
        if vocab_size < 256:
            return dg.AssetCheckResult(
                passed=False,
                description=f"Vocab size {vocab_size} too small (should be >= 256)",
            )

        return dg.AssetCheckResult(
            passed=True,
            description=f"Canonical tokenizer file is valid (vocab_size: {vocab_size:,})",
            metadata={
                "vocab_size": vocab_size,
                "num_merges": actual_merges,
                "pattern": dg.MetadataValue.text(data.get("pattern", "N/A")[:100]),
                "file_path": dg.MetadataValue.text(tokenizer_path),
            },
        )

    except json.JSONDecodeError as e:
        return dg.AssetCheckResult(
            passed=False,
            description=f"Invalid JSON: {str(e)}",
        )
    except Exception as e:
        return dg.AssetCheckResult(
            passed=False,
            description=f"Error loading tokenizer: {str(e)}",
        )


@dg.asset_check(asset=tokenizer_training, blocking=False)
def tokenizer_encode_decode(
    context: dg.AssetCheckExecutionContext,
) -> dg.AssetCheckResult:
    """Check that the canonical tokenizer can encode and decode text correctly."""

    # Get the canonical tokenizer file
    tokenizer_path = _get_canonical_tokenizer_file()

    if tokenizer_path is None:
        return dg.AssetCheckResult(
            passed=False,
            description="Canonical tokenizer file not found",
        )

    # Load the tokenizer data
    with open(tokenizer_path, "r") as f:
        data = json.load(f)

    # Reconstruct tiktoken encoding
    mergeable_ranks = {
        bytes(item["bytes"]): item["token_id"] for item in data["mergeable_ranks"]
    }
    pattern = data["pattern"]

    enc = tiktoken.Encoding(
        name="test_tokenizer",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )

    # Test encoding and decoding
    test_text = "Hello, world! This is a test of the tokenizer."

    try:
        encoded = enc.encode_ordinary(test_text)
        decoded = enc.decode(encoded)

        if decoded != test_text:
            return dg.AssetCheckResult(
                passed=False,
                description="Encode/decode round-trip failed: input != output",
                metadata={
                    "input": dg.MetadataValue.text(test_text),
                    "output": dg.MetadataValue.text(decoded),
                    "tested_file": dg.MetadataValue.text(
                        os.path.basename(tokenizer_path)
                    ),
                },
            )

        return dg.AssetCheckResult(
            passed=True,
            description="Tokenizer encode/decode works correctly",
            metadata={
                "test_text_length": len(test_text),
                "num_tokens": len(encoded),
                "tested_file": dg.MetadataValue.text(os.path.basename(tokenizer_path)),
            },
        )
    except Exception as e:
        return dg.AssetCheckResult(
            passed=False,
            description=f"Encode/decode failed with error: {str(e)}",
            metadata={
                "tested_file": dg.MetadataValue.text(os.path.basename(tokenizer_path)),
            },
        )


@dg.asset_check(asset=tokenizer_training, blocking=False)
def tokenizer_compression_ratio(
    context: dg.AssetCheckExecutionContext,
) -> dg.AssetCheckResult:
    """Evaluate tokenizer compression ratio on sample texts."""

    # Get the canonical tokenizer file
    tokenizer_path = _get_canonical_tokenizer_file()

    if tokenizer_path is None:
        return dg.AssetCheckResult(
            passed=False,
            description="Canonical tokenizer file not found",
        )

    # Load the tokenizer data
    with open(tokenizer_path, "r") as f:
        data = json.load(f)

    # Reconstruct tiktoken encoding
    mergeable_ranks = {
        bytes(item["bytes"]): item["token_id"] for item in data["mergeable_ranks"]
    }
    pattern = data["pattern"]

    enc = tiktoken.Encoding(
        name="test_tokenizer",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )

    # Evaluate compression on sample texts
    compression_results = {}
    total_bytes = 0
    total_tokens = 0

    for text_type, text in EVAL_TEXTS.items():
        encoded = enc.encode_ordinary(text)
        text_bytes = len(text.encode("utf-8"))
        num_tokens = len(encoded)
        ratio = text_bytes / num_tokens if num_tokens > 0 else 0

        compression_results[text_type] = {
            "bytes": text_bytes,
            "tokens": num_tokens,
            "ratio": ratio,
        }

        total_bytes += text_bytes
        total_tokens += num_tokens

    overall_ratio = total_bytes / total_tokens if total_tokens > 0 else 0

    # Check that compression ratio is reasonable (should be > 1.0, ideally 3-4 for English)
    min_acceptable_ratio = 1.5

    if overall_ratio < min_acceptable_ratio:
        return dg.AssetCheckResult(
            passed=False,
            severity=dg.AssetCheckSeverity.WARN,
            description=f"Compression ratio {overall_ratio:.2f} is below acceptable threshold {min_acceptable_ratio}",
            metadata={
                "overall_ratio": overall_ratio,
                "news_ratio": compression_results["news"]["ratio"],
                "code_ratio": compression_results["code"]["ratio"],
                "tested_file": dg.MetadataValue.text(os.path.basename(tokenizer_path)),
            },
        )

    return dg.AssetCheckResult(
        passed=True,
        description=f"Tokenizer compression ratio: {overall_ratio:.2f} bytes/token",
        metadata={
            "overall_ratio": overall_ratio,
            "news_ratio": compression_results["news"]["ratio"],
            "korean_ratio": compression_results["korean"]["ratio"],
            "code_ratio": compression_results["code"]["ratio"],
            "science_ratio": compression_results["science"]["ratio"],
            "total_bytes": total_bytes,
            "total_tokens": total_tokens,
            "tested_file": dg.MetadataValue.text(os.path.basename(tokenizer_path)),
        },
    )


def _compare_tokenizer_vs_gpt2_single_text(
    text_type: str, text: str
) -> dg.AssetCheckResult:
    """
    Helper function to compare tokenizer compression against GPT-2 for a single text.

    Args:
        text_type: Type of text (e.g., "news", "korean", "code", "science")
        text: The actual text to tokenize

    Returns:
        AssetCheckResult with comparison metrics
    """
    # Get the canonical tokenizer file
    tokenizer_path = _get_canonical_tokenizer_file()

    if tokenizer_path is None:
        return dg.AssetCheckResult(
            passed=False,
            description="Canonical tokenizer file not found",
        )

    # Load our tokenizer
    with open(tokenizer_path, "r") as f:
        data = json.load(f)

    mergeable_ranks = {
        bytes(item["bytes"]): item["token_id"] for item in data["mergeable_ranks"]
    }
    pattern = data["pattern"]

    our_enc = tiktoken.Encoding(
        name="our_tokenizer",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )

    # Load GPT-2 tokenizer for comparison
    try:
        gpt2_enc = tiktoken.get_encoding("gpt2")
    except Exception as e:
        return dg.AssetCheckResult(
            passed=False,
            severity=dg.AssetCheckSeverity.WARN,
            description=f"Could not load GPT-2 tokenizer for comparison: {str(e)}",
        )

    # Compare on the specific text
    text_bytes = len(text.encode("utf-8"))
    our_tokens = len(our_enc.encode_ordinary(text))
    gpt2_tokens = len(gpt2_enc.encode_ordinary(text))
    relative_diff = (
        ((gpt2_tokens - our_tokens) / gpt2_tokens * 100) if gpt2_tokens > 0 else 0
    )

    # Check passes if we're not significantly worse than GPT-2
    # Negative = we use fewer tokens (better) → always pass
    # Positive = we use more tokens (worse) → only fail if > 30% worse
    passed = relative_diff <= 30.0

    metadata = {
        "text_type": text_type,
        "our_tokens": our_tokens,
        "gpt2_tokens": gpt2_tokens,
        "our_ratio": text_bytes / our_tokens if our_tokens > 0 else 0,
        "gpt2_ratio": text_bytes / gpt2_tokens if gpt2_tokens > 0 else 0,
        "relative_diff_pct": relative_diff,
        "text_bytes": text_bytes,
        "our_vocab_size": data["vocab_size"],
        "gpt2_vocab_size": gpt2_enc.n_vocab,
        "tested_file": dg.MetadataValue.text(os.path.basename(tokenizer_path)),
    }

    # Only include severity if the check failed
    if passed:
        if relative_diff < 0:
            description = (
                f"{text_type.title()} text: {relative_diff:+.1f}% vs GPT-2 (better)"
            )
        else:
            description = (
                f"{text_type.title()} text: {relative_diff:+.1f}% vs GPT-2 (acceptable)"
            )
        return dg.AssetCheckResult(
            passed=passed,
            description=description,
            metadata=metadata,
        )
    else:
        return dg.AssetCheckResult(
            passed=passed,
            severity=dg.AssetCheckSeverity.WARN,
            description=f"{text_type.title()} text: {relative_diff:+.1f}% vs GPT-2 (>30% worse)",
            metadata=metadata,
        )


@dg.asset_check(asset=tokenizer_training, blocking=False)
def tokenizer_vs_gpt2_news(
    context: dg.AssetCheckExecutionContext,
) -> dg.AssetCheckResult:
    """Compare tokenizer compression vs GPT-2 on news text (formal English)."""
    return _compare_tokenizer_vs_gpt2_single_text("news", EVAL_TEXTS["news"])


@dg.asset_check(asset=tokenizer_training, blocking=False)
def tokenizer_vs_gpt2_korean(
    context: dg.AssetCheckExecutionContext,
) -> dg.AssetCheckResult:
    """Compare tokenizer compression vs GPT-2 on Korean text (non-Latin script)."""
    return _compare_tokenizer_vs_gpt2_single_text("korean", EVAL_TEXTS["korean"])


@dg.asset_check(asset=tokenizer_training, blocking=False)
def tokenizer_vs_gpt2_code(
    context: dg.AssetCheckExecutionContext,
) -> dg.AssetCheckResult:
    """Compare tokenizer compression vs GPT-2 on Python code (technical syntax)."""
    return _compare_tokenizer_vs_gpt2_single_text("code", EVAL_TEXTS["code"])


@dg.asset_check(asset=tokenizer_training, blocking=False)
def tokenizer_vs_gpt2_science(
    context: dg.AssetCheckExecutionContext,
) -> dg.AssetCheckResult:
    """Compare tokenizer compression vs GPT-2 on scientific text (domain vocabulary)."""
    return _compare_tokenizer_vs_gpt2_single_text("science", EVAL_TEXTS["science"])


# =============================================================================
# SFT Model Accuracy Checks
# =============================================================================


@dg.asset_check(asset="sft_checkpoint", blocking=False)
def sft_mmlu_accuracy(
    context: dg.AssetCheckExecutionContext,
) -> dg.AssetCheckResult:
    """Evaluate SFT model accuracy on MMLU (multiple choice reasoning)."""

    context.log.info("Evaluating MMLU accuracy on SFT model...")

    # Device setup
    device_type = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_type)

    # Find the latest SFT checkpoint
    model_tag = "d4"  # Default tag
    sft_checkpoint_dir = os.path.join(SFT_CHECKPOINT_DIRECTORY, model_tag)

    if not os.path.exists(sft_checkpoint_dir):
        return dg.AssetCheckResult(
            passed=False,
            description=f"SFT checkpoint directory not found: {sft_checkpoint_dir}",
        )

    # Load the canonical checkpoint.json
    checkpoint_path = os.path.join(sft_checkpoint_dir, "checkpoint.json")
    if not os.path.exists(checkpoint_path):
        return dg.AssetCheckResult(
            passed=False,
            description=f"Checkpoint file not found: {checkpoint_path}",
        )

    try:
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        # Load model
        model_config_dict = checkpoint_data["model_config"]
        model_config = GPTConfig(**model_config_dict)
        model = GPT(model_config)

        model_path = os.path.join(sft_checkpoint_dir, "model.pt")
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model = model.to(device)
        model.eval()

        # Load tokenizer
        tokenizer_path = os.path.join(sft_checkpoint_dir, "tokenizer.json")
        with open(tokenizer_path, "r") as f:
            tokenizer_data = json.load(f)

        base_vocab_size = tokenizer_data["vocab_size"]
        tokenizer = create_tokenizer_with_special_tokens(
            tokenizer_data, base_vocab_size
        )

        # Create engine
        engine = Engine(model, tokenizer)

        # Run MMLU evaluation (limited to 256 problems for quick check)
        with torch.no_grad():
            accuracy = run_chat_eval(
                "MMLU",
                model,
                tokenizer,
                engine,
                batch_size=8,
                max_problems=256,
            )

        # MMLU random baseline is 0.25 (1 in 4 choices)
        # Good performance is > 0.30
        baseline = 0.25
        threshold = 0.28

        passed = accuracy >= threshold

        return dg.AssetCheckResult(
            passed=passed,
            description=f"MMLU accuracy: {accuracy:.2%} (baseline: {baseline:.2%}, threshold: {threshold:.2%})",
            metadata={
                "accuracy": accuracy,
                "baseline": baseline,
                "threshold": threshold,
                "above_baseline": accuracy > baseline,
                "num_problems_tested": 256,
                "model_tag": model_tag,
                "checkpoint": "checkpoint.json",
            },
        )

    except Exception as e:
        return dg.AssetCheckResult(
            passed=False,
            description=f"Error evaluating MMLU: {str(e)}",
        )


@dg.asset_check(asset="sft_checkpoint", blocking=False)
def sft_arc_easy_accuracy(
    context: dg.AssetCheckExecutionContext,
) -> dg.AssetCheckResult:
    """Evaluate SFT model accuracy on ARC-Easy (science reasoning)."""

    context.log.info("Evaluating ARC-Easy accuracy on SFT model...")

    # Device setup
    device_type = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_type)

    # Find the latest SFT checkpoint
    model_tag = "d4"  # Default tag
    sft_checkpoint_dir = os.path.join(SFT_CHECKPOINT_DIRECTORY, model_tag)

    if not os.path.exists(sft_checkpoint_dir):
        return dg.AssetCheckResult(
            passed=False,
            description=f"SFT checkpoint directory not found: {sft_checkpoint_dir}",
        )

    # Load the canonical checkpoint.json
    checkpoint_path = os.path.join(sft_checkpoint_dir, "checkpoint.json")
    if not os.path.exists(checkpoint_path):
        return dg.AssetCheckResult(
            passed=False,
            description=f"Checkpoint file not found: {checkpoint_path}",
        )

    try:
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        # Load model
        model_config_dict = checkpoint_data["model_config"]
        model_config = GPTConfig(**model_config_dict)
        model = GPT(model_config)

        model_path = os.path.join(sft_checkpoint_dir, "model.pt")
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model = model.to(device)
        model.eval()

        # Load tokenizer
        tokenizer_path = os.path.join(sft_checkpoint_dir, "tokenizer.json")
        with open(tokenizer_path, "r") as f:
            tokenizer_data = json.load(f)

        base_vocab_size = tokenizer_data["vocab_size"]
        tokenizer = create_tokenizer_with_special_tokens(
            tokenizer_data, base_vocab_size
        )

        # Create engine
        engine = Engine(model, tokenizer)

        # Run ARC-Easy evaluation (limited to 256 problems for quick check)
        with torch.no_grad():
            accuracy = run_chat_eval(
                "ARC-Easy",
                model,
                tokenizer,
                engine,
                batch_size=8,
                max_problems=256,
            )

        # ARC-Easy random baseline is 0.25 (1 in 4 choices)
        # Good performance is > 0.35
        baseline = 0.25
        threshold = 0.30

        passed = accuracy >= threshold

        return dg.AssetCheckResult(
            passed=passed,
            description=f"ARC-Easy accuracy: {accuracy:.2%} (baseline: {baseline:.2%}, threshold: {threshold:.2%})",
            metadata={
                "accuracy": accuracy,
                "baseline": baseline,
                "threshold": threshold,
                "above_baseline": accuracy > baseline,
                "num_problems_tested": 256,
                "model_tag": model_tag,
                "checkpoint": "checkpoint.json",
            },
        )

    except Exception as e:
        return dg.AssetCheckResult(
            passed=False,
            description=f"Error evaluating ARC-Easy: {str(e)}",
        )


@dg.asset_check(asset="serverless_endpoint", blocking=True)
def serverless_endpoint_exists(
    context: dg.AssetCheckExecutionContext,
    serverless: ServerlessResource,
    serverless_endpoint: str,
) -> dg.AssetCheckResult:
    """
    Check if the serverless endpoint exists and is accessible.

    This validates that:
    1. The endpoint ID is valid
    2. The endpoint exists in RunPod
    3. The endpoint is accessible via API
    """
    endpoint_id = serverless_endpoint

    context.log.info(f"Checking if endpoint exists: {endpoint_id}")

    try:
        endpoint_info = serverless.get_endpoint(endpoint_id, context=context)

        if endpoint_info:
            endpoint_name = endpoint_info.get("name", "N/A")
            workers_min = endpoint_info.get("workersMin", "N/A")
            workers_max = endpoint_info.get("workersMax", "N/A")

            context.log.info(f"Endpoint found: {endpoint_name}")
            context.log.info(f"Workers: min={workers_min}, max={workers_max}")

            return dg.AssetCheckResult(
                passed=True,
                description=f"Endpoint '{endpoint_name}' (ID: {endpoint_id}) exists and is accessible",
                metadata={
                    "endpoint_id": dg.MetadataValue.text(endpoint_id),
                    "endpoint_name": dg.MetadataValue.text(endpoint_name),
                    "workers_min": workers_min,
                    "workers_max": workers_max,
                },
            )
        else:
            context.log.warning(f"Endpoint {endpoint_id} not found")
            return dg.AssetCheckResult(
                passed=False,
                description=f"Endpoint {endpoint_id} does not exist or was deleted",
            )

    except Exception as e:
        context.log.error(f"Error checking endpoint: {e}")
        return dg.AssetCheckResult(
            passed=False,
            description=f"Failed to check endpoint: {str(e)}",
        )
