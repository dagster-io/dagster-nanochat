"""Common setup utilities for training assets."""

import os


def setup_hf_cache(cache_dir: str) -> str:
    """
    Setup HuggingFace datasets cache directory.

    Args:
        cache_dir: Path to cache directory (e.g., "data/hf_datasets_cache")

    Returns:
        Absolute path to cache directory
    """
    hf_cache_dir = os.path.abspath(cache_dir)
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
    os.environ["HF_HOME"] = hf_cache_dir
    return hf_cache_dir
