"""Checkpoint loading and S3 utilities."""

import json
import os
import tarfile
import time
from typing import Any

import dagster as dg
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


def wait_for_s3_checkpoint(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    context: dg.AssetExecutionContext,
    max_wait_seconds: int = 7200,
    check_interval: int = 60,
) -> None:
    """
    Poll S3 until a checkpoint appears, indicating training completion.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        s3_key: S3 key to poll for
        context: Dagster context for logging
        max_wait_seconds: Maximum time to wait (default: 2 hours)
        check_interval: Seconds between checks (default: 60s)

    Raises:
        TimeoutError: If checkpoint doesn't appear within max_wait_seconds
    """
    context.log.info(f"Waiting for checkpoint: s3://{bucket}/{s3_key}")

    start_time = time.time()

    while time.time() - start_time < max_wait_seconds:
        try:
            s3_client.head_object(Bucket=bucket, Key=s3_key)
            elapsed = int(time.time() - start_time)
            context.log.info(f"✓ Checkpoint found after {elapsed}s!")
            return
        except s3_client.exceptions.ClientError:
            # Checkpoint not ready yet
            elapsed = int(time.time() - start_time)
            if elapsed % 120 == 0:  # Log every 2 minutes
                context.log.info(f"   Still waiting... ({elapsed}s elapsed)")
            time.sleep(check_interval)

    raise TimeoutError(
        f"Checkpoint did not appear in S3 within {max_wait_seconds}s. "
        f"Check training logs for errors."
    )


def download_and_extract_checkpoint(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    output_dir: str,
    context: dg.AssetExecutionContext,
) -> dict:
    """
    Download checkpoint tarball from S3 and extract it.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        s3_key: S3 key of the checkpoint tarball
        output_dir: Local directory to extract checkpoint to
        context: Dagster context for logging

    Returns:
        Checkpoint metadata dictionary loaded from checkpoint.json

    Raises:
        FileNotFoundError: If checkpoint.json not found after extraction
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download tarball
    context.log.info(f"Downloading checkpoint from S3...")
    tarball_path = os.path.join(output_dir, "checkpoint.tar.gz")
    s3_client.download_file(bucket, s3_key, tarball_path)

    # Extract tarball
    tarball_size_mb = os.path.getsize(tarball_path) / (1024 * 1024)
    context.log.info(f"Extracting checkpoint ({tarball_size_mb:.1f} MB)...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(output_dir)

    # Remove tarball
    os.remove(tarball_path)
    context.log.info(f"Checkpoint ready: {output_dir}")

    # Load and return checkpoint metadata
    metadata_path = os.path.join(output_dir, "checkpoint.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"checkpoint.json not found in extracted checkpoint at {metadata_path}"
        )

    with open(metadata_path, "r") as f:
        checkpoint_metadata = json.load(f)

    return checkpoint_metadata


def wait_and_download_checkpoint(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    output_dir: str,
    context: dg.AssetExecutionContext,
    max_wait_seconds: int = 7200,
    check_interval: int = 60,
) -> dict:
    """
    Wait for checkpoint to appear in S3, then download and extract it.

    This is a convenience function that combines wait_for_s3_checkpoint
    and download_and_extract_checkpoint.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        s3_key: S3 key of the checkpoint tarball
        output_dir: Local directory to extract checkpoint to
        context: Dagster context for logging
        max_wait_seconds: Maximum time to wait (default: 2 hours)
        check_interval: Seconds between checks (default: 60s)

    Returns:
        Checkpoint metadata dictionary loaded from checkpoint.json

    Raises:
        TimeoutError: If checkpoint doesn't appear within max_wait_seconds
        FileNotFoundError: If checkpoint.json not found after extraction
    """
    # Wait for checkpoint to appear
    wait_for_s3_checkpoint(
        s3_client=s3_client,
        bucket=bucket,
        s3_key=s3_key,
        context=context,
        max_wait_seconds=max_wait_seconds,
        check_interval=check_interval,
    )

    # Download and extract
    checkpoint_metadata = download_and_extract_checkpoint(
        s3_client=s3_client,
        bucket=bucket,
        s3_key=s3_key,
        output_dir=output_dir,
        context=context,
    )

    return checkpoint_metadata
