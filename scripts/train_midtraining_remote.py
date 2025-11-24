"""
Midtraining script for RunPod GPU execution.

This script:
1. Downloads config, base checkpoint, and tokenizer from S3
2. Loads and extends base model for conversation tokens
3. Downloads pre-serialized JSONL datasets from S3
4. Trains on conversational data
5. Uploads midtrained checkpoint to S3
"""

import argparse
import json
import logging
import os
import tarfile
import time
import warnings

import boto3
import torch
import torch.distributed as dist

warnings.filterwarnings("ignore")

from dagster_nanochat.defs.config import CONVERSATION_SPECIAL_TOKENS
from dagster_nanochat.nanochat.common import get_dist_info, print0
from dagster_nanochat.nanochat.gpt import GPT, GPTConfig
from dagster_nanochat.utils.data_generators import create_midtraining_data_generator
from dagster_nanochat.utils.model_utils import extend_model_vocab_for_special_tokens
from dagster_nanochat.utils.tokenizer_utils import create_tokenizer_with_special_tokens


def get_s3_client():
    """Create S3 client with RunPod secrets."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["RUNPOD_SECRET_AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def download_from_s3(s3_client, bucket, key, local_path):
    """Download a file from S3."""
    print0(f"Downloading from S3: s3://{bucket}/{key}", flush=True)
    s3_client.download_file(bucket, key, local_path)
    print0(f"Downloaded to: {local_path}", flush=True)


def upload_to_s3(s3_client, local_path, bucket, key):
    """Upload a file to S3."""
    print0(f"Uploading to S3: s3://{bucket}/{key}", flush=True)
    s3_client.upload_file(local_path, bucket, key)
    print0(f"Uploaded from: {local_path}", flush=True)


def load_jsonl_datasets(s3_client, bucket, train_key, val_key):
    """Load pre-serialized JSONL datasets from S3.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        train_key: S3 key for training dataset
        val_key: S3 key for validation dataset

    Returns:
        Tuple of (train_dataset, val_dataset) as lists of conversations
    """
    print0("Loading pre-serialized datasets from S3...")

    # Download train dataset
    train_path = "train_dataset.jsonl"
    print0(f"Downloading train dataset: s3://{bucket}/{train_key}")
    download_from_s3(s3_client, bucket, train_key, train_path)

    train_dataset = []
    with open(train_path, "r") as f:
        for line in f:
            conversation = json.loads(line)
            train_dataset.append(conversation)

    print0(f"Training dataset: {len(train_dataset):,} examples")

    # Download val dataset
    val_path = "val_dataset.jsonl"
    print0(f"Downloading val dataset: s3://{bucket}/{val_key}")
    download_from_s3(s3_client, bucket, val_key, val_path)

    val_dataset = []
    with open(val_path, "r") as f:
        for line in f:
            conversation = json.loads(line)
            val_dataset.append(conversation)

    print0(f"Validation dataset: {len(val_dataset):,} examples")

    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--s3-config-key", required=True, help="S3 key for training config"
    )
    args = parser.parse_args()

    # Initialize DDP if available
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    print0("Starting midtraining on RunPod GPU...", flush=True)

    # Setup device (single GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        print0(f"🎮 Using device: {device} (type: {device_type})", flush=True)
        print0(f"   GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print0("Using CPU", flush=True)

    # Initialize S3 client
    s3_client = get_s3_client()

    # Download and load config
    config_path = "config.json"
    print0("Downloading config from S3...", flush=True)
    download_from_s3(s3_client, args.s3_bucket, args.s3_config_key, config_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    print0(f"Config: {json.dumps(config, indent=2)}", flush=True)

    # Extract config parameters
    s3_train_dataset_key = config["s3_train_dataset_key"]
    s3_val_dataset_key = config["s3_val_dataset_key"]
    max_seq_len = config["max_seq_len"]
    device_batch_size = config["device_batch_size"]
    total_batch_size = config["total_batch_size"]
    num_iterations = config["num_iterations"]
    eval_every = config["eval_every"]
    eval_tokens = config["eval_tokens"]
    model_tag = config["model_tag"]

    # Download base checkpoint from S3
    checkpoint_dir = "base_checkpoint"
    print0("Downloading base checkpoint from S3...", flush=True)
    checkpoint_tarball = "base_checkpoint.tar.gz"
    download_from_s3(
        s3_client, args.s3_bucket, config["s3_base_checkpoint_key"], checkpoint_tarball
    )

    # Extract checkpoint
    print0("Extracting base checkpoint...", flush=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with tarfile.open(checkpoint_tarball, "r:gz") as tar:
        tar.extractall(checkpoint_dir)
    print0(f"Extracted to: {checkpoint_dir}", flush=True)

    # Load base checkpoint metadata
    metadata_path = os.path.join(checkpoint_dir, "checkpoint.json")
    with open(metadata_path, "r") as f:
        checkpoint_metadata = json.load(f)

    model_config_dict = checkpoint_metadata["model_config"]
    base_vocab_size = model_config_dict["vocab_size"]

    print0(f"Base model config: {model_config_dict}", flush=True)

    # Create model from checkpoint
    print0("Loading base model...", flush=True)
    model_config = GPTConfig(
        sequence_len=model_config_dict["sequence_len"],
        vocab_size=base_vocab_size,
        n_layer=model_config_dict["n_layer"],
        n_head=model_config_dict["n_head"],
        n_kv_head=model_config_dict["n_kv_head"],
        n_embd=model_config_dict["n_embd"],
    )

    with torch.device("meta"):
        model = GPT(model_config)

    model.to_empty(device=device)

    # Load weights
    model_path = os.path.join(checkpoint_dir, "model.pt")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print0("Base model weights loaded", flush=True)

    # Download tokenizer from S3
    tokenizer_path = "tokenizer.json"
    print0("Downloading tokenizer from S3...", flush=True)
    download_from_s3(
        s3_client, args.s3_bucket, config["s3_tokenizer_key"], tokenizer_path
    )

    # Load tokenizer
    print0("Loading tokenizer...", flush=True)
    with open(tokenizer_path, "r") as f:
        tokenizer_data = json.load(f)

    # Extend model vocabulary for conversation special tokens
    print0("Extending model vocabulary for conversation tokens...", flush=True)

    # Simple context logger for extend_model_vocab_for_special_tokens
    class SimpleLogger:
        def info(self, msg):
            print0(msg, flush=True)

    class SimpleContext:
        def __init__(self):
            self.log = SimpleLogger()

    context = SimpleContext()
    extended_vocab_size = extend_model_vocab_for_special_tokens(
        model, base_vocab_size, device, context
    )

    print0(
        f"Vocabulary extended: {base_vocab_size} → {extended_vocab_size}", flush=True
    )

    # Create tokenizer with special tokens
    tokenizer = create_tokenizer_with_special_tokens(tokenizer_data, base_vocab_size)
    print0("Tokenizer loaded with special tokens", flush=True)

    # Load pre-serialized datasets from S3
    train_dataset, val_dataset = load_jsonl_datasets(
        s3_client, args.s3_bucket, s3_train_dataset_key, s3_val_dataset_key
    )

    # Setup optimizers
    print0("Setting up optimizers...", flush=True)
    optimizers = model.setup_optimizers(
        unembedding_lr=config["unembedding_lr"] * config["init_lr_frac"],
        embedding_lr=config["embedding_lr"] * config["init_lr_frac"],
        matrix_lr=config["matrix_lr"] * config["init_lr_frac"],
        weight_decay=config["weight_decay"],
    )

    # Save initial LR for each optimizer
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    print0("Optimizers initialized", flush=True)

    # Synchronize all ranks after optimizer setup
    if ddp:
        dist.barrier()
        print0("All ranks ready for training", flush=True)

    # Create data generators
    print0("Creating data generators...", flush=True)
    mid_data_generator, get_last_step, get_progress = create_midtraining_data_generator(
        train_dataset,
        val_dataset,
        tokenizer,
        device_batch_size,
        max_seq_len,
        num_iterations,
        device_type,
        device,
    )

    train_loader = mid_data_generator("train")
    build_val_loader = lambda: mid_data_generator("val")

    # Learning rate scheduler (decay after 80%)
    def get_lr_multiplier(progress):
        return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2

    # Autocast context for mixed precision
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else torch.no_grad()
    )

    # Training loop
    print0("Starting training loop...", flush=True)
    start_time = time.time()
    step = 0
    min_val_loss = float("inf")

    with autocast_ctx:
        for x, y in train_loader:
            # Validation (do at step 0, then every eval_every steps)
            if step % eval_every == 0:
                print0(
                    f"Step {step:05d}/{num_iterations:05d} | Evaluating...", flush=True
                )
                val_loader = build_val_loader()
                total_loss = 0.0
                num_batches = 0

                # Calculate how many validation batches to run based on eval_tokens
                max_val_batches = max(
                    1, eval_tokens // (device_batch_size * max_seq_len)
                )

                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        val_loss = model(x_val, y_val)
                        total_loss += val_loss.item()
                        num_batches += 1

                        # Stop after max_val_batches to avoid infinite loop
                        if num_batches >= max_val_batches:
                            break

                avg_val_loss = (
                    total_loss / num_batches if num_batches > 0 else float("inf")
                )
                min_val_loss = min(min_val_loss, avg_val_loss)
                print0(
                    f"Step {step:05d} | Val Loss: {avg_val_loss:.4f} (best: {min_val_loss:.4f})",
                    flush=True,
                )

            # Forward pass
            loss = model(x, y)

            # Backward pass
            loss.backward()

            # Optimizer step
            for opt in optimizers:
                opt.step()
                opt.zero_grad()

            # Update learning rates based on progress
            progress = step / num_iterations if num_iterations > 0 else 0
            lr_multiplier = get_lr_multiplier(progress)
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * lr_multiplier

            # Increment step counter
            step += 1

            # Stop after num_iterations
            if step >= num_iterations:
                break

    # Final validation
    print0(f"Step {step:05d}/{num_iterations:05d} | Evaluating...", flush=True)
    val_loader = build_val_loader()
    total_loss = 0.0
    num_batches = 0
    max_val_batches = max(1, eval_tokens // (device_batch_size * max_seq_len))

    with torch.no_grad():
        for x_val, y_val in val_loader:
            val_loss = model(x_val, y_val)
            total_loss += val_loss.item()
            num_batches += 1
            if num_batches >= max_val_batches:
                break

    avg_val_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    min_val_loss = min(min_val_loss, avg_val_loss)
    print0(
        f"Step {step:05d} | Val Loss: {avg_val_loss:.4f} (best: {min_val_loss:.4f})",
        flush=True,
    )

    training_time = time.time() - start_time
    final_loss = loss.item()

    print0(f"Training complete in {training_time:.2f}s", flush=True)

    # Synchronize all ranks before saving
    if ddp:
        dist.barrier()

    # Save checkpoint (only rank 0)
    if ddp_rank == 0:
        print0("Saving checkpoint...", flush=True)
        output_dir = "checkpoint"
        os.makedirs(output_dir, exist_ok=True)

        # Save model weights
        output_model_path = os.path.join(output_dir, "model.pt")
        torch.save(model.state_dict(), output_model_path)
        print0(f"Model weights saved to: {output_model_path}", flush=True)

        # Save metadata
        updated_model_config = model_config_dict.copy()
        updated_model_config["vocab_size"] = extended_vocab_size

        checkpoint_metadata = {
            "step": step,
            "final_loss": final_loss,
            "min_val_loss": min_val_loss if min_val_loss != float("inf") else None,
            "model_config": updated_model_config,
            "training_config": {
                "max_seq_len": max_seq_len,
                "device_batch_size": device_batch_size,
                "total_batch_size": total_batch_size,
                "num_iterations": num_iterations,
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
                "base_vocab_size": base_vocab_size,
                "num_special_tokens": len(CONVERSATION_SPECIAL_TOKENS),
            },
            "training_time_seconds": training_time,
            "device": str(device),
            "base_checkpoint": model_tag,
            "ddp_world_size": ddp_world_size if ddp else 1,
        }

        metadata_output_path = os.path.join(output_dir, "checkpoint.json")
        with open(metadata_output_path, "w") as f:
            json.dump(checkpoint_metadata, f, indent=2)
        print0(f"Metadata saved to: {metadata_output_path}", flush=True)

        # Create tarball
        print0("Creating checkpoint tarball...", flush=True)
        tarball_path = "checkpoint.tar.gz"
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(output_dir, arcname=".")
        print0(f"Created tarball: {tarball_path}", flush=True)

        # Upload to S3
        upload_to_s3(s3_client, tarball_path, args.s3_bucket, config["s3_output_key"])

        print0("Midtraining complete!", flush=True)
        print0(f"Final loss: {final_loss:.4f}", flush=True)
        print0(f"Best validation loss: {min_val_loss:.4f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"❌ Training failed with error: {e}", flush=True)
        print("Full traceback:", flush=True)
        traceback.print_exc()
        raise
