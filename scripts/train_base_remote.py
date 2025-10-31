#!/usr/bin/env python
"""
Standalone training script for base model pretraining.

This script runs on RunPod GPU instances and is called by the base_model_checkpoint asset.
It downloads the tokenizer from S3, downloads training data from HuggingFace, trains the model,
and uploads the checkpoint to S3.

Usage:
    python train_base_remote.py --config config.json
"""

import argparse
import json
import logging
import os
import tarfile
import time
import warnings
from contextlib import nullcontext

import boto3
import requests
import tiktoken
import torch
import torch.distributed as dist

# Suppress verbose logging
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from dagster_nanochat.nanochat.common import get_dist_info, print0
from dagster_nanochat.nanochat.dataloader import tokenizing_distributed_data_loader
from dagster_nanochat.nanochat.gpt import GPT, GPTConfig
from dagster_nanochat.nanochat.loss_eval import evaluate_bpb
from dagster_nanochat.nanochat.tokenizer import RustBPETokenizer


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


def download_shard(url: str, filepath: str, filename: str, shard_type: str) -> None:
    """Download a single shard with progress reporting."""
    if os.path.exists(filepath):
        print0(f"{shard_type} shard already exists, skipping: {filename}")
        return

    print0(f"Downloading {shard_type} shard: {filename}")
    temp_path = filepath + ".tmp"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        os.rename(temp_path, filepath)
        print0(f"Downloaded {filename}")
    except Exception as e:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to download {filename}: {e}") from e


def download_training_data(train_shard_ids: list[int]) -> None:
    """Download specific training data shards from HuggingFace.

    Args:
        train_shard_ids: List of shard IDs to download (e.g., [0, 1, 2, 7, 19, 32])
    """
    BASE_URL = (
        "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
    )

    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/validation", exist_ok=True)

    print0(f"Downloading {len(train_shard_ids)} training shards...")
    print0(
        f"   Shard IDs: {train_shard_ids[:10]}{'...' if len(train_shard_ids) > 10 else ''}"
    )

    # Download specific training shards
    for shard_id in train_shard_ids:
        filename = f"shard_{shard_id:05d}.parquet"
        url = f"{BASE_URL}/{filename}"
        filepath = f"data/raw/{filename}"
        download_shard(url, filepath, filename, "Training")

    # Always download the canonical validation shard (shard 1822)
    print0(f"\nDownloading validation shard...")
    val_filename = "shard_01822.parquet"
    val_url = f"{BASE_URL}/{val_filename}"
    val_filepath = f"data/validation/{val_filename}"
    download_shard(val_url, val_filepath, val_filename, "Validation")

    print0(f"\nDownload complete!")
    print0(f"   Training shards: {len(train_shard_ids)}")
    print0(f"   Validation shard: 1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name")
    parser.add_argument("--s3-config-key", required=True, help="S3 key for config JSON")
    args = parser.parse_args()

    print0("Starting base model training on RunPod GPU...")

    # Create S3 client
    s3_client = get_s3_client()

    # Download configuration from S3
    print0("Downloading config from S3...", flush=True)
    config_path = "config.json"
    download_from_s3(s3_client, args.s3_bucket, args.s3_config_key, config_path)

    with open(config_path, "r") as f:
        config = json.load(f)
    print0(f"Config: {json.dumps(config, indent=2)}")

    # Download tokenizer from S3
    s3_bucket = config["s3_bucket"]
    s3_tokenizer_key = config["s3_tokenizer_key"]
    s3_checkpoint_key = config["s3_checkpoint_key"]

    # Check if running with DDP - only rank 0 downloads data
    rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        # Only rank 0 downloads to avoid race conditions
        if not os.path.exists("tokenizer.json"):
            print0("Downloading tokenizer from S3...", flush=True)
            download_from_s3(s3_client, s3_bucket, s3_tokenizer_key, "tokenizer.json")
        else:
            print0("Tokenizer already exists, skipping download")

        # Download training data from HuggingFace
        # Use exact shard IDs from config to match what tokenizer was trained on
        train_shard_ids = config.get("train_shard_ids", list(range(1822)))
        download_training_data(train_shard_ids)
        print0("Data download complete (rank 0)")

    # Wait for rank 0 to finish downloading before other ranks proceed
    if rank != 0:
        print0(f"Rank {rank}: Waiting for rank 0 to download data...")
        # Wait up to 10 minutes for data to be downloaded
        max_wait = 600
        start = time.time()

        def check_data_ready():
            """Check if tokenizer and training data are ready."""
            tokenizer_exists = os.path.exists("tokenizer.json")
            try:
                train_data_exists = (
                    os.path.exists("data/raw") and len(os.listdir("data/raw")) > 0
                )
            except (FileNotFoundError, OSError):
                train_data_exists = False
            return tokenizer_exists and train_data_exists

        while not check_data_ready() and (time.time() - start) < max_wait:
            time.sleep(5)

        if not check_data_ready():
            raise RuntimeError(
                f"Rank {rank}: Timeout waiting for rank 0 to download data"
            )

        print0(f"Rank {rank}: Data ready, proceeding...")

    # Disable torch.compile for compatibility
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    # Extract config parameters
    depth = config["depth"]
    max_seq_len = config["max_seq_len"]
    device_batch_size = config["device_batch_size"]
    total_batch_size = config["total_batch_size"]
    num_iterations = config["num_iterations"]
    eval_every = config["eval_every"]
    vocab_size = config["vocab_size"]

    # Initialize distributed training (DDP) if available
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    master_process = ddp_rank == 0  # Only rank 0 logs and saves checkpoints

    device_type = "cuda"
    if ddp:
        torch.cuda.set_device(ddp_local_rank)
        device = torch.device(f"cuda:{ddp_local_rank}")
        dist.init_process_group(backend="nccl")
        print0(
            f"DDP initialized: rank {ddp_rank}/{ddp_world_size}, device {ddp_local_rank}"
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    print0(f"🎮 Using device: {device} (type: {device_type})")
    if torch.cuda.is_available():
        print0(f"   GPU: {torch.cuda.get_device_name(device)}")

    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    # Load tokenizer
    print0("Loading tokenizer...")
    with open("tokenizer.json") as f:
        tokenizer_data = json.load(f)

    mergeable_ranks = {
        bytes(item["bytes"]): item["token_id"]
        for item in tokenizer_data["mergeable_ranks"]
    }
    pattern = tokenizer_data["pattern"]

    # Add special tokens (for consistency)
    special_tokens = {
        token: vocab_size + i for i, token in enumerate(config["special_tokens"])
    }

    tokenizer_enc = tiktoken.Encoding(
        name="training_tokenizer",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    tokenizer = RustBPETokenizer(tokenizer_enc, "<|bos|>")

    # Compute token_bytes for BPB evaluation
    print0("Computing token bytes...")
    token_bytes_list = []
    for token_id in range(vocab_size):
        try:
            token_bytes_val = len(tokenizer_enc.decode_single_token_bytes(token_id))
            token_bytes_list.append(token_bytes_val)
        except Exception:
            # Some tokens may not be decodable, use 0 for these
            token_bytes_list.append(0)
    for _ in config["special_tokens"]:
        token_bytes_list.append(0)
    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device=device)

    # Model architecture
    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    num_kv_heads = num_heads

    # Model vocab size must include special tokens
    model_vocab_size = vocab_size + len(config["special_tokens"])
    print0(f"Model: {num_layers} layers, {model_dim} dim, {num_heads} heads")
    print0(
        f"Vocab: {vocab_size} base + {len(config['special_tokens'])} special = {model_vocab_size} total"
    )

    # Create model
    print0("Creating model...")
    model_config_kwargs = dict(
        sequence_len=max_seq_len,
        vocab_size=model_vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
    )

    with torch.device("meta"):
        model_config = GPTConfig(**model_config_kwargs)
        model = GPT(model_config)

    print0("Allocating model on GPU...", flush=True)
    model.to_empty(device=device)

    print0("Initializing model weights...", flush=True)
    model.init_weights()
    print0("Model weights initialized", flush=True)
    orig_model = model

    # Synchronize all ranks after model initialization
    if ddp:
        dist.barrier()
        print0(f"Model initialized on all ranks")

    # Wrap model in DDP if using distributed training
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ddp_local_rank],
            output_device=ddp_local_rank,
        )
        print0(f"Model wrapped in DDP")

    # Setup optimizers
    print0("Setting up optimizers...")
    optimizers = orig_model.setup_optimizers(
        unembedding_lr=config["unembedding_lr"],
        embedding_lr=config["embedding_lr"],
        matrix_lr=config["matrix_lr"],
        weight_decay=config["weight_decay"],
    )

    # Synchronize all ranks after optimizer setup
    if ddp:
        dist.barrier()

    # Calculate gradient accumulation accounting for world size
    tokens_per_fwdbwd = device_batch_size * max_seq_len  # per GPU
    world_tokens_per_fwdbwd = (
        tokens_per_fwdbwd * ddp_world_size
    )  # total across all GPUs
    grad_accum_steps = max(1, total_batch_size // world_tokens_per_fwdbwd)
    print0(f"Tokens/micro-batch/GPU: {tokens_per_fwdbwd:,}")
    print0(f"Tokens/micro-batch (all GPUs): {world_tokens_per_fwdbwd:,}")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")

    # Create data loaders
    print0("Creating data loaders...")
    train_loader = tokenizing_distributed_data_loader(
        device_batch_size,
        max_seq_len,
        split="train",
        tokenizer=tokenizer,
        device=device,
        data_dir="data/raw",
        validation_dir="data/validation",
    )

    build_val_loader = lambda: tokenizing_distributed_data_loader(
        device_batch_size,
        max_seq_len,
        split="val",
        tokenizer=tokenizer,
        device=device,
        data_dir="data/raw",
        validation_dir="data/validation",
    )

    x, y = next(train_loader)

    # Learning rate scheduler
    def get_lr_multiplier(it):
        warmup_iters = round(config["warmup_ratio"] * num_iterations)
        warmdown_iters = round(config["warmdown_ratio"] * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters if warmup_iters > 0 else 1.0
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (
                (num_iterations - it) / warmdown_iters if warmdown_iters > 0 else 0.0
            )
            return progress * 1.0 + (1 - progress) * config["final_lr_frac"]

    # Training loop
    print0("Starting training loop...")
    start_time = time.time()
    model.train()
    min_val_bpb = float("inf")

    for step in range(num_iterations):
        # Evaluate validation loss (all ranks participate)
        last_step = step == num_iterations - 1
        if last_step or (eval_every > 0 and step % eval_every == 0):
            model.eval()
            val_loader = build_val_loader()
            eval_steps = min(
                config["eval_steps"],
                config["eval_tokens"]
                // (device_batch_size * max_seq_len * ddp_world_size),
            )

            with autocast_ctx:
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)

            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb

            print0(
                f"Step {step:05d} | Validation BPB: {val_bpb:.4f} (best: {min_val_bpb:.4f})"
            )
            model.train()

        # Synchronize all processes before training step
        if ddp:
            dist.barrier()

        # Forward and backward
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y = next(train_loader)

        # Gradient clipping
        if config["grad_clip"] > 0.0:
            torch.nn.utils.clip_grad_norm_(orig_model.parameters(), config["grad_clip"])

        # Update learning rates
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm

        # Optimizer step
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        final_loss = train_loss.item()

        # Log training progress (only on master process)
        if master_process and (
            step % (eval_every * 4) == 0 or step == num_iterations - 1
        ):
            pct_done = 100 * step / num_iterations
            print0(
                f"Step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
                f"train loss: {final_loss:.6f} | lrm: {lrm:.4f}"
            )

    training_time = time.time() - start_time
    print0(f"Training complete in {training_time:.2f}s")

    # Synchronize all processes before saving
    if ddp:
        dist.barrier()

    # Save checkpoint (only on master process)
    if master_process:
        print0("Saving checkpoint...")
        os.makedirs("checkpoint", exist_ok=True)

        torch.save(orig_model.state_dict(), "checkpoint/model.pt")
        torch.save([opt.state_dict() for opt in optimizers], "checkpoint/optimizer.pt")

        checkpoint_metadata = {
            "step": num_iterations,
            "final_loss": final_loss,
            "best_val_bpb": min_val_bpb,
            "model_config": model_config_kwargs,
            "training_config": config,
            "training_time_seconds": training_time,
            "device": str(device),
            "ddp_world_size": ddp_world_size,
        }

        with open("checkpoint/checkpoint.json", "w") as f:
            json.dump(checkpoint_metadata, f, indent=2)

        print0(f"Checkpoint saved to checkpoint/")
        print0(f"Best validation BPB: {min_val_bpb:.4f}")

        # Upload checkpoint to S3
        print0("Creating checkpoint tarball...", flush=True)
        tarball_path = "checkpoint.tar.gz"
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add("checkpoint/model.pt", arcname="model.pt")
            tar.add("checkpoint/optimizer.pt", arcname="optimizer.pt")
            tar.add("checkpoint/checkpoint.json", arcname="checkpoint.json")
        print0(f"Created tarball: {tarball_path}", flush=True)

        upload_to_s3(s3_client, tarball_path, s3_bucket, s3_checkpoint_key)

    # Cleanup DDP
    if ddp:
        dist.destroy_process_group()

    print0("Training complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"❌ Training failed with error: {e}", flush=True)
        print("Full traceback:", flush=True)
        traceback.print_exc()
        raise
