"""
SFT (Supervised Fine-Tuning) script for RunPod GPU execution.

This script:
1. Downloads config, midtraining checkpoint, and tokenizer from S3
2. Downloads pre-processed datasets from S3
3. Trains on conversational data with specialized learning rates
4. Uploads SFT checkpoint to S3
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

# Suppress verbose logging
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from dagster_nanochat.nanochat.common import print0
from dagster_nanochat.nanochat.gpt import GPT, GPTConfig
from dagster_nanochat.tasks.common import Task
from dagster_nanochat.utils.data_generators import create_sft_data_generator
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


class JSONLDataset(Task):
    """Dataset loaded from JSONL file."""

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        # Load all conversations from JSONL
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    self.conversations.append(json.loads(line))

    def num_examples(self):
        return len(self.conversations)

    def get_example(self, index):
        return self.conversations[index]


def main():
    parser = argparse.ArgumentParser(description="SFT Training on RunPod")
    parser.add_argument("--s3-bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument(
        "--s3-config-key", type=str, required=True, help="S3 key for training config"
    )
    args = parser.parse_args()

    print0("=" * 80)
    print0("Starting Supervised Fine-Tuning (SFT) on RunPod")
    print0("=" * 80)
    print0(f"S3 Bucket: {args.s3_bucket}")
    print0(f"Config Key: {args.s3_config_key}")

    # Setup device (single GPU for SFT)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        torch.cuda.set_device(0)
        print0(f"Using device: cuda:0", flush=True)
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print0(f"No GPU available, using CPU", flush=True)

    # Initialize S3 client
    s3_client = get_s3_client()

    # Download and load config
    config_path = "config.json"
    print0("Downloading config from S3...", flush=True)
    download_from_s3(s3_client, args.s3_bucket, args.s3_config_key, config_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    print0(f"Config loaded: {json.dumps(config, indent=2)}", flush=True)

    # Extract config values
    model_tag = config["model_tag"]
    quick_mode = config["quick_mode"]
    s3_mid_checkpoint_key = config["s3_mid_checkpoint_key"]
    s3_tokenizer_key = config["s3_tokenizer_key"]
    s3_train_dataset_key = config["s3_train_dataset_key"]
    s3_val_dataset_key = config["s3_val_dataset_key"]
    s3_output_key = config["s3_output_key"]
    num_epochs = config["num_epochs"]
    device_batch_size = config["device_batch_size"]
    target_examples_per_step = config["target_examples_per_step"]
    max_seq_len = config["max_seq_len"]
    eval_every = config["eval_every"]
    eval_steps = config["eval_steps"]
    unembedding_lr = config["unembedding_lr"]
    embedding_lr = config["embedding_lr"]
    matrix_lr = config["matrix_lr"]
    weight_decay = config["weight_decay"]
    init_lr_frac = config["init_lr_frac"]

    # Download midtraining checkpoint from S3
    checkpoint_dir = "mid_checkpoint"
    print0("Downloading midtraining checkpoint from S3...", flush=True)
    checkpoint_tarball = "mid_checkpoint.tar.gz"
    download_from_s3(
        s3_client, args.s3_bucket, s3_mid_checkpoint_key, checkpoint_tarball
    )

    # Extract checkpoint
    print0("Extracting midtraining checkpoint...", flush=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with tarfile.open(checkpoint_tarball, "r:gz") as tar:
        tar.extractall(checkpoint_dir)
    print0(f"Extracted to: {checkpoint_dir}", flush=True)

    # Load checkpoint metadata
    checkpoint_metadata_path = os.path.join(checkpoint_dir, "checkpoint.json")
    with open(checkpoint_metadata_path, "r") as f:
        checkpoint_metadata = json.load(f)

    model_config_dict = checkpoint_metadata["model_config"]
    model_config = GPTConfig(**model_config_dict)

    # Initialize model
    print0("Initializing model...", flush=True)
    model = GPT(model_config)
    print0(
        f"Model initialized: {model_config.n_layer} layers, "
        f"{model_config.n_head} heads, {model_config.n_embd} dim",
        flush=True,
    )

    # Load model weights
    model_path = os.path.join(checkpoint_dir, "model.pt")
    print0("Loading model weights...", flush=True)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.train()
    print0("Midtraining model weights loaded", flush=True)

    # Download tokenizer from S3
    tokenizer_path = "tokenizer.json"
    print0("Downloading tokenizer from S3...", flush=True)
    download_from_s3(s3_client, args.s3_bucket, s3_tokenizer_key, tokenizer_path)

    with open(tokenizer_path, "r") as f:
        tokenizer_data = json.load(f)

    base_vocab_size = tokenizer_data["vocab_size"]
    tokenizer = create_tokenizer_with_special_tokens(tokenizer_data, base_vocab_size)
    print0(f"Tokenizer loaded with vocab size: {model_config.vocab_size:,}", flush=True)

    # Download pre-processed datasets from S3
    train_dataset_path = "train.jsonl"
    val_dataset_path = "val.jsonl"

    print0("Downloading training dataset from S3...", flush=True)
    download_from_s3(
        s3_client, args.s3_bucket, s3_train_dataset_key, train_dataset_path
    )

    print0("Downloading validation dataset from S3...", flush=True)
    download_from_s3(s3_client, args.s3_bucket, s3_val_dataset_key, val_dataset_path)

    # Load datasets
    print0("Loading datasets from JSONL files...", flush=True)
    train_dataset = JSONLDataset(train_dataset_path)
    val_dataset = JSONLDataset(val_dataset_path)

    train_size = len(train_dataset)
    val_size = len(val_dataset)

    print0(f"Training dataset: {train_size:,} examples", flush=True)
    print0(f"Validation dataset: {val_size:,} examples", flush=True)

    # Calculate training iterations
    examples_per_step = device_batch_size
    grad_accum_steps = max(1, target_examples_per_step // examples_per_step)
    num_iterations = (train_size // target_examples_per_step) * num_epochs

    if quick_mode:
        num_iterations = min(num_iterations, 50)  # Cap at 50 iterations for quick mode

    print0(f"Device batch size: {device_batch_size}", flush=True)
    print0(f"Examples per step: {examples_per_step}", flush=True)
    print0(f"Gradient accumulation steps: {grad_accum_steps}", flush=True)
    print0(f"Total iterations: {num_iterations}", flush=True)

    # Setup optimizers with specialized learning rates
    print0("Setting up optimizers with specialized learning rates...", flush=True)
    optimizers = model.setup_optimizers(
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
    )

    # Set initial learning rate as a fraction of base LR
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["lr"] * init_lr_frac
            group["initial_lr"] = group["lr"]

    print0(f"Optimizers initialized with {len(optimizers)} groups", flush=True)
    print0(f"   Unembedding LR: {unembedding_lr}", flush=True)
    print0(f"   Embedding LR: {embedding_lr}", flush=True)
    print0(f"   Matrix LR: {matrix_lr}", flush=True)
    print0(f"   Init LR Fraction: {init_lr_frac}", flush=True)

    # Create SFT data generator (conversation-level batching)
    print0("Creating SFT data generator...", flush=True)
    sft_data_gen = create_sft_data_generator(
        train_dataset,
        val_dataset,
        tokenizer,
        device_batch_size,
        device,
        num_epochs,
    )

    train_loader = sft_data_gen("train")
    build_val_loader = lambda: sft_data_gen("val")

    # Learning rate scheduler (linear decay)
    def get_lr_multiplier(it):
        return 1.0 - it / num_iterations

    # Training loop
    print0("Starting training loop...", flush=True)
    step = 0
    train_losses = []
    val_losses = []
    min_val_loss = float("inf")
    final_loss = None

    train_iter = iter(train_loader)
    start_time = time.time()

    for step in range(num_iterations):
        last_step = step == num_iterations - 1

        # Validation
        if last_step or (eval_every > 0 and step % eval_every == 0):
            model.eval()
            val_iter = iter(build_val_loader())
            losses = []

            max_val_batches = max(1, eval_steps)
            num_batches = 0

            with torch.no_grad():
                for x_val, y_val in val_iter:
                    loss = model(x_val, y_val)
                    losses.append(loss.item())
                    num_batches += 1
                    if num_batches >= max_val_batches:
                        break

            if losses:
                val_loss = sum(losses) / len(losses)
                val_losses.append(val_loss)
                min_val_loss = min(min_val_loss, val_loss)
                print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}", flush=True)

            model.train()

        if last_step:
            break

        # Training step with gradient accumulation
        num_tokens = 0
        for micro_step in range(grad_accum_steps):
            try:
                train_inputs, train_targets = next(train_iter)
            except StopIteration:
                # Restart iterator if we run out
                train_iter = iter(train_loader)
                train_inputs, train_targets = next(train_iter)

            loss = model(train_inputs, train_targets)
            train_loss = loss.detach().item()
            loss = loss / grad_accum_steps
            loss.backward()

            num_tokens += (train_targets >= 0).sum().item()

        # Update learning rate
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm

        # Optimizer step
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        # Logging
        train_losses.append(train_loss)
        final_loss = train_loss

        if step % 10 == 0 or last_step:
            print0(
                f"Step {step:05d}/{num_iterations:05d} | "
                f"Train loss: {train_loss:.6f} | "
                f"LR multiplier: {lrm:.6f} | "
                f"Tokens: {num_tokens:,}",
                flush=True,
            )

        # Break condition
        if step >= num_iterations:
            break

    training_time = time.time() - start_time
    print0(f"Training complete in {training_time:.1f}s", flush=True)

    # Save checkpoint
    print0("Saving checkpoint...", flush=True)
    output_dir = "checkpoint"
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    output_model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), output_model_path)
    print0(f"Model weights saved to: {output_model_path}", flush=True)

    # Save tokenizer
    output_tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    with open(output_tokenizer_path, "w") as f:
        json.dump(tokenizer_data, f, indent=2)
    print0(f"Tokenizer saved to: {output_tokenizer_path}", flush=True)

    # Save checkpoint metadata
    output_checkpoint_path = os.path.join(output_dir, "checkpoint.json")

    checkpoint_metadata = {
        "model_config": model_config_dict,
        "training": {
            "num_epochs": num_epochs,
            "num_iterations": num_iterations,
            "device_batch_size": device_batch_size,
            "target_examples_per_step": target_examples_per_step,
            "grad_accum_steps": grad_accum_steps,
            "quick_mode": quick_mode,
            "train_dataset_size": train_size,
            "val_dataset_size": val_size,
        },
        "optimization": {
            "unembedding_lr": unembedding_lr,
            "embedding_lr": embedding_lr,
            "matrix_lr": matrix_lr,
            "weight_decay": weight_decay,
            "init_lr_frac": init_lr_frac,
        },
        "results": {
            "step": step,
            "final_train_loss": final_loss,
            "min_val_loss": min_val_loss if min_val_loss != float("inf") else None,
            "training_time_seconds": training_time,
        },
        "device": str(device),
        "source_checkpoint": f"midtraining/{model_tag}/checkpoint",
    }

    with open(output_checkpoint_path, "w") as f:
        json.dump(checkpoint_metadata, f, indent=2)
    print0(f"Checkpoint metadata saved to: {output_checkpoint_path}", flush=True)

    # Create tarball
    checkpoint_tarball = "checkpoint.tar.gz"
    print0(f"Creating tarball: {checkpoint_tarball}", flush=True)
    with tarfile.open(checkpoint_tarball, "w:gz") as tar:
        tar.add(output_dir, arcname=".")
    print0(f"Tarball created", flush=True)

    # Upload checkpoint to S3
    print0("Uploading checkpoint to S3...", flush=True)
    upload_to_s3(s3_client, checkpoint_tarball, args.s3_bucket, s3_output_key)

    print0("=" * 80)
    print0("SFT Training Complete!")
    print0("=" * 80)
    print0(f"Final train loss: {final_loss:.6f}")
    print0(f"Min validation loss: {min_val_loss:.6f}")
    print0(f"Training time: {training_time:.1f}s")
    print0(f"Checkpoint uploaded to: s3://{args.s3_bucket}/{s3_output_key}")


if __name__ == "__main__":
    main()
