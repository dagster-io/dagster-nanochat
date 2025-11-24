"""Dagster assets for nanochat training pipeline."""

import json
import os
import shutil
import tarfile
import time
from urllib.parse import urlparse

import dagster as dg
from dagster_aws.s3 import S3Resource

# Local Rust BPE tokenizer
import rustbpe

# Configuration and constants
from dagster_nanochat.defs.config import (
    BASE_URL,
    CHECKPOINT_DIRECTORY,
    CONVERSATION_SPECIAL_TOKENS,
    FILE_DIRECTORY,
    HF_DATASETS_CACHE,
    S3_BUCKET_NAME,
    SFT_CHECKPOINT_DIRECTORY,
    SFT_DATASETS_CACHE,
    TRAINING_SET,
    BaseTrainingConfig,
    ChatInferenceConfig,
    MidtrainingConfig,
    SFTConfig,
)
from dagster_nanochat.defs.runpod_resource import RunPodResource
from dagster_nanochat.defs.serverless_resource import ServerlessResource

# Task datasets
from dagster_nanochat.tasks.arc import ARC
from dagster_nanochat.tasks.common import TaskMixture
from dagster_nanochat.tasks.customjson import CustomJSON
from dagster_nanochat.tasks.gsm8k import GSM8K
from dagster_nanochat.tasks.mmlu import MMLU
from dagster_nanochat.tasks.smoltalk import SmolTalk
from dagster_nanochat.tasks.spellingbee import SimpleSpelling, SpellingBee

# Utilities
from dagster_nanochat.utils import (
    download_file,
)
from dagster_nanochat.utils.checkpoint_utils import wait_and_download_checkpoint

# =============================================================================
# Tokenizer Training
# =============================================================================


raw_data = dg.AssetSpec(
    "huggface_karpathy_datasets",
    description="Raw data from Hugging Face Karpathy datasets",
    metadata={
        "url": dg.MetadataValue.url(BASE_URL),
    },
)


@dg.asset(
    deps=[raw_data],
    partitions_def=dg.StaticPartitionsDefinition(TRAINING_SET),
    group_name="tokenizer",
)
def training_files(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Download training data shards from HuggingFace.

    This implements nanochat pipeline ingestion:
    - Downloads parquet files from Karpathy's FineWeb-Edu dataset
    - Partitioned across 1822 shards (shard_00000 through shard_01821), excluding 1823 for validation
    - Each partition can download in parallel for faster ingestion
    - Files are saved to data/raw/ directory
    """
    url_path = context.partition_key
    filename = os.path.basename(urlparse(url_path).path)

    # Use absolute path to avoid any UPath protocol issues
    file_path = os.path.abspath(os.path.join(FILE_DIRECTORY, filename))

    download_file(url_path, file_path)

    return dg.MaterializeResult(
        metadata={
            "file_path": dg.MetadataValue.text(str(file_path)),
            "url": dg.MetadataValue.url(url_path),
        }
    )


@dg.asset(
    kinds={"rust", "s3"},
    deps=[training_files],
    group_name="tokenizer",
    backfill_policy=dg.BackfillPolicy.single_run(),
)
def tokenizer_training(
    context: dg.AssetExecutionContext,
    s3: S3Resource,
) -> dg.MaterializeResult:
    """
    Train a single BPE tokenizer on training data using Rust.

    This implements tokenizer training of the nanochat pipeline:
    - Uses high-performance Rust BPE implementation
    - Trains on ALL parquet shards (streaming from data/raw/)
    - Creates ONE canonical tokenizer file: data/tokenizer/tokenizer.json
    - Learns merge rules optimized for the full dataset distribution
    - Uses GPT-4 regex pattern for tokenization

    Note: Always trains on all available data for best tokenizer quality.
    Tokenizer training is fast enough that there's no need for a "quick mode".
    """
    context.log.info("Starting tokenizer training on ALL available data...")

    # Read all .parquet files in the training directory
    raw_dir = os.path.abspath(FILE_DIRECTORY)
    training_files_list = sorted(
        [
            os.path.join(raw_dir, f)
            for f in os.listdir(raw_dir)
            if f.endswith(".parquet")
        ]
    )

    # Create tokenizer instance and train on all files
    tokenizer = rustbpe.Tokenizer()
    max_chars = 10_000_000_000  # 10 billion characters
    vocab_size = 10000

    # Train from multiple parquet files
    tokenizer.train_from_parquet_files(
        parquet_paths=training_files_list,
        vocab_size=vocab_size,
        pattern=None,  # Use default GPT-4 pattern
        text_column="text",
        doc_cap=0,  # No document cap (0 = unlimited)
        max_chars=max_chars,
    )

    context.log.info("Tokenizer training complete!")

    # Get mergeable ranks and pattern
    mergeable_ranks = tokenizer.get_mergeable_ranks()
    pattern = tokenizer.get_pattern()

    # Create canonical tokenizer directory
    tokenizer_dir = os.path.abspath("data/tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    # Save to the canonical location
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")

    # Convert mergeable ranks to JSON-serializable format
    with open(tokenizer_path, "w") as f:
        json.dump(
            {
                "vocab_size": len(mergeable_ranks),
                "pattern": pattern,
                "mergeable_ranks": [
                    {"bytes": list(bytes_val), "token_id": int(token_id)}
                    for bytes_val, token_id in mergeable_ranks
                ],
                "num_training_files": len(training_files_list),
                "max_chars_processed": max_chars,
            },
            f,
            indent=2,
        )

    context.log.info(f"Saved tokenizer to: {tokenizer_path}")

    # Upload tokenizer to S3 for RunPod access
    context.log.info("Uploading tokenizer to S3...")
    s3_key = "tokenizer.json"

    with open(tokenizer_path, "rb") as f:
        s3.get_client().upload_fileobj(f, S3_BUCKET_NAME, s3_key)

    context.log.info(f"Uploaded tokenizer to s3://{S3_BUCKET_NAME}/{s3_key}")

    return dg.MaterializeResult(
        metadata={
            "tokenizer_path": dg.MetadataValue.text(tokenizer_path),
            "s3_key": dg.MetadataValue.text(f"s3://{S3_BUCKET_NAME}/{s3_key}"),
            "vocab_size": len(mergeable_ranks),
            "pattern": dg.MetadataValue.text(pattern),
            "num_training_files": len(training_files_list),
            "max_chars_processed": max_chars,
        }
    )


# =============================================================================
# Base Model Training
# =============================================================================


@dg.asset(
    kinds={"docker"},
    group_name="image",
)
def nanochat_training_image(context: dg.AssetExecutionContext) -> str:
    """Docker image containing all nanochat training code and dependencies."""
    return "dhume/dagster-nanochat:latest"


@dg.asset(
    deps=[tokenizer_training],
    kinds={"s3"},
    group_name="base_model_training",
)
def model_run_config(
    context: dg.AssetExecutionContext, config: BaseTrainingConfig, s3: S3Resource
) -> str:
    """Upload training configuration to S3 for RunPod to download."""
    depth = config.depth
    max_seq_len = config.max_seq_len
    device_batch_size = config.device_batch_size
    total_batch_size = config.total_batch_size
    num_iterations = config.num_iterations
    eval_every = config.eval_every

    # Load tokenizer info to get vocab size
    tokenizer_path = os.path.abspath("data/tokenizer/tokenizer.json")
    with open(tokenizer_path, "r") as f:
        tokenizer_data = json.load(f)
    vocab_size = tokenizer_data["vocab_size"]

    # Read which shards actually exist in the raw directory
    # These are the files that training_files downloaded and tokenizer was trained on
    raw_dir = os.path.abspath(FILE_DIRECTORY)
    train_shard_ids = []

    if os.path.exists(raw_dir):
        for filename in sorted(os.listdir(raw_dir)):
            if filename.startswith("shard_") and filename.endswith(".parquet"):
                # Extract shard number from filename: shard_XXXXX.parquet
                shard_num = int(filename.split("_")[1].split(".")[0])
                train_shard_ids.append(shard_num)

    if not train_shard_ids:
        raise ValueError(f"No training shards found in {raw_dir}")

    context.log.info(f"Found {len(train_shard_ids)} training shards in {raw_dir}")
    context.log.info(
        f"   Shard IDs: {train_shard_ids[:10]}{'...' if len(train_shard_ids) > 10 else ''}"
    )

    # Prepare training config for remote script
    training_config = {
        "depth": depth,
        "max_seq_len": max_seq_len,
        "device_batch_size": device_batch_size,
        "total_batch_size": total_batch_size,
        "num_iterations": num_iterations,
        "eval_every": eval_every,
        "vocab_size": vocab_size,
        "train_shard_ids": train_shard_ids,  # List of specific shard IDs to use
        "unembedding_lr": config.unembedding_lr,
        "embedding_lr": config.embedding_lr,
        "matrix_lr": config.matrix_lr,
        "weight_decay": config.weight_decay,
        "grad_clip": config.grad_clip,
        "warmup_ratio": config.warmup_ratio,
        "warmdown_ratio": config.warmdown_ratio,
        "final_lr_frac": config.final_lr_frac,
        "eval_tokens": config.eval_tokens,
        "sample_every": config.sample_every,
        "special_tokens": CONVERSATION_SPECIAL_TOKENS,
        "s3_bucket": S3_BUCKET_NAME,
        "s3_tokenizer_key": "tokenizer.json",
        "s3_checkpoint_key": f"checkpoints/{config.model_tag}/checkpoint.tar.gz",
    }

    # Upload config to S3
    config_s3_key = f"configs/{config.model_tag}/config_{context.run_id[:8]}.json"
    context.log.info(f"Uploading config to S3: s3://{S3_BUCKET_NAME}/{config_s3_key}")

    config_json = json.dumps(training_config, indent=2)
    s3.get_client().put_object(
        Bucket=S3_BUCKET_NAME,
        Key=config_s3_key,
        Body=config_json.encode("utf-8"),
        ContentType="application/json",
    )

    return config_s3_key


@dg.asset(
    deps=[
        tokenizer_training,
        raw_data,
        nanochat_training_image,
        model_run_config,
    ],
    kinds={"pytorch", "runpod"},
    group_name="base_model_training",
    backfill_policy=dg.BackfillPolicy.single_run(),
)
def base_model_checkpoint(
    context: dg.AssetExecutionContext,
    config: BaseTrainingConfig,
    runpod: RunPodResource,
    s3: S3Resource,
    nanochat_training_image: str,
    model_run_config: str,
) -> dg.MaterializeResult:
    """
    Train a base Transformer model from scratch on RunPod GPU using DDP.

    This asset orchestrates training on a remote RunPod GPU instance using a pre-built
    Docker image that contains all code, dependencies, and rustbpe tokenizer:

    1. Uploads training config to S3
    2. Creates GPU pod (2-8 GPUs, default 2) with nanochat Docker image
    3. Pod downloads config and tokenizer from S3 (using RunPod secrets for AWS)
    4. Pod downloads training data from HuggingFace (1822 shards + validation)
    5. Executes distributed training with DDP (streams logs to Dagster)
    6. Pod uploads checkpoint to S3
    7. Downloads checkpoint from S3 to local machine
    8. Terminates pod automatically

    The training implements:
    - GPT model with random initialization
    - Distributed Data Parallel (DDP) training across 2+ GPUs
    - Streaming tokenized data from training files
    - Validation on canonical shard 1822 to track generalization
    - Validation BPB (bits per byte) metric

    Quick mode reduces model depth and iterations but uses the same training data
    to ensure consistency with the tokenizer vocabulary.
    """

    context.log.info("Training base model on RunPod GPU (2+ GPUs, DDP)")

    # Get training config from S3 key
    s3_config_key = model_run_config

    # Download config to get checkpoint key for later
    config_obj = s3.get_client().get_object(Bucket=S3_BUCKET_NAME, Key=s3_config_key)
    training_config = json.loads(config_obj["Body"].read().decode("utf-8"))

    # Build training command that exits container after completion
    gpu_count = runpod.gpu_count
    train_cmd = (
        f"bash -c 'torchrun --standalone --nproc_per_node={gpu_count} "
        f"/workspace/scripts/train_base_remote.py "
        f"--s3-bucket {S3_BUCKET_NAME} "
        f"--s3-config-key {s3_config_key} && exit 0 || exit 1'"
    )

    # Run training on pod
    pod_name = f"nanochat-base-{config.model_tag}-{context.run_id[:8]}"
    context.log.info(f"Config: s3://{S3_BUCKET_NAME}/{s3_config_key}")
    context.log.info(f"Image: {nanochat_training_image}")
    context.log.info(f"Training on {gpu_count} GPUs with DDP")

    pod = runpod.run_pod(
        pod_name, nanochat_training_image, train_cmd, context, volume_in_gb=300
    )
    pod_id = pod["id"]

    try:
        # Wait for training to complete (monitor pod status)
        context.log.info(f"Training in progress on pod: {pod_id}")
        context.log.info("   Check RunPod dashboard for live logs")

        max_wait = 7200  # 2 hours
        check_interval = 60  # Check every minute
        start_time = time.time()

        # Poll for checkpoint completion instead of pod exit
        # (RunPod containers restart on exit, so we check S3 instead)
        s3_checkpoint_key = training_config["s3_checkpoint_key"]
        checkpoint_found = False

        while time.time() - start_time < max_wait:
            time.sleep(check_interval)

            elapsed = int(time.time() - start_time)

            # Check if checkpoint has been uploaded to S3
            try:
                s3.get_client().head_object(
                    Bucket=S3_BUCKET_NAME, Key=s3_checkpoint_key
                )
                context.log.info(f"[{elapsed}s] Checkpoint detected in S3!")
                checkpoint_found = True
                break
            except:
                # Checkpoint not ready yet, continue waiting
                context.log.info(f"[{elapsed}s] Waiting for training to complete...")

                # Also check pod status for early failure detection
                pod_status = runpod.get_pod(pod_id)
                if isinstance(pod_status, list):
                    pod_status = pod_status[0]

                desired_status = pod_status.get("desiredStatus")
                if desired_status == "FAILED":
                    raise RuntimeError("Training pod failed - check RunPod logs")

        if not checkpoint_found:
            raise TimeoutError(
                f"Training exceeded {max_wait}s timeout. Checkpoint not found in S3."
            )

        if time.time() - start_time >= max_wait:
            raise TimeoutError(f"Training exceeded {max_wait}s timeout")

        # Training complete - checkpoint is in S3
        context.log.info("Downloading checkpoint from S3...")
        context.log.info(f"   S3 key: {s3_checkpoint_key}")

        # Create local checkpoint directory
        model_tag = (
            config.model_tag if config.model_tag else f"d{training_config['depth']}"
        )
        checkpoint_dir = os.path.abspath(os.path.join(CHECKPOINT_DIRECTORY, model_tag))
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Download the tarball from S3
        local_tarball = f"{checkpoint_dir}/checkpoint.tar.gz"

        context.log.info(
            f"Downloading from s3://{S3_BUCKET_NAME}/{s3_checkpoint_key}..."
        )
        s3.get_client().download_file(S3_BUCKET_NAME, s3_checkpoint_key, local_tarball)

        # Extract tarball
        context.log.info("Extracting checkpoint...")

        with tarfile.open(local_tarball, "r:gz") as tar:
            tar.extractall(checkpoint_dir)

        # Remove tarball
        os.remove(local_tarball)

        # Load checkpoint metadata for result
        with open(f"{checkpoint_dir}/checkpoint.json") as f:
            checkpoint_metadata = json.load(f)

        context.log.info("Remote training complete!")
        context.log.info(
            f"🎯 Best validation BPB: {checkpoint_metadata['best_val_bpb']:.4f}"
        )

        return dg.MaterializeResult(
            metadata={
                "checkpoint_dir": dg.MetadataValue.path(checkpoint_dir),
                "model_tag": model_tag,
                "num_layers": training_config["depth"],
                "vocab_size": training_config["vocab_size"],
                "num_iterations": training_config["num_iterations"],
                "training_time_seconds": checkpoint_metadata["training_time_seconds"],
                "final_loss": checkpoint_metadata["final_loss"],
                "best_val_bpb": checkpoint_metadata["best_val_bpb"],
                "device": checkpoint_metadata["device"],
                "pod_id": pod_id,
                "execution_mode": "runpod",
                "gpu_count": runpod.gpu_count,
                "ddp_world_size": checkpoint_metadata["ddp_world_size"],
            }
        )

    finally:
        # Always cleanup pod
        context.log.info("Cleaning up RunPod instance...")
        runpod.terminate_pod(pod_id, context)


# =============================================================================
# Midtraining
# =============================================================================


@dg.asset(
    deps=[raw_data],
    kinds={"huggingface", "s3"},
    group_name="midtraining",
)
def midtraining_datasets(
    context: dg.AssetExecutionContext,
    config: MidtrainingConfig,
    s3: S3Resource,
) -> dg.MaterializeResult:
    """
    Download midtraining datasets from HuggingFace, serialize, and upload to S3.

    This asset:
    - Downloads conversational and reasoning datasets from HuggingFace
    - Serializes train/val datasets as JSONL files
    - Uploads to S3 for use in RunPod training

    Datasets:
    - SmolTalk: conversational data
    - MMLU: multiple choice reasoning
    - GSM8K: math word problems
    - Spelling tasks: SimpleSpelling + SpellingBee

    Returns:
        MaterializeResult with S3 paths and dataset metadata
    """
    context.log.info("Preparing midtraining datasets...")

    # Set HuggingFace cache to local directory within the repo
    hf_cache_dir = os.path.abspath(HF_DATASETS_CACHE)
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
    os.environ["HF_HOME"] = hf_cache_dir
    context.log.info(f"HuggingFace cache directory: {hf_cache_dir}")

    # Create cache directory for metadata
    midtraining_cache = "data/midtraining_datasets"
    os.makedirs(midtraining_cache, exist_ok=True)

    # Initialize datasets (triggers download if needed)
    # Use smaller datasets for $1 tier (max_seq_len == 512)
    if config.max_seq_len == 512:
        context.log.info("$1 tier: Loading small dataset subsets...")
        train_datasets = [
            SmolTalk(split="train", stop=1000),  # 1K rows
            MMLU(subset="auxiliary_train", split="train", stop=500),  # 500 rows
        ]  # Total: ~1.5K rows
        val_datasets = [
            SmolTalk(split="test", stop=100),  # 100 rows
            MMLU(subset="all", split="test", stop=100),  # 100 rows
        ]
    else:
        context.log.info("$10/$100 tier: Loading complete datasets...")
        train_datasets = [
            SmolTalk(split="train"),  # Full training set
            MMLU(subset="auxiliary_train", split="train"),  # Full auxiliary train
            GSM8K(subset="main", split="train"),  # ~8K rows
            SimpleSpelling(size=200000, split="train"),  # 200K rows
            SpellingBee(size=80000, split="train"),  # 80K rows
        ]
        val_datasets = [
            SmolTalk(split="test"),  # Full test set
            MMLU(subset="all", split="test", stop=5200),  # 5.2K rows
            GSM8K(subset="main", split="test", stop=420),  # 420 rows
        ]

    # Create task mixtures
    from dagster_nanochat.tasks.common import TaskMixture

    train_dataset = TaskMixture(train_datasets)
    val_dataset = TaskMixture(val_datasets)

    train_size = len(train_dataset)
    val_size = len(val_dataset)

    context.log.info(f"Training dataset: {train_size:,} examples")
    context.log.info(f"Validation dataset: {val_size:,} examples")

    # Serialize and upload to S3
    context.log.info("Serializing datasets to JSONL...")
    model_tag = config.model_tag if config.model_tag else "d4"

    # Create temporary directory for serialization
    temp_dir = os.path.join("data", "temp_midtraining_datasets")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Serialize train dataset
        train_path = os.path.join(temp_dir, "train.jsonl")
        context.log.info(f"Serializing {len(train_dataset)} training examples...")
        with open(train_path, "w") as f:
            for i in range(len(train_dataset)):
                conversation = train_dataset[i]
                f.write(json.dumps(conversation) + "\n")
        context.log.info(f"Saved training data to: {train_path}")

        # Serialize val dataset
        val_path = os.path.join(temp_dir, "val.jsonl")
        context.log.info(f"Serializing {len(val_dataset)} validation examples...")
        with open(val_path, "w") as f:
            for i in range(len(val_dataset)):
                conversation = val_dataset[i]
                f.write(json.dumps(conversation) + "\n")
        context.log.info(f"Saved validation data to: {val_path}")

        # Upload to S3
        s3_client = s3.get_client()
        train_s3_key = f"datasets/midtraining/{model_tag}/train.jsonl"
        val_s3_key = f"datasets/midtraining/{model_tag}/val.jsonl"

        context.log.info(f"Uploading train dataset to S3: {train_s3_key}")
        s3_client.upload_file(train_path, S3_BUCKET_NAME, train_s3_key)

        context.log.info(f"Uploading val dataset to S3: {val_s3_key}")
        s3_client.upload_file(val_path, S3_BUCKET_NAME, val_s3_key)

        context.log.info("All datasets uploaded to S3")

    finally:
        # Cleanup temporary files
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            context.log.info(f"Cleaned up temporary directory: {temp_dir}")

    # Store metadata for reference
    dataset_info = {
        "model_tag": config.model_tag,
        "train_size": train_size,
        "val_size": val_size,
        "num_train_datasets": len(train_datasets),
        "num_val_datasets": len(val_datasets),
        "train_datasets": [type(ds).__name__ for ds in train_datasets],
        "val_datasets": [type(ds).__name__ for ds in val_datasets],
    }

    metadata_path = os.path.join(midtraining_cache, "dataset_info.json")
    with open(metadata_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    context.log.info(f"Dataset info saved to: {metadata_path}")

    return dg.MaterializeResult(
        metadata={
            "train_s3_key": train_s3_key,
            "val_s3_key": val_s3_key,
            "train_size": train_size,
            "val_size": val_size,
            "model_tag": config.model_tag,
            "num_train_datasets": len(train_datasets),
            "num_val_datasets": len(val_datasets),
        }
    )


@dg.asset(
    deps=[midtraining_datasets],
    kinds={"s3"},
    group_name="midtraining",
)
def midtraining_run_config(
    context: dg.AssetExecutionContext,
    config: MidtrainingConfig,
    s3: S3Resource,
    midtraining_datasets: dg.MaterializeResult,
) -> str:
    """
    Create and upload midtraining configuration to S3.

    This asset prepares the training configuration including S3 paths to pre-serialized
    JSONL datasets, hyperparameters, and checkpoint paths. It also uploads the locally
    cached base checkpoint to S3 for the RunPod instance to download.
    """
    context.log.info("📝 Preparing midtraining configuration...")

    # Get model tag
    model_tag = config.model_tag if config.model_tag else "d4"

    # Upload base checkpoint to S3 (if not already there)
    context.log.info("Uploading base checkpoint to S3...")
    checkpoint_dir = os.path.abspath(os.path.join(CHECKPOINT_DIRECTORY, model_tag))

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(
            f"Base checkpoint not found at {checkpoint_dir}. "
            f"Please run base_model_checkpoint first."
        )

    # Create tarball of base checkpoint
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_tarball = tmp.name

    try:
        with tarfile.open(tmp_tarball, "w:gz") as tar:
            tar.add(checkpoint_dir, arcname=".")

        # Upload to S3 at a predictable location
        s3_base_checkpoint_key = f"checkpoints/{model_tag}/checkpoint.tar.gz"
        context.log.info(f"Uploading to s3://{S3_BUCKET_NAME}/{s3_base_checkpoint_key}")

        s3.get_client().upload_file(tmp_tarball, S3_BUCKET_NAME, s3_base_checkpoint_key)
        context.log.info("Base checkpoint uploaded to S3")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_tarball):
            os.remove(tmp_tarball)

    # Get dataset S3 keys from upstream asset
    train_s3_key = midtraining_datasets.metadata["train_s3_key"].value
    val_s3_key = midtraining_datasets.metadata["val_s3_key"].value
    train_size = midtraining_datasets.metadata["train_size"].value
    val_size = midtraining_datasets.metadata["val_size"].value

    context.log.info(
        f"Using pre-serialized datasets: {train_size} train, {val_size} val examples"
    )
    context.log.info(f"Train dataset S3: s3://{S3_BUCKET_NAME}/{train_s3_key}")
    context.log.info(f"Val dataset S3: s3://{S3_BUCKET_NAME}/{val_s3_key}")

    # Always use config values for training params
    max_seq_len = config.max_seq_len
    device_batch_size = config.device_batch_size
    total_batch_size = config.total_batch_size
    num_iterations = config.num_iterations
    eval_every = config.eval_every
    eval_tokens = config.eval_tokens

    # Prepare training configuration
    training_config = {
        # Dataset paths (pre-processed JSONL on S3)
        "s3_bucket": S3_BUCKET_NAME,
        "s3_train_dataset_key": train_s3_key,
        "s3_val_dataset_key": val_s3_key,
        # Training hyperparameters
        "max_seq_len": max_seq_len,
        "device_batch_size": device_batch_size,
        "total_batch_size": total_batch_size,
        "num_iterations": num_iterations,
        "eval_every": eval_every,
        "eval_tokens": eval_tokens,
        "unembedding_lr": config.unembedding_lr,
        "embedding_lr": config.embedding_lr,
        "matrix_lr": config.matrix_lr,
        "init_lr_frac": config.init_lr_frac,
        "weight_decay": config.weight_decay,
        # Model and checkpoint paths
        "model_tag": model_tag,
        "s3_base_checkpoint_key": f"checkpoints/{model_tag}/checkpoint.tar.gz",
        "s3_tokenizer_key": "tokenizer.json",
        "s3_output_key": f"checkpoints/{model_tag}-mid/checkpoint.tar.gz",
    }

    # Upload config to S3
    config_s3_key = f"configs/midtraining/{model_tag}/config_{context.run_id[:8]}.json"
    context.log.info(f"Uploading config to S3: s3://{S3_BUCKET_NAME}/{config_s3_key}")

    config_json = json.dumps(training_config, indent=2)
    s3.get_client().put_object(
        Bucket=S3_BUCKET_NAME,
        Key=config_s3_key,
        Body=config_json.encode("utf-8"),
        ContentType="application/json",
    )

    context.log.info(
        f"Config uploaded with datasets: {train_size} train examples, {val_size} val examples"
    )

    return config_s3_key


@dg.asset(
    deps=[
        base_model_checkpoint,
        midtraining_run_config,
        nanochat_training_image,
        midtraining_datasets,
    ],
    kinds={"pytorch", "runpod"},
    group_name="midtraining",
    backfill_policy=dg.BackfillPolicy.single_run(),
)
def midtraining_checkpoint(
    context: dg.AssetExecutionContext,
    config: MidtrainingConfig,
    runpod: RunPodResource,
    s3: S3Resource,
    midtraining_run_config: str,
    nanochat_training_image: str,
) -> dg.MaterializeResult:
    """
    Midtrain the model on conversational data using RunPod GPU.

    This runs midtraining on RunPod:
    - Downloads base checkpoint from S3
    - Downloads pre-serialized JSONL datasets from S3
    - Extends vocabulary for conversation tokens
    - Trains on conversational data
    - Uploads midtrained checkpoint to S3
    """
    context.log.info("Running midtraining on RunPod GPU...")

    # Get training config from S3 key
    s3_config_key = midtraining_run_config

    # Download config to get checkpoint key
    config_obj = s3.get_client().get_object(Bucket=S3_BUCKET_NAME, Key=s3_config_key)
    training_config = json.loads(config_obj["Body"].read().decode("utf-8"))

    # Build training command (single GPU for simplicity)
    # Midtraining is fast enough with 1 GPU and avoids DDP complexity
    train_cmd = (
        f"bash -c 'python /workspace/scripts/train_midtraining_remote.py "
        f"--s3-bucket {S3_BUCKET_NAME} "
        f"--s3-config-key {s3_config_key} && exit 0 || exit 1'"
    )

    # Run training on pod (single GPU for midtraining)
    model_tag = config.model_tag if config.model_tag else "d4"
    pod_name = f"nanochat-mid-{model_tag}-{context.run_id[:8]}"
    context.log.info(f"Config: s3://{S3_BUCKET_NAME}/{s3_config_key}")
    context.log.info(f"Image: {nanochat_training_image}")
    context.log.info(f"Training on single GPU (midtraining is fast)")

    pod = runpod.run_pod(
        pod_name,
        nanochat_training_image,
        train_cmd,
        context,
        volume_in_gb=300,
        gpu_count=1,
    )
    pod_id = pod["id"]

    try:
        # Wait for training to complete and download checkpoint
        context.log.info(f"Training in progress on pod: {pod_id}")
        context.log.info("   Check RunPod dashboard for live logs")

        s3_checkpoint_key = training_config["s3_output_key"]
        checkpoint_dir = os.path.abspath(
            os.path.join(CHECKPOINT_DIRECTORY, f"{model_tag}-mid")
        )

        checkpoint_metadata = wait_and_download_checkpoint(
            s3_client=s3.get_client(),
            bucket=S3_BUCKET_NAME,
            s3_key=s3_checkpoint_key,
            output_dir=checkpoint_dir,
            context=context,
            max_wait_seconds=7200,  # 2 hours
            check_interval=60,  # Check every minute
        )

        context.log.info("Midtraining complete!")
        context.log.info(
            f"   Final loss: {checkpoint_metadata.get('final_loss', 'N/A')}"
        )
        context.log.info(
            f"   Best val loss: {checkpoint_metadata.get('min_val_loss', 'N/A')}"
        )
        context.log.info(
            f"   Training time: {checkpoint_metadata.get('training_time_seconds', 0):.1f}s"
        )

        return dg.MaterializeResult(
            metadata={
                "checkpoint_dir": dg.MetadataValue.path(checkpoint_dir),
                "model_tag": f"{model_tag}-mid",
                "training_time_seconds": checkpoint_metadata.get(
                    "training_time_seconds"
                ),
                "final_loss": checkpoint_metadata.get("final_loss"),
                "min_val_loss": checkpoint_metadata.get("min_val_loss"),
                "pod_id": pod_id,
                "execution_mode": "runpod",
                "gpu_count": runpod.gpu_count,
            }
        )

    finally:
        # Always cleanup pod
        context.log.info("Cleaning up RunPod instance...")
        runpod.terminate_pod(pod_id, context)


# =============================================================================
# Supervised Fine-Tuning (SFT)
# =============================================================================


@dg.asset(
    deps=[raw_data],
    kinds={"huggingface", "s3"},
    group_name="supervised_fine_tuning",
)
def sft_datasets(
    context: dg.AssetExecutionContext,
    config: SFTConfig,
    s3: S3Resource,
) -> dg.MaterializeResult:
    """
    Download SFT datasets from HuggingFace, serialize, and upload to S3.

    This asset:
    - Downloads SFT-specific datasets from HuggingFace
    - Loads local identity conversations
    - Serializes train/val datasets as JSONL files
    - Uploads to S3 for use in RunPod training

    Datasets:
    - ARC (Easy + Challenge): reasoning tasks
    - GSM8K: math word problems
    - SmolTalk: conversational data
    - Identity conversations: synthetic identity data
    - Spelling tasks: SimpleSpelling + SpellingBee

    Quick mode uses smaller subsets for faster testing.

    Returns:
        MaterializeResult with S3 paths and dataset metadata
    """
    context.log.info("Preparing SFT datasets...")

    # Set HuggingFace cache to local directory within the repo
    hf_cache_dir = os.path.abspath(HF_DATASETS_CACHE)
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
    os.environ["HF_HOME"] = hf_cache_dir
    context.log.info(f"HuggingFace cache directory: {hf_cache_dir}")

    # Create cache directory for metadata
    os.makedirs(SFT_DATASETS_CACHE, exist_ok=True)

    # Initialize datasets (triggers download if needed)
    # Use smaller datasets for $1 tier (max_seq_len == 512)
    if config.max_seq_len == 512:
        context.log.info("$1 tier: Loading small dataset subsets...")
        train_datasets = [
            ARC(subset="ARC-Easy", split="train", stop=50),  # 50 rows
            ARC(subset="ARC-Challenge", split="train", stop=30),  # 30 rows
            GSM8K(subset="main", split="train", stop=100),  # 100 rows
            SmolTalk(split="train", stop=200),  # 200 rows
            CustomJSON(
                filepath=os.path.abspath("data/identity_conversations.jsonl")
            ),  # 25 rows
            SimpleSpelling(size=20, split="train"),  # 20 rows
            SpellingBee(size=20, split="train"),  # 20 rows
        ]  # Total: ~445 rows
        val_datasets = [
            SmolTalk(split="test", stop=100),  # 100 rows
        ]
    else:
        context.log.info("$10/$100 tier: Loading complete datasets...")
        train_datasets = [
            ARC(subset="ARC-Easy", split="train"),  # 2.3K rows
            ARC(subset="ARC-Challenge", split="train"),  # 1.1K rows
            GSM8K(subset="main", split="train"),  # 8K rows
            SmolTalk(split="train", stop=10_000),  # 10K rows
            CustomJSON(
                filepath=os.path.abspath("data/identity_conversations.jsonl")
            ),  # 25 rows
            SimpleSpelling(size=300, split="train"),  # 300 rows
            SpellingBee(size=300, split="train"),  # 300 rows
        ]  # Total: ~23K rows
        val_datasets = [
            SmolTalk(split="test", stop=1000),  # 1K rows
        ]

    # Create task mixtures
    train_dataset = TaskMixture(train_datasets)
    val_dataset = TaskMixture(val_datasets)

    train_size = len(train_dataset)
    val_size = len(val_dataset)

    context.log.info(f"Training dataset: {train_size:,} examples")
    context.log.info(f"Validation dataset: {val_size:,} examples")

    # Serialize and upload to S3
    context.log.info("Serializing datasets to JSONL...")
    model_tag = config.model_tag if config.model_tag else "d4"

    # Create temporary directory for serialization
    temp_dir = os.path.join("data", "temp_sft_datasets")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Serialize train dataset
        train_path = os.path.join(temp_dir, "train.jsonl")
        context.log.info(f"📝 Serializing {len(train_dataset)} training examples...")
        with open(train_path, "w") as f:
            for i in range(len(train_dataset)):
                conversation = train_dataset[i]
                f.write(json.dumps(conversation) + "\n")
        context.log.info(f"Saved training data to: {train_path}")

        # Serialize val dataset
        val_path = os.path.join(temp_dir, "val.jsonl")
        context.log.info(f"📝 Serializing {len(val_dataset)} validation examples...")
        with open(val_path, "w") as f:
            for i in range(len(val_dataset)):
                conversation = val_dataset[i]
                f.write(json.dumps(conversation) + "\n")
        context.log.info(f"Saved validation data to: {val_path}")

        # Upload to S3
        s3_client = s3.get_client()
        train_s3_key = f"datasets/sft/{model_tag}/train.jsonl"
        val_s3_key = f"datasets/sft/{model_tag}/val.jsonl"
        identity_s3_key = "datasets/identity_conversations.jsonl"

        context.log.info(f"Uploading train dataset to S3: {train_s3_key}")
        s3_client.upload_file(train_path, S3_BUCKET_NAME, train_s3_key)

        context.log.info(f"Uploading val dataset to S3: {val_s3_key}")
        s3_client.upload_file(val_path, S3_BUCKET_NAME, val_s3_key)

        # Upload identity conversations (local file)
        identity_path = os.path.abspath("data/identity_conversations.jsonl")
        if os.path.exists(identity_path):
            context.log.info(
                f"Uploading identity conversations to S3: {identity_s3_key}"
            )
            s3_client.upload_file(identity_path, S3_BUCKET_NAME, identity_s3_key)
        else:
            context.log.warning(
                f"Identity conversations file not found: {identity_path}"
            )

        context.log.info("All datasets uploaded to S3")

    finally:
        # Cleanup temporary files
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            context.log.info(f"Cleaned up temporary directory: {temp_dir}")

    # Store metadata for reference
    dataset_info = {
        "model_tag": config.model_tag,
        "train_size": train_size,
        "val_size": val_size,
        "num_train_datasets": len(train_datasets),
        "num_val_datasets": len(val_datasets),
        "train_datasets": [type(ds).__name__ for ds in train_datasets],
        "val_datasets": [type(ds).__name__ for ds in val_datasets],
    }

    metadata_path = os.path.join(SFT_DATASETS_CACHE, "dataset_info.json")
    with open(metadata_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    context.log.info(f"Dataset info saved to: {metadata_path}")

    return dg.MaterializeResult(
        metadata={
            "train_s3_key": train_s3_key,
            "val_s3_key": val_s3_key,
            "identity_s3_key": identity_s3_key,
            "train_size": train_size,
            "val_size": val_size,
            "model_tag": config.model_tag,
            "num_train_datasets": len(train_datasets),
            "num_val_datasets": len(val_datasets),
            "s3_bucket": S3_BUCKET_NAME,
            "hf_datasets_cache_dir": dg.MetadataValue.path(hf_cache_dir),
            "metadata_cache_dir": dg.MetadataValue.path(SFT_DATASETS_CACHE),
        }
    )


@dg.asset(
    deps=[sft_datasets],
    kinds={"s3"},
    group_name="supervised_fine_tuning",
)
def sft_run_config(
    context: dg.AssetExecutionContext,
    config: SFTConfig,
    s3: S3Resource,
) -> str:
    """
    Generate SFT training configuration and upload to S3.

    This asset:
    - Creates a JSON configuration with training hyperparameters
    - References pre-processed datasets uploaded by sft_datasets
    - Uploads config to S3 for RunPod execution

    Returns:
        S3 key of the uploaded config
    """
    context.log.info("Generating SFT training configuration...")

    model_tag = config.model_tag if config.model_tag else "d4"

    # Construct dataset S3 keys (predictable paths)
    train_s3_key = f"datasets/sft/{model_tag}/train.jsonl"
    val_s3_key = f"datasets/sft/{model_tag}/val.jsonl"
    identity_s3_key = "datasets/identity_conversations.jsonl"

    # Build training configuration
    training_config = {
        "model_tag": model_tag,
        # Dataset paths (pre-processed on S3)
        "s3_bucket": S3_BUCKET_NAME,
        "s3_train_dataset_key": train_s3_key,
        "s3_val_dataset_key": val_s3_key,
        "s3_identity_conversations_key": identity_s3_key,
        # Checkpoint paths
        "s3_mid_checkpoint_key": f"checkpoints/{model_tag}-mid/checkpoint.tar.gz",
        "s3_tokenizer_key": "tokenizer.json",
        "s3_output_key": f"checkpoints/{model_tag}-sft/checkpoint.tar.gz",
        # Training hyperparameters
        "num_epochs": config.num_epochs,
        "device_batch_size": config.device_batch_size,
        "target_examples_per_step": config.target_examples_per_step,
        "max_seq_len": config.max_seq_len,
        "eval_every": config.eval_every,
        "eval_steps": config.eval_steps,
        # Specialized learning rates for SFT
        "unembedding_lr": config.unembedding_lr,
        "embedding_lr": config.embedding_lr,
        "matrix_lr": config.matrix_lr,
        "weight_decay": config.weight_decay,
        "init_lr_frac": config.init_lr_frac,
    }

    # Upload config to S3
    config_s3_key = f"configs/sft/{model_tag}/config_{context.run_id[:8]}.json"
    s3_client = s3.get_client()

    context.log.info(f"Uploading config to S3: {config_s3_key}")
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=config_s3_key,
        Body=json.dumps(training_config, indent=2),
        ContentType="application/json",
    )

    context.log.info(f"Config uploaded to s3://{S3_BUCKET_NAME}/{config_s3_key}")

    return config_s3_key


@dg.asset(
    deps=[
        midtraining_checkpoint,
        sft_run_config,
        sft_datasets,
        nanochat_training_image,
    ],
    kinds={"runpod", "Python"},
    group_name="supervised_fine_tuning",
)
def sft_checkpoint(
    context: dg.AssetExecutionContext,
    config: SFTConfig,
    sft_run_config: str,
    nanochat_training_image: str,
    runpod: RunPodResource,
    s3: S3Resource,
) -> dg.MaterializeResult:
    """
    Supervised fine-tuning on RunPod for better conversational abilities.

    This asset:
    - Orchestrates SFT training on a RunPod GPU instance
    - Loads pre-processed datasets from S3
    - Trains on conversational datasets (ARC, GSM8K, SmolTalk, identity, spelling)
    - Uses specialized learning rates (high for embeddings, low for matrices)
    - Evaluates on validation loss
    - Downloads final checkpoint to data/sft_checkpoints/

    Returns:
        MaterializeResult with checkpoint metadata
    """
    context.log.info("Starting Supervised Fine-Tuning (SFT) on RunPod...")
    start_time = time.time()

    # Get S3 config key from sft_run_config
    s3_config_key = sft_run_config
    s3_client = s3.get_client()

    # Download training config from S3 to get output key
    config_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_config_key)
    training_config = json.loads(config_obj["Body"].read())

    # Build training command (single GPU for simplicity, like midtraining)
    train_cmd = (
        f"bash -c 'python /workspace/scripts/train_sft_remote.py "
        f"--s3-bucket {S3_BUCKET_NAME} "
        f"--s3-config-key {s3_config_key} && exit 0 || exit 1'"
    )

    # Run training on pod (single GPU)
    model_tag = config.model_tag if config.model_tag else "d4"
    pod_name = f"nanochat-sft-{model_tag}-{context.run_id[:8]}"
    context.log.info(f"Config: s3://{S3_BUCKET_NAME}/{s3_config_key}")
    context.log.info(f"Image: {nanochat_training_image}")
    context.log.info(f"Training on single GPU (SFT uses conversation-level batching)")

    pod = runpod.run_pod(
        pod_name,
        nanochat_training_image,
        train_cmd,
        context,
        volume_in_gb=300,
        gpu_count=1,  # Single GPU for SFT
    )

    try:
        # Wait for training to complete and download checkpoint
        s3_output_key = training_config["s3_output_key"]
        sft_checkpoint_dir = os.path.join(SFT_CHECKPOINT_DIRECTORY, model_tag)

        checkpoint_metadata = wait_and_download_checkpoint(
            s3_client=s3_client,
            bucket=S3_BUCKET_NAME,
            s3_key=s3_output_key,
            output_dir=sft_checkpoint_dir,
            context=context,
            max_wait_seconds=3600,  # 1 hour
            check_interval=30,
        )

        # Prepare checkpoint for serverless Docker build
        # Copy to ./checkpoint/ at repo root for Dockerfile.serverless
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        serverless_checkpoint_dir = os.path.join(project_root, "checkpoint")

        context.log.info(f"Preparing checkpoint for serverless build...")

        # Remove old checkpoint directory if it exists
        if os.path.exists(serverless_checkpoint_dir):
            shutil.rmtree(serverless_checkpoint_dir)

        # Copy checkpoint files to repo root
        shutil.copytree(sft_checkpoint_dir, serverless_checkpoint_dir)
        context.log.info(f"Checkpoint copied to: {serverless_checkpoint_dir}")
        context.log.info(
            f"Ready for serverless build: docker buildx build -f Dockerfile.serverless ..."
        )

        training_time = time.time() - start_time
        results = checkpoint_metadata.get("results", {})

        return dg.MaterializeResult(
            metadata={
                "checkpoint_dir": dg.MetadataValue.path(sft_checkpoint_dir),
                "model_tag": model_tag,
                "num_steps": results.get("step", 0),
                "training_time_seconds": training_time,
                "final_train_loss": results.get("final_train_loss"),
                "min_val_loss": results.get("min_val_loss"),
                "model_tag": model_tag,
                "s3_checkpoint_key": s3_output_key,
            }
        )

    finally:
        # Always terminate the pod
        context.log.info(f"Terminating RunPod: {pod['id']}")
        runpod.terminate_pod(pod["id"], context)


# =============================================================================
# Serverless Deployment
# =============================================================================


@dg.asset(
    kinds={"docker"},
    group_name="image",
    deps=[sft_checkpoint],
)
def nanochat_serverless_image(context: dg.AssetExecutionContext) -> str:
    """Docker image containing nanochat serverless code and dependencies."""
    return "dhume/dagster-nanochat-serverless:latest"


@dg.asset(
    kinds={"runpod"},
    group_name="serverless",
    deps=[nanochat_serverless_image],
)
def serverless_endpoint(
    context: dg.AssetExecutionContext,
    config: SFTConfig,
    serverless: ServerlessResource,
    nanochat_serverless_image: str,
) -> str:
    """
    Create or update a RunPod serverless endpoint (fully automated).

    This asset:
    1. Searches for existing template/endpoint by name
    2. Creates new template if not found, or updates existing
    3. Creates new endpoint if not found, or updates existing

    The endpoint name is deterministic (based on model_tag), so:
    - First run: Creates new template and endpoint
    - Subsequent runs: Finds existing resources and updates them

    No self-referencing needed - we search by name each time.

    Returns the endpoint ID as a string for use in inference.
    """
    image_uri = nanochat_serverless_image
    model_tag = config.model_tag if config.model_tag else "d4"
    template_name = f"nanochat-{model_tag}-template"
    endpoint_name = f"nanochat-{model_tag}"

    # Step 1: Create or update template
    # Search for existing template by name
    existing_template = serverless.find_template_by_name(
        name=template_name,
        context=context,
    )

    if existing_template:
        # Template exists - update with new image
        template_id = existing_template.get("id")
        serverless.update_template(
            template_id=template_id,
            image_name=image_uri,
            context=context,
        )
    else:
        # Template doesn't exist - create new one
        template_id = serverless.create_template(
            name=template_name,
            image_name=image_uri,
            container_disk_gb=10,
            is_serverless=True,
            context=context,
        )

    # Step 2: Create or update endpoint
    # Search for existing endpoint by name
    existing_endpoint = serverless.find_endpoint_by_name(
        name=endpoint_name,
        context=context,
    )

    if existing_endpoint:
        # Endpoint exists - update with new template
        endpoint_id = existing_endpoint.get("id")
        serverless.update_endpoint(
            endpoint_id=endpoint_id,
            template_id=template_id,
            context=context,
        )
    else:
        # Endpoint doesn't exist - create new one
        endpoint_id = serverless.create_endpoint(
            name=endpoint_name,
            template_id=template_id,
            gpu_count=1,
            workers_min=0,  # Scale to zero when idle
            workers_max=2,  # Scale up to 2 workers under load (RunPod quota limit)
            idle_timeout=30,
            context=context,
        )

    context.log.info(f"View in console: https://www.runpod.io/console/serverless")

    # Return with metadata storing both IDs
    return dg.MaterializeResult(
        value=endpoint_id,
        metadata={
            "endpoint_id": dg.MetadataValue.text(endpoint_id),
            "template_id": dg.MetadataValue.text(template_id),
            "image": dg.MetadataValue.text(image_uri),
        },
    )


# =============================================================================
# Model Inference
# =============================================================================


@dg.asset(
    deps=[serverless_endpoint],
    group_name="inference",
)
def chat_inference(
    context: dg.AssetExecutionContext,
    config: ChatInferenceConfig,
    serverless: ServerlessResource,
    serverless_endpoint: str,
) -> dg.MaterializeResult:
    """
    Generate a response to a user question using RunPod Serverless.

    This asset calls your deployed serverless endpoint to generate responses,
    providing fast, scalable inference without loading the model locally.

    The endpoint auto-scales based on demand and you only pay for inference time.

    Example config:
    {
      "question": "What is machine learning?",
      "max_tokens": 256,
      "temperature": 1.0
    }
    """
    # Get endpoint ID from upstream asset
    endpoint_id = serverless_endpoint

    context.log.info(f"Using serverless endpoint: {endpoint_id}")
    context.log.info(f"Question: {config.question}")

    # Prepare conversation messages (single-turn for now)
    messages = [{"role": "user", "content": config.question}]

    # Run inference on serverless endpoint
    start_time = time.time()

    response = serverless.run_inference(
        endpoint_id=endpoint_id,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        context=context,
    )

    inference_time = time.time() - start_time

    # Extract response from our custom handler format
    answer = response.get("response", "")
    tokens_generated = response.get("tokens_generated", 0)
    finish_reason = response.get("finish_reason", "unknown")
    model_info = response.get("model_info", {})

    context.log.info(f"Response: {answer}")
    context.log.info(f"Tokens: {tokens_generated}, Time: {inference_time:.2f}s")

    # Return result with metadata
    return dg.MaterializeResult(
        metadata={
            "question": dg.MetadataValue.text(config.question),
            "response": dg.MetadataValue.text(answer),
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "tokens_generated": tokens_generated,
            "finish_reason": finish_reason,
            "inference_time_seconds": inference_time,
            "endpoint_id": endpoint_id,
            "model_layers": model_info.get("n_layers"),
            "vocab_size": model_info.get("vocab_size"),
        }
    )
