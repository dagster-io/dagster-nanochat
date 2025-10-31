# Dagster-Nanochat: Complete Training Pipeline Guide

A production-grade implementation of a complete language model training pipeline using Dagster, from raw text to conversational AI.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Pipeline Architecture](#pipeline-architecture)
- [Training Stages](#training-stages)
  - [Data Ingestion](#stage-1-data-ingestion)
  - [Tokenizer Training](#stage-2-tokenizer-training)
  - [Base Model Pretraining](#stage-3-base-model-pretraining)
  - [Midtraining](#stage-4-midtraining)
  - [Supervised Fine-Tuning](#stage-5-supervised-fine-tuning)
- [Inference](#inference)
  - [Local Inference](#local-inference)
  - [Serverless Deployment](#serverless-deployment)
- [RunPod Setup](#runpod-setup)
- [Troubleshooting](#troubleshooting)

---

## Overview

**Dagster-Nanochat** demonstrates how modern data orchestration tools can transform LLM training into a structured, observable, and reproducible workflow. The pipeline trains a GPT-style transformer through six distinct stages:

1. **Data Ingestion** - Download 1823 parquet shards in parallel
2. **Tokenizer Training** - Train custom BPE tokenizers using Rust
3. **Base Pretraining** - Train transformer on raw text (RunPod GPU)
4. **Midtraining** - Teach conversation format (RunPod GPU)
5. **Supervised Fine-Tuning** - Domain adaptation for chat (RunPod GPU)
6. **Inference** - Interactive chat via serverless endpoint

### Quick Mode vs. Full Mode

**Quick Mode** (`quick_mode=True`, default):
- Model: 4 layers, ~1M parameters
- Datasets: Small subsets (1000s of examples)
- Iterations: Limited (50-100)
- Time: 5-10 minutes total on GPU
- Purpose: Pipeline testing and development

**Full Mode** (`quick_mode=False`):
- Model: 12 layers, ~10M parameters
- Datasets: Complete (100Ks-1M examples)
- Iterations: Auto-calculated from data
- Time: 2-4 hours on RunPod GPUs
- Purpose: Real model training

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment variables
export RUNPOD_API_KEY="your-runpod-api-key"
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"

# 3. Add RunPod secrets (in RunPod dashboard)
# - RUNPOD_SECRET_AWS_ACCESS_KEY_ID
# - RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY

# 4. Start Dagster UI
dagster dev

# 5. Materialize assets in order (via UI or CLI)
dagster asset materialize --select training_files
dagster asset materialize --select tokenizer_training
dagster asset materialize --select base_model_checkpoint  # Runs on RunPod
dagster asset materialize --select midtraining_checkpoint # Runs on RunPod
dagster asset materialize --select sft_checkpoint         # Runs on RunPod

# 6. Deploy serverless endpoint (fully automated)
# a. Materialize SFT checkpoint
dagster asset materialize --select sft_checkpoint

# b. Build and push Docker image
./build_and_push_serverless.sh

# c. Deploy endpoint (creates template + endpoint automatically)
dagster asset materialize --select serverless_endpoint

# 7. Run inference
dagster asset materialize --select chat_inference
```

---

## Installation

### Prerequisites

- **Python 3.12+**
- **Rust** (for rustbpe tokenizer)
- **Docker** (for RunPod image and serverless deployment)
- **RunPod account** with API key
- **AWS S3** bucket for checkpoints and configs

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/dagster-nanochat
cd dagster-nanochat

# Install with uv
uv sync

# Build rustbpe tokenizer
cd rustbpe
uv pip install -e .
cd ..

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start Dagster
dagster dev
```

---

## Pipeline Architecture

### Asset Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Data Ingestion                             │
└────────────────┬────────────────────────────────────────────────────┘
                 ↓
         training_files (1822 shards)
                 ↓
┌────────────────────────────────────────────────────────────────────┐
│                       Tokenizer Training                            │
└────────────────┬───────────────────────────────────────────────────┘
                 ↓
         tokenizer_training
                 ↓
┌────────────────────────────────────────────────────────────────────┐
│                      Base Model Training (RunPod)                   │
└────────────────┬───────────────────────────────────────────────────┘
                 ↓
         model_run_config (S3) ──→ base_model_checkpoint (RunPod GPU)
                 ↓
┌────────────────────────────────────────────────────────────────────┐
│                       Midtraining (RunPod)                          │
└────────────────┬───────────────────────────────────────────────────┘
                 ↓
    midtraining_run_config (S3) ──→ midtraining_checkpoint (RunPod GPU)
                 ↓
┌────────────────────────────────────────────────────────────────────┐
│                    Supervised Fine-Tuning (RunPod)                  │
└────────────────┬───────────────────────────────────────────────────┘
                 ↓
    sft_datasets + sft_run_config (S3) ──→ sft_checkpoint (RunPod GPU)
                 ↓
┌────────────────────────────────────────────────────────────────────┐
│                        Inference (Serverless)                       │
└────────────────┬───────────────────────────────────────────────────┘
                 ↓
    serverless_endpoint (RunPod Serverless) ──→ chat_inference
```

### RunPod Training Pattern

All GPU training stages (base, midtraining, SFT) follow this pattern:

1. **Local (Dagster)**:
   - Upload training config to S3
   - Create RunPod GPU instance with pre-built Docker image
   
2. **Remote (RunPod)**:
   - Download config and tokenizer from S3
   - Download training data from HuggingFace
   - Execute training (with DDP if multi-GPU)
   - Upload checkpoint to S3
   
3. **Local (Dagster)**:
   - Download checkpoint from S3
   - Terminate RunPod instance
   - Extract and store checkpoint locally

---

## Training Stages

### Stage 1: Data Ingestion

Downloads 1823 parquet shards from HuggingFace's FineWeb-Edu dataset.

**Assets:**
- `training_files` (1822 partitions): Training data shards
- `validation_files` (1 file): Validation shard

**Output:**
```
data/raw/shard_00000.parquet ... shard_01821.parquet
data/validation/shard_01822.parquet
```

**Run:**
```bash
dagster asset materialize --select training_files
```

---

### Stage 2: Tokenizer Training

Trains a custom BPE tokenizer using a high-performance Rust implementation.

**Asset:** `tokenizer_training`

**Configuration:**
- Vocabulary size: 10,000 tokens
- Pattern: GPT-4 regex for token splitting
- Trains on ALL available training data (no quick mode)

**Output:**
```
data/tokenizer/tokenizer.json
```

**Asset Checks:**
- `tokenizer_file_valid`: Validates JSON structure
- `tokenizer_encode_decode`: Tests round-trip encoding
- `tokenizer_compression_ratio`: Measures compression (≥1.5 bytes/token)
- `tokenizer_vs_gpt2_*`: Benchmarks against GPT-2 on various text types

**Run:**
```bash
dagster asset materialize --select tokenizer_training
```

---

### Stage 3: Base Model Pretraining

Trains a GPT transformer from scratch on raw text using RunPod GPUs.

**Assets:**
- `nanochat_training_image`: Docker image for training
- `model_run_config`: Training configuration (uploaded to S3)
- `base_model_checkpoint`: Training execution on RunPod

**Configuration (Quick Mode):**
- Model: 4 layers, 256 dim, 4 heads, ~1M params
- Batch size: 1 per device
- Sequence length: 512 tokens
- Iterations: 100
- Time: ~2-3 minutes on GPU

**Configuration (Full Mode):**
- Model: 12 layers, 1280 dim, 10 heads, ~10M params
- Batch size: 32 per device
- Sequence length: 2048 tokens
- Iterations: Auto-calculated
- Time: 1-2 hours on RunPod (2× NVIDIA A40)

**Training Process:**
1. Dagster uploads config to S3
2. Creates RunPod pod with 2 GPUs (NVIDIA A40)
3. Pod downloads config and tokenizer from S3
4. Pod downloads training data from HuggingFace (1822 shards)
5. Executes distributed training with DDP
6. Uploads checkpoint to S3
7. Dagster downloads checkpoint locally
8. Pod terminates automatically

**Optimizers:**
- **AdamW**: Embeddings and output layer
- **Muon**: Transformer weight matrices

**Output:**
```
data/checkpoints/d4/
├── model.pt          # PyTorch model weights
├── optimizer.pt      # Optimizer state
└── checkpoint.json   # Training metadata
```

**Run:**
```bash
# Quick mode (default)
dagster asset materialize --select base_model_checkpoint

# Full mode
dagster asset materialize --select base_model_checkpoint \
  -c '{"quick_mode": false}'
```

---

### Stage 4: Midtraining

Teaches the base model conversation format through continued pretraining on conversational datasets.

**Assets:**
- `midtraining_run_config`: Training configuration (uploaded to S3)
- `midtraining_checkpoint`: Training execution on RunPod (single GPU)

**Datasets (Full Mode):**
- SmolTalk: 460K conversations
- MMLU: 100K multiple choice problems
- GSM8K: 8K math problems with calculator tool
- SimpleSpelling: 200K spelling tasks
- SpellingBee: 80K letter-counting tasks

**Key Features:**
1. **Vocabulary Extension**: Adds 9 conversation special tokens:
   - `<|user_start|>`, `<|user_end|>`
   - `<|assistant_start|>`, `<|assistant_end|>`
   - `<|python_start|>`, `<|python_end|>`
   - `<|output_start|>`, `<|output_end|>`
   - `<|bos|>` (beginning of sequence)

2. **Conversation Rendering**: Formats multi-turn conversations with special tokens

3. **Single GPU**: Runs on 1 GPU for simplicity (no DDP)

**Training Process:**
1. Dagster uploads config to S3
2. Creates RunPod pod with 1 GPU
3. Pod downloads config, base checkpoint, and tokenizer from S3
4. Pod downloads datasets from HuggingFace
5. Extends model vocabulary for conversation tokens
6. Trains with conversation-formatted data
7. Uploads checkpoint to S3
8. Dagster downloads checkpoint locally

**Output:**
```
data/mid_checkpoints/d4/
├── model.pt          # Midtrained weights
├── tokenizer.json    # Tokenizer (included for convenience)
└── checkpoint.json   # Training metadata
```

**Run:**
```bash
dagster asset materialize --select midtraining_checkpoint
```

---

### Stage 5: Supervised Fine-Tuning

Domain adaptation for better conversational abilities using specialized learning rates.

**Assets:**
- `sft_datasets`: Prepare and upload datasets to S3
- `sft_run_config`: Training configuration (uploaded to S3)
- `sft_checkpoint`: Training execution on RunPod (single GPU)

**Datasets (Full Mode):**
- ARC-Easy: 2.3K reasoning problems
- ARC-Challenge: 1.1K challenging problems
- GSM8K: 8K math word problems
- SmolTalk: 10K conversations
- Identity Conversations: 25 synthetic examples
- SimpleSpelling: 300 tasks
- SpellingBee: 300 tasks

**Key Innovations:**

1. **Specialized Learning Rates**:
   - Embedding layer: 0.2 (high - adapt token representations)
   - Unembedding layer: 0.004 (low - preserve stability)
   - Transformer matrices: 0.02 (medium - balanced adaptation)

2. **Conversation-Level Batching**:
   - Complete conversations per batch (not token chunks)
   - Padding to equal length
   - **Mask user messages** - only supervise assistant responses

3. **Identity Conversations**:
   Custom dataset (`data/identity_conversations.jsonl`) teaches:
   - Model's name (NanoChat)
   - Purpose (helpful AI assistant)
   - Limitations (no internet, knowledge cutoff, etc.)

**Training Process:**
1. Dagster prepares datasets and uploads to S3 as JSONL
2. Uploads config to S3
3. Creates RunPod pod with 1 GPU
4. Pod downloads config, midtraining checkpoint, and datasets from S3
5. Trains with conversation-level batching
6. Uploads checkpoint to S3
7. Dagster downloads checkpoint locally
8. **Prepares checkpoint for serverless deployment** (copies to `./checkpoint/`)

**Asset Checks:**
- `sft_mmlu_accuracy`: Evaluates on MMLU (≥28% pass threshold)
- `sft_arc_easy_accuracy`: Evaluates on ARC-Easy (≥30% pass threshold)

**Output:**
```
data/sft_checkpoints/d4/
├── model.pt          # SFT-tuned weights
├── tokenizer.json    # Tokenizer config
└── checkpoint.json   # Training metadata

checkpoint/           # Copy for Docker build
├── model.pt
├── tokenizer.json
└── checkpoint.json
```

**Run:**
```bash
dagster asset materialize --select sft_checkpoint
```

---

## Inference

### Local Inference

For quick testing without deploying a serverless endpoint.

**Asset:** `chat_inference`

**Note:** The `chat_inference` asset is configured to use the **serverless endpoint**. For local inference, you would need to modify the asset or create a separate asset.

---

### Serverless Deployment

Deploy your trained model as a RunPod Serverless endpoint for production inference.

#### Overview

Four-step **fully automated** deployment process:

1. **Prepare Checkpoint** - `sft_checkpoint` asset downloads and packages model
2. **Build Docker Image** - Script builds image with model baked in
3. **Deploy Endpoint** - `serverless_endpoint` asset creates/updates template + endpoint (searches by name)
4. **Run Inference** - `chat_inference` asset calls the endpoint

No manual RunPod UI steps or environment variables required!

#### Step 1: Prepare Checkpoint

The `sft_checkpoint` asset automatically prepares the checkpoint:

```bash
dagster asset materialize --select sft_checkpoint
```

This:
- Downloads checkpoint from S3
- Extracts to `data/sft_checkpoints/d4/`
- Copies to `./checkpoint/` for Docker build

#### Step 2: Build & Push Docker Image

Build a Docker image with your trained model:

```bash
# Verify checkpoint exists
ls -lh checkpoint/

# Use the provided script (or run docker buildx manually)
./build_and_push_serverless.sh
```

The script builds and pushes:

**What's in the image:**
- PyTorch 2.4.0 with CUDA
- All Python dependencies
- Rust toolchain and rustbpe
- Your source code
- **Trained model weights** (baked in for fast cold starts)
- Inference handler

#### Step 3: Update Image Name

In `src/dagster_nanochat/defs/assets.py`, update the `nanochat_serverless_image` asset:

```python
@dg.asset(kinds={"docker"})
def nanochat_serverless_image(context: dg.AssetExecutionContext) -> str:
    image = "YOUR_USERNAME/nanochat-serverless:latest"  # ← Change this
    return image
```

#### Step 4: Deploy Endpoint (Fully Automated)

```bash
dagster asset materialize --select serverless_endpoint
```

This single command:
1. **Searches for existing template/endpoint** by name (based on model_tag)
2. **Creates/updates a RunPod template** with your Docker image
3. **Creates/updates a serverless endpoint** using that template

**First run**: Creates new template and endpoint
**Subsequent runs**: Finds and updates existing template/endpoint with new image

No manual steps or environment variables needed!

#### Step 5: Run Inference

```bash
# Via Dagster UI with config:
{
  "chat_inference": {
    "config": {
      "question": "What is machine learning?",
      "max_tokens": 100,
      "temperature": 1.0,
      "model_tag": "d4"
    }
  }
}

# Or via CLI
dagster asset materialize --select chat_inference
```

#### RunPod Quotas

**Worker Limits**: RunPod accounts have a maximum worker quota across all endpoints (typically 5 for free tier). If you hit this limit, you have two options:

1. **Reduce workers on other endpoints** in RunPod console
2. **Delete unused endpoints** to free up quota

The asset is configured to use 2 max workers by default to stay within typical limits.

#### Costs

**Image Build**: Free (runs locally)

**Serverless Endpoint**:
- **Idle**: $0 (scales to zero)
- **Inference**: ~$0.00009/second (RTX A4000)
- **Example**: 1000 inferences × 1.5s = **$0.13**

**Compare to Training Pod**:
- Training: ~$0.70/hour (2× A40) = ~$500/month 24/7
- **Serverless is >>99% cheaper for inference**

#### Updating the Model

After retraining:

```bash
# 1. Retrain
dagster asset materialize --select sft_checkpoint

# 2. Rebuild image (checkpoint is auto-updated)
docker buildx build --platform linux/amd64 \
  -f Dockerfile.serverless \
  -t YOUR_USERNAME/nanochat-serverless:latest \
  --push .

# 3. Update endpoint in RunPod console
# Go to https://www.runpod.io/console/serverless
# Select your endpoint → Settings → Update Image
# Enter new image URI and save

# 4. Test
export RUNPOD_SERVERLESS_ENDPOINT_ID="your-endpoint-id"
dagster asset materialize --select serverless_endpoint
dagster asset materialize --select chat_inference
```

**Note**: You can reuse the same endpoint by updating its image, or create a new endpoint for A/B testing.

---

## RunPod Setup

### Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://www.runpod.io/)
2. **API Key**: Get from Settings → API Keys
3. **Docker Hub**: For custom training image (or use `dhume/dagster-nanochat:latest`)

### Environment Variables

```bash
# Required for Dagster
export RUNPOD_API_KEY="your-runpod-api-key"
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"

# Required for Serverless Inference (after creating endpoint)
export RUNPOD_SERVERLESS_ENDPOINT_ID="your-endpoint-id"
```

### RunPod Secrets

Add these in [RunPod Settings → Secrets](https://www.runpod.io/console/user/secrets):

- `RUNPOD_SECRET_AWS_ACCESS_KEY_ID` = your AWS access key
- `RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY` = your AWS secret key

These are injected into pods for S3 access.

### Docker Image for Training

The training pipeline uses a pre-built Docker image with all dependencies.

**Use the public image** (recommended):
```python
# In src/dagster_nanochat/defs/assets.py
@dg.asset(kinds={"docker"})
def nanochat_training_image(context: dg.AssetExecutionContext) -> str:
    return "dhume/dagster-nanochat:latest"
```

**Or build your own**:
```bash
# Build
docker buildx build --platform linux/amd64 \
  -t YOUR_USERNAME/dagster-nanochat:latest \
  --push .

# Update assets.py
return "YOUR_USERNAME/dagster-nanochat:latest"
```

**What's pre-installed:**
- PyTorch 2.4.0 with CUDA 12.1
- Rust toolchain
- rustbpe (pre-built)
- All Python dependencies
- Training scripts

**Benefits:**
- Fast pod startup (~30 seconds vs 20 minutes)
- No per-run dependency installation
- Consistent environment
- **Saves $10-40 per run** in setup time costs

### GPU Types

| GPU | VRAM | Quick Mode | Full Mode | Cost/hr |
|-----|------|------------|-----------|---------|
| RTX A4000 | 16GB | ✅ (5 min) | ✅ (1.5 hrs) | ~$0.30 |
| RTX A5000 | 24GB | ✅ (4 min) | ✅ (1 hr) | ~$0.50 |
| NVIDIA A40 | 48GB | ✅ (3 min) | ✅ (1 hr) | ~$0.70 |
| NVIDIA A100 | 80GB | ✅ (2 min) | ✅ (45 min) | ~$1.50 |

**Recommendation**: RTX A4000 for testing, NVIDIA A40 for production.

### Customizing GPU Settings

Edit `src/dagster_nanochat/defs/resources.py`:

```python
runpod_resource = RunPodResource(
    api_key=dg.EnvVar("RUNPOD_API_KEY"),
    gpu_count=2,  # For base training (DDP)
    gpu_type_id="NVIDIA A40",  # Change GPU type
    env_variables={
        "RUNPOD_SECRET_AWS_ACCESS_KEY_ID": dg.EnvVar("AWS_ACCESS_KEY_ID"),
        "RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY": dg.EnvVar("AWS_SECRET_ACCESS_KEY"),
    },
)
```

**Note**: Midtraining and SFT override `gpu_count=1` in their assets.

---

## Troubleshooting

### Common Issues

#### "No GPUs available"

**Solution**: Try different GPU types or cloud types:
```python
# In resources.py
gpu_type_id="RTX A4000",  # Try cheaper GPU
```

Check [RunPod Status](https://uptime.runpod.io/)

#### "Connection timeout" or "Pod not starting"

- Pods take 2-5 minutes to initialize
- Check RunPod web UI for pod status
- Try different GPU type

#### "CUDA out of memory"

Reduce batch size in config:
```json
{
  "device_batch_size": 16  // Reduce from default 32
}
```

#### "Training failed but pod still running"

1. Check RunPod web UI
2. Manually terminate orphaned pods
3. Check Dagster logs for `pod_id`

The asset's `finally` block should terminate pods automatically.

#### "Asset check failed" (MMLU/ARC)

**Causes:**
- Old intermediate checkpoint loaded (wrong vocab size)
- Model undertrained (quick mode)

**Solutions:**
- Delete old `checkpoint_*.json` files in `data/sft_checkpoints/d4/`
- Asset checks now load `checkpoint.json` (final checkpoint)
- Run full mode for better accuracy

#### "Checkpoint not found on S3"

**Causes:**
- Training failed on RunPod
- S3 upload failed
- Incorrect S3 path

**Solutions:**
- Check RunPod logs for training errors
- Verify AWS credentials in RunPod secrets
- Check S3 bucket contents

#### "Import Error: cannot import 'GPT'"

**Solution**: Already fixed. `GPT`, `GPTConfig`, and utilities are now imported from their proper modules (`nanochat.gpt`, `utils.tokenizer_utils`) not from `assets.py`.

#### "Serverless build fails: checkpoint not found"

**Solution**: Materialize `sft_checkpoint` first:
```bash
dagster asset materialize --select sft_checkpoint
```

This automatically prepares `./checkpoint/` for Docker build.

---

## Cost Optimization

### Training Costs

**Quick Mode** (testing):
- Base: ~$0.02-0.05 per run (5 min)
- Midtraining: ~$0.01-0.03 per run (3 min)
- SFT: ~$0.02-0.05 per run (5 min)
- **Total**: ~$0.05-0.13 per full pipeline test

**Full Mode** (production):
- Base: ~$0.60-1.20 per run (1-2 hrs, 2 GPUs)
- Midtraining: ~$0.20-0.40 per run (30-60 min, 1 GPU)
- SFT: ~$0.20-0.40 per run (30-60 min, 1 GPU)
- **Total**: ~$1.00-2.00 per full pipeline run

### Serverless Costs

**Inference**:
- ~$0.00009/second of inference
- 1000 requests × 1.5s = **$0.13**
- 10,000 requests = **$1.30**
- Idle: **$0** (scales to zero)

**Compare to Dedicated Pod**:
- 24/7 pod: $0.30/hr × 720 hrs = **$216/month**
- Serverless saves **99%+** for typical inference loads

### Cost Optimization Tips

1. **Start with Quick Mode**: Test pipeline with `quick_mode=True` before full runs
2. **Use Cheaper GPUs**: RTX A4000 is sufficient for most training
3. **Monitor Pods**: Ensure pods terminate after training (check `finally` blocks)
4. **Reuse Checkpoints**: Don't retrain unnecessarily
5. **Batch Inference**: Group inference requests when possible
6. **Delete Old Endpoints**: Remove unused serverless endpoints in RunPod console

---

## Hardware Requirements

### Local Development

**Minimum**:
- CPU: 4+ cores
- RAM: 8GB
- Storage: 50GB
- Purpose: Running Dagster, data ingestion, tokenizer training

**Recommended**:
- CPU: 8+ cores
- RAM: 16GB
- Storage: 200GB
- Purpose: Faster data processing and local development

**Note**: Local GPU is NOT required. All GPU training runs on RunPod.

### RunPod Training

**Quick Mode**:
- 1-2 GPUs (16GB+ VRAM)
- 32GB RAM
- 100GB storage
- Time: 10-15 minutes total

**Full Mode**:
- 2 GPUs (48GB+ VRAM) for base training
- 1 GPU (16GB+ VRAM) for midtraining/SFT
- 64GB RAM
- 300GB storage
- Time: 2-4 hours total

---

## Files and Structure

### Key Directories

```
dagster-nanochat/
├── data/                    # Data and checkpoints (gitignored)
│   ├── raw/                 # Training shards
│   ├── validation/          # Validation shard
│   ├── tokenizer/           # Trained tokenizer
│   ├── checkpoints/         # Base model checkpoints
│   ├── mid_checkpoints/     # Midtraining checkpoints
│   └── sft_checkpoints/     # SFT checkpoints
├── checkpoint/              # For serverless Docker build
├── scripts/                 # Remote training scripts
│   ├── train_base_remote.py
│   ├── train_midtraining_remote.py
│   └── train_sft_remote.py
├── src/dagster_nanochat/
│   ├── defs/               # Dagster definitions
│   │   ├── assets.py       # All assets
│   │   ├── asset_checks.py # Quality checks
│   │   ├── config.py       # Configuration
│   │   ├── resources.py    # Resources
│   │   ├── runpod_resource.py
│   │   └── serverless_resource.py
│   ├── nanochat/           # Training modules
│   ├── tasks/              # Evaluation tasks
│   └── utils/              # Utilities
├── serverless/
│   └── handler.py          # Serverless inference handler
├── Dockerfile              # Training image
├── Dockerfile.serverless   # Serverless image
└── GUIDE.md               # This file
```

### Important Files

- **`assets.py`**: All Dagster assets (data, training, inference)
- **`asset_checks.py`**: Quality checks and evaluations
- **`config.py`**: Configuration classes
- **`runpod_resource.py`**: RunPod pod management
- **`serverless_resource.py`**: RunPod Serverless management
- **Training scripts**: Standalone scripts for RunPod execution
- **Dockerfiles**: Image definitions for training and inference

---

## Next Steps

1. ✅ **Set up RunPod**: Create account, get API key, configure secrets
2. ✅ **Run Quick Mode**: Test full pipeline with small datasets
3. ✅ **Deploy Serverless**: Build and deploy inference endpoint
4. ✅ **Run Full Mode**: Train production model
5. ✅ **Monitor Costs**: Check RunPod dashboard for usage
6. ✅ **Experiment**: Try different hyperparameters and datasets

---

## Resources

- **Dagster Docs**: https://docs.dagster.io
- **RunPod Docs**: https://docs.runpod.io
- **RunPod Discord**: https://discord.gg/runpod
- **RunPod Pricing**: https://www.runpod.io/pricing
- **Project GitHub**: [Your repository URL]

---

## Questions?

For issues or questions:
1. Check this guide
2. Review Dagster asset logs
3. Check RunPod dashboard
4. Review S3 bucket contents
5. Open a GitHub issue

Happy training! 🚀
