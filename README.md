# Dagster-Nanochat

A production-grade Dagster pipeline for training language models from scratch—from raw text to conversational AI.

## What is This?

This project demonstrates how to build a complete LLM training pipeline using Dagster as the orchestration layer. Starting with the FineWeb-Edu dataset, it trains a GPT-style model through six distinct stages:

1. **Data Ingestion** - Download 1823 parquet shards in parallel
2. **Tokenizer Training** - Train custom BPE tokenizers using Rust
3. **Tokenizer Validation** - Automated quality checks
4. **Base Pretraining** - Train transformer on raw text (~11B tokens)
5. **Midtraining** - Teach conversation format and tool use
6. **Supervised Fine-Tuning** - Domain adaptation for chat abilities
7. **Inference** - Interactive chat with the trained model

**All steps are fully implemented and working!** This is a complete, end-to-end pipeline with 1822 parallel data processing jobs, automated quality checks, and production-ready training code.

---

## Quick Start

```bash
# 1. Install dependencies (including Rust tokenizer)
uv sync

# 2. Start Dagster UI
dagster dev

# 3. Open http://localhost:3000

# 4. Run the complete pipeline (quick mode ~15-20 min)
# In UI: Jobs → model_training → Launch Run
```

**Quick mode** trains a tiny 4-layer model for testing the entire pipeline end-to-end.

---

## Installation

### Prerequisites

- Python 3.9-3.13
- Rust and Cargo (for building the `rustbpe` tokenizer)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

## Pipeline Overview

### Key Features

- **Parallel Processing**: 1822 concurrent data downloads and tokenizer training jobs
- **Quality Gates**: Automated asset checks at every stage
- **Flexible Configuration**: Quick mode for testing, full mode for production
- **Complete Observability**: Track lineage, metadata, and progress in Dagster UI
- **Self-Contained**: All modules bundled—no external setup required

---

## Documentation

📚 **[GUIDE.md](GUIDE.md)** - **Complete pipeline guide** with detailed documentation for:
- Installation and setup
- Pipeline architecture
- All 6 training steps
- Configuration options
- Troubleshooting
- Advanced topics

☁️ **[RUNPOD.md](RUNPOD.md)** - **Run training on cloud GPUs** using RunPod:
- Cost-effective GPU access (~$0.30-$1.50/hr)
- Manual and automated workflows
- Full integration with Dagster

📄 **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Project description and blog post series outline

## Configuration

### Quick Mode vs. Full Mode

**Quick Mode** (`quick_mode=True`, default):
- Model: 4 layers, ~1M parameters
- Time: 5-10 minutes on GPU (local or cloud)
- Purpose: Fast pipeline testing and development

**Full Mode** (`quick_mode=False`):
- Model: 12 layers, ~10M parameters (depth automatically set to 12)
- Time: 1-2 hours on GPU, 4-8 hours on CPU
- Purpose: Production training

Change in UI when materializing assets, or via config:

```python
{"quick_mode": false}
```

## Learn More

- [Dagster University](https://courses.dagster.io/) - Free Dagster courses
- [Dagster Slack Community](https://dagster.io/slack) - Get help and discuss

---

**Ready to train your own language model?** Check out **[GUIDE.md](GUIDE.md)** for the complete documentation! 🚀
