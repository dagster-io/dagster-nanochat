# Dagster-Nanochat

A production-grade Dagster pipeline for training language models from scratch—from raw text to conversational AI.

## What is This?

This project demonstrates how to build a complete LLM training pipeline using Dagster as the orchestration layer. Starting with the FineWeb-Edu dataset, it trains a GPT-style model through six distinct stages:

1. **Data Ingestion** - Download 1823 parquet shards in parallel
2. **Tokenizer Training** - Train custom BPE tokenizers using Rust
3. **Tokenizer Validation** - Automated quality checks
4. **Base Pretraining** - Train transformer on raw text
5. **Midtraining** - Teach conversation format and tool use
6. **Supervised Fine-Tuning** - Domain adaptation for chat abilities
7. **Inference** - Interactive chat with the trained model