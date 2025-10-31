# Nanochat training Docker image with GPU support
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV TORCHDYNAMO_DISABLE=1

# Install build dependencies for rustbpe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /bin/uv

# Install Rust for rustbpe tokenizer (minimal install)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace

# Copy dependency files first (changes less frequently)
COPY pyproject.toml uv.lock /workspace/
COPY rustbpe /workspace/rustbpe/

# Install Python dependencies with uv (includes building rustbpe)
RUN uv pip install --system -e . && \
    rm -rf /root/.cargo/registry /root/.cargo/git && \
    rm -rf rustbpe/target/debug rustbpe/target/release/build rustbpe/target/release/deps rustbpe/target/release/incremental

# Copy source code last (changes most frequently)
COPY src /workspace/src/
COPY scripts /workspace/scripts/

# Set Python path
ENV PYTHONPATH=/workspace/src:/workspace:${PYTHONPATH}

# Create data directories
RUN mkdir -p data/raw data/validation checkpoint