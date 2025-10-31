"""Data generators for training."""

from collections import deque

import torch

from dagster_nanochat.nanochat.tokenizer import RustBPETokenizer
from dagster_nanochat.tasks.common import TaskMixture


def create_midtraining_data_generator(
    train_dataset: TaskMixture,
    val_dataset: TaskMixture,
    tokenizer: RustBPETokenizer,
    device_batch_size: int,
    max_seq_len: int,
    num_iterations: int,
    device_type: str,
    device: torch.device,
):
    """
    Create data generator for midtraining with conversation-level tokenization.

    This function handles the special case of midtraining where data consists
    of full conversations. Unlike base training which uses fixed-length chunks,
    this concatenates rendered conversations and yields continuous token streams.

    Args:
        train_dataset: Training TaskMixture with conversation data
        val_dataset: Validation TaskMixture
        tokenizer: RustBPETokenizer with conversation special tokens
        device_batch_size: Batch size for training
        max_seq_len: Maximum sequence length
        num_iterations: Number of training iterations (-1 for one epoch)
        device_type: "cuda", "mps", or "cpu"
        device: PyTorch device for tensors

    Returns:
        Tuple of (data_generator, last_step_fn, progress_fn) where:
        - data_generator yields (inputs, targets) tensor pairs
        - last_step_fn returns boolean indicating final step
        - progress_fn returns approximate training progress
    """
    last_step = False
    approx_progress = 0.0

    def mid_data_generator(split):
        nonlocal last_step, approx_progress
        dataset = train_dataset if split == "train" else val_dataset
        dataset_size = len(dataset)
        needed_tokens = device_batch_size * max_seq_len + 1
        token_buffer = deque()
        scratch = torch.empty(
            needed_tokens, dtype=torch.int64, pin_memory=(device_type == "cuda")
        )
        cursor = 0
        it = 0

        while True:
            # Accumulate tokens
            while len(token_buffer) < needed_tokens:
                conversation = dataset[cursor]
                ids, _ = tokenizer.render_conversation(conversation)
                token_buffer.extend(ids)
                cursor += 1
                if cursor >= dataset_size:
                    cursor = 0
                    if split == "train":
                        last_step = True

            it += 1
            if num_iterations > 0 and it >= num_iterations:
                last_step = True

            # Build batch
            for i in range(needed_tokens):
                scratch[i] = token_buffer.popleft()

            inputs_cpu = scratch[:-1].to(dtype=torch.int32)
            targets_cpu = scratch[1:]
            inputs = inputs_cpu.view(device_batch_size, max_seq_len).to(
                device=device, dtype=torch.int32, non_blocking=True
            )
            targets = targets_cpu.view(device_batch_size, max_seq_len).to(
                device=device, dtype=torch.int64, non_blocking=True
            )

            if split == "train":
                if num_iterations > 0:
                    approx_progress = it / num_iterations
                else:
                    approx_progress = cursor / dataset_size

            yield inputs, targets

    return mid_data_generator, lambda: last_step, lambda: approx_progress


def create_sft_data_generator(
    train_dataset: TaskMixture,
    val_dataset: TaskMixture,
    tokenizer: RustBPETokenizer,
    device_batch_size: int,
    device: torch.device,
    num_epochs: int,
    world_size: int = 1,
    rank: int = 0,
):
    """
    Create data generator for SFT with conversation-level batching and masking.

    This function handles supervised fine-tuning where only assistant responses
    are trained. User messages are masked from the loss to prevent the model
    from learning to generate user-like text.

    Key differences from midtraining:
    - Uses conversation-level batching (full conversations per batch)
    - Pads shorter conversations to max length in batch
    - Masks user turns and padding from loss computation
    - Iterates through data by epoch rather than token count

    Args:
        train_dataset: Training TaskMixture with conversational data
        val_dataset: Validation TaskMixture
        tokenizer: RustBPETokenizer with conversation special tokens
        device_batch_size: Batch size per device
        device: PyTorch device to place tensors on
        num_epochs: Number of training epochs
        world_size: Total number of processes (for DDP, unused in single-GPU)
        rank: Process rank (for DDP, unused in single-GPU)

    Yields:
        Tuple of (inputs, targets) tensors where:
        - inputs: [batch_size, seq_len] input token IDs
        - targets: [batch_size, seq_len] target token IDs (with -100 for masked positions)
    """
    pad_token_id = tokenizer.encode_special(
        "<|assistant_end|>"
    )  # use as pad token (positions masked in loss)

    def collate_and_yield(batch):
        """Collate a batch of tokenized conversations into padded tensors."""
        nrows = len(batch)
        ncols = (
            max(len(ids) for ids, mask in batch) - 1
        )  # seq of n creates inputs/targets of n-1

        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)  # -1 is ignore index

        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, : n - 1] = ids_tensor[:-1]

            # Mask out targets where mask is 0 (i.e., user messages)
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1  # mask user messages
            targets[i, : n - 1] = row_targets

        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets

    def sft_data_generator(split):
        """Generate batches from the specified split."""
        dataset = train_dataset if split == "train" else val_dataset
        batch = []

        for _ in range(num_epochs):
            for i in range(rank, len(dataset), world_size):
                doc = dataset[i]
                ids, mask = tokenizer.render_conversation(doc)
                batch.append((ids, mask))

                if len(batch) == device_batch_size:
                    yield collate_and_yield(batch)
                    batch = []

            # Yield remaining batch at end of epoch
            if batch:
                yield collate_and_yield(batch)
                batch = []

    return sft_data_generator
