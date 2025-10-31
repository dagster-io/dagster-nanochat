"""
Evaluate the Chat model (simplified for Dagster single-GPU usage).

This module provides evaluation functions for chat models on various tasks.
"""

from functools import partial

import torch

from dagster_nanochat.tasks.arc import ARC
from dagster_nanochat.tasks.gsm8k import GSM8K
from dagster_nanochat.tasks.humaneval import HumanEval
from dagster_nanochat.tasks.mmlu import MMLU
from dagster_nanochat.tasks.spellingbee import SpellingBee


def run_generative_eval(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
):
    """
    Generative evaluation loop (one problem at a time, sample, evaluate).

    Args:
        task_object: Task dataset object
        tokenizer: Tokenizer
        model: Model to evaluate
        engine: Engine for generation
        num_samples: Number of samples per problem
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        max_problems: Maximum problems to evaluate (None = all)

    Returns:
        Accuracy (fraction of problems passed)
    """
    device = model.get_device()
    num_problems = (
        len(task_object)
        if max_problems is None
        else min(len(task_object), max_problems)
    )

    # Run the evaluation
    num_passed, total = 0, 0
    for i in range(num_problems):
        conversation = task_object[i]

        # Tokenize the prompt
        encoded_prompt = tokenizer.render_for_completion(conversation)

        # Get the completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        # Decode the completions as text
        prefix_length = len(encoded_prompt)
        completions = [
            tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results
        ]

        # Evaluate success criteria
        outcomes = [
            task_object.evaluate(conversation, completion) for completion in completions
        ]
        passed = any(outcomes)

        # Keep stats
        total += 1
        num_passed += int(passed)

        # Logging (overwrite the same line in the console)
        if (i + 1) % 10 == 0 or i == num_problems - 1:
            print(
                f"\r\033[K{num_passed}/{total} ({100 * num_passed / total:.2f}%)",
                end="",
                flush=True,
            )

    # Finish the in-place progress line with a newline
    print()
    print("=" * 50)
    print(f"Final: {num_passed}/{total} ({100 * num_passed / total:.2f}%)")

    # Return the accuracy
    return num_passed / total


def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):
    """
    Categorical evaluation loop (batched, check logits for correct answer).

    This is more efficient than generative evaluation because we don't need to sample.
    We just check the logits at the answer position for the correct letter.

    Args:
        task_object: Task dataset object
        tokenizer: Tokenizer
        model: Model to evaluate
        batch_size: Batch size for evaluation
        max_problems: Maximum problems to evaluate (None = all)

    Returns:
        Accuracy (fraction of problems passed)
    """
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()  # use BOS as pad token

    # Process batches of independent problems
    num_problems = (
        len(task_object)
        if max_problems is None
        else min(len(task_object), max_problems)
    )
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    # Run the evaluation
    letter_to_id_cache = {}  # cache for letter token IDs
    num_passed, total = 0, 0

    for i in range(num_batches):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch of problems (pad to same length)
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [
            tokenizer.render_for_completion(conversation)
            for conversation in conversations
        ]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [
            len(ids) - 1 for ids in prompt_ids
        ]  # where the answer token should be
        padded_prompt_ids = [
            ids + [bos] * (max_length - len(ids)) for ids in prompt_ids
        ]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # Get the logits for the whole batch in parallel
        with torch.no_grad():
            logits = model(prompt_ids)  # (B, T, V)

        # Focus on the answer position and available letters
        for idx, conversation in enumerate(conversations):
            # Get token IDs for all available letters
            letters = conversation["letters"]
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, (
                        "Each letter must be a single token"
                    )
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])

            # Focus logits on the answer position and available letters
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]

            # Get the argmax letter (predicted answer)
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]

            # Evaluate the outcome
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    average = num_passed / total
    print(f"Final: {num_passed}/{total} ({100 * average:.2f}%)")
    return average


def run_chat_eval(
    task_name,
    model,
    tokenizer,
    engine,
    batch_size=1,
    num_samples=1,
    max_new_tokens=512,
    temperature=0.0,
    top_k=50,
    max_problems=None,
):
    """
    Run evaluation on a specific task.

    Args:
        task_name: Name of the task (e.g., 'MMLU', 'ARC-Easy')
        model: Model to evaluate
        tokenizer: Tokenizer
        engine: Engine for generation
        batch_size: Batch size for categorical evaluation
        num_samples: Number of samples for generative evaluation
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        max_problems: Maximum problems to evaluate

    Returns:
        Accuracy (float)
    """
    # Create the evaluation object
    task_module = {
        "HumanEval": HumanEval,
        "MMLU": partial(MMLU, subset="all", split="test"),
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test"),
        "SpellingBee": partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object = task_module()

    # Run the evaluation
    if task_object.eval_type == "generative":
        acc = run_generative_eval(
            task_object,
            tokenizer,
            model,
            engine,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            max_problems=max_problems,
        )
    elif task_object.eval_type == "categorical":
        acc = run_categorical_eval(
            task_object, tokenizer, model, batch_size, max_problems=max_problems
        )
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")

    return acc
