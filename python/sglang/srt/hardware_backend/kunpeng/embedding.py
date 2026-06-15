"""Embedding lookup using kunpeng async_compute operators."""

import torch

from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import (
    load_async_compute,
)


def embedding_forward(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
) -> torch.Tensor:
    """Embedding table lookup for TP-sharded vocabulary.

    Args:
        input_ids: [n_tokens] int64, token IDs to look up
        weight: [vocab_shard_size, hidden] bf16, TP-sharded embedding table
        vocab_start: start of this shard's vocab range (inclusive)
        vocab_end: end of this shard's vocab range (exclusive)

    Returns:
        [n_tokens, hidden] bf16, embedding vectors
    """
    load_async_compute()

    if input_ids.dtype != torch.int64:
        input_ids = input_ids.to(torch.int64)
    if weight.dtype != torch.bfloat16:
        weight = weight.to(torch.bfloat16)

    n_tokens = input_ids.shape[0]
    hidden = weight.shape[1]

    output = torch.empty(
        (n_tokens, hidden), dtype=torch.bfloat16, device=input_ids.device
    )

    torch.ops.async_compute.embedding_out(
        input_ids, weight, output, vocab_start, vocab_end
    )

    return output
