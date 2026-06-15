"""Argmax operations using kunpeng async_compute operators for greedy sampling."""

import numpy as np
import torch

from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import (
    load_async_compute,
)


def argmax_forward(logits: torch.Tensor) -> torch.Tensor:
    """Compute argmax over the last dimension using kunpeng async_compute.

    Args:
        logits: [batch_size, vocab_size] BF16 tensor

    Returns:
        [batch_size] int64 tensor containing the argmax indices
    """
    load_async_compute()

    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits, got shape {logits.shape}")

    batch_size, vocab_size = logits.shape

    # Ensure input is bf16
    if logits.dtype != torch.bfloat16:
        logits = logits.to(torch.bfloat16)

    # Ensure contiguous
    if not logits.is_contiguous():
        logits = logits.contiguous()

    # ArgmaxResult struct: {int32 index; float32 value;} = 8 bytes
    ARGMAX_RESULT_SIZE = 8
    output_buffer = torch.empty(
        (batch_size, ARGMAX_RESULT_SIZE),
        dtype=torch.uint8,
        device=logits.device,
    )

    # Call the async_compute argmax operator
    torch.ops.async_compute.argmax_out(logits, output_buffer)

    # Parse the output buffer to extract indices
    # Convert to CPU for numpy parsing, then back to device
    output_cpu = output_buffer.cpu().numpy()

    # Define structured dtype matching ArgmaxResult
    argmax_dtype = np.dtype([("index", np.int32), ("value", np.float32)])

    # Parse the buffer
    parsed = np.frombuffer(output_cpu.tobytes(), dtype=argmax_dtype)

    # Extract indices and convert to int64 tensor
    indices = torch.from_numpy(parsed["index"].copy()).to(torch.int64)

    # Move back to original device if needed
    if logits.device.type != "cpu":
        indices = indices.to(logits.device)

    return indices
