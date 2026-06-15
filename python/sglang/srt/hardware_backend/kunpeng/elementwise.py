"""Element-wise operations using kunpeng async_compute operators.

All operations work on bf16 tensors and require contiguous memory layout.
These are utility functions that can be called by other modules as needed.
"""

import torch

from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import (
    load_async_compute,
)


def add_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise addition: x + y (bf16).

    Args:
        x: Input tensor (any dtype, will be converted to bf16)
        y: Input tensor (any dtype, will be converted to bf16)

    Returns:
        x + y as bf16 tensor
    """
    load_async_compute()

    # Convert to bf16 and ensure contiguous
    x_bf16 = x.to(torch.bfloat16).contiguous()
    y_bf16 = y.to(torch.bfloat16).contiguous()

    # Allocate output
    out = torch.empty_like(x_bf16)

    # Call C++ operator
    torch.ops.async_compute.add_out(x_bf16, y_bf16, out)

    return out


def add_scalar_forward(x: torch.Tensor, scalar: float) -> torch.Tensor:
    """Element-wise scalar addition: x + scalar (bf16).

    Args:
        x: Input tensor (any dtype, will be converted to bf16)
        scalar: Scalar value to add

    Returns:
        x + scalar as bf16 tensor
    """
    load_async_compute()

    # Convert to bf16 and ensure contiguous
    x_bf16 = x.to(torch.bfloat16).contiguous()

    # Allocate output
    out = torch.empty_like(x_bf16)

    # Call C++ operator
    torch.ops.async_compute.add_scalar_out(x_bf16, out, float(scalar))

    return out


def mul_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise multiplication: x * y (bf16).

    Args:
        x: Input tensor (any dtype, will be converted to bf16)
        y: Input tensor (any dtype, will be converted to bf16)

    Returns:
        x * y as bf16 tensor
    """
    load_async_compute()

    # Convert to bf16 and ensure contiguous
    x_bf16 = x.to(torch.bfloat16).contiguous()
    y_bf16 = y.to(torch.bfloat16).contiguous()

    # Allocate output
    out = torch.empty_like(x_bf16)

    # Call C++ operator
    torch.ops.async_compute.mul_out(x_bf16, y_bf16, out)

    return out


def tanh_forward(x: torch.Tensor) -> torch.Tensor:
    """Element-wise tanh activation (bf16).

    Args:
        x: Input tensor (any dtype, will be converted to bf16)

    Returns:
        tanh(x) as bf16 tensor
    """
    load_async_compute()

    # Convert to bf16 and ensure contiguous
    x_bf16 = x.to(torch.bfloat16).contiguous()

    # Allocate output
    out = torch.empty_like(x_bf16)

    # Call C++ operator
    torch.ops.async_compute.tanh_out(x_bf16, out)

    return out


def tanh_backward_forward(
    grad_output: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """Tanh backward pass: grad_output * (1 - output^2) (bf16).

    Args:
        grad_output: Gradient from downstream (any dtype, will be converted to bf16)
        output: Forward pass output (tanh result, any dtype, will be converted to bf16)

    Returns:
        grad_input as bf16 tensor
    """
    load_async_compute()

    # Convert to bf16 and ensure contiguous
    grad_output_bf16 = grad_output.to(torch.bfloat16).contiguous()
    output_bf16 = output.to(torch.bfloat16).contiguous()

    # Allocate output
    grad_input = torch.empty_like(grad_output_bf16)

    # Call C++ operator
    torch.ops.async_compute.tanh_backward_out(grad_output_bf16, output_bf16, grad_input)

    return grad_input


def mul_scalar_add_forward(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> torch.Tensor:
    """In-place: x += y * alpha, returns x. All bf16.

    Used in DeepSeek MoE to merge shared expert and routed expert outputs:
        shared_output += routed_output * routed_scaling_factor

    Args:
        x: [M, H] BF16 tensor (will be modified in-place)
        y: [M, H] BF16 tensor (read-only)
        alpha: scalar multiplier (converted to BF16)

    Returns:
        x after in-place modification (x += y * alpha)
    """
    load_async_compute()

    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    if y.dtype != torch.bfloat16:
        y = y.to(torch.bfloat16)
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    # Create a bf16 tensor filled with alpha (contiguous, not a view)
    alpha_t = torch.full_like(y, alpha, dtype=torch.bfloat16, device=x.device)

    # Step 1: y_scaled = y * alpha
    y_scaled = torch.empty_like(y)
    torch.ops.async_compute.mul_out(y, alpha_t, y_scaled)

    # Step 2: x += y_scaled (in-place add)
    torch.ops.async_compute.add_out(x, y_scaled, x)

    return x
