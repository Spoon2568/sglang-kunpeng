"""RMSNorm forward using kunpeng async_compute operators."""

from typing import Optional, Tuple, Union

import torch

from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import (
    load_async_compute,
)


def _needs_reshape(x: torch.Tensor) -> bool:
    return x.dim() != 2


def rmsnorm_forward_kunpeng(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    load_async_compute()

    if _needs_reshape(x):
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        if residual is not None:
            residual = residual.reshape(-1, orig_shape[-1])
        if post_residual_addition is not None:
            post_residual_addition = post_residual_addition.reshape(
                -1, orig_shape[-1]
            )
    else:
        orig_shape = None

    if residual is not None:
        if post_residual_addition is not None:
            residual.add_(post_residual_addition)
        torch.ops.async_compute.add_rmsnorm_out(
            x, self.weight.data, residual, x, self.variance_epsilon
        )
    else:
        if not x.is_contiguous():
            x = x.contiguous()
        torch.ops.async_compute.rmsnorm_out(
            x, self.weight.data, x, self.variance_epsilon
        )

    if orig_shape is not None:
        x = x.reshape(orig_shape)
        if residual is not None:
            residual = residual.reshape(orig_shape)

    if residual is None:
        return x
    return x, residual
