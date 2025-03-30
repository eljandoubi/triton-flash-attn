"""
Add torch ops.
"""

import torch
import triton
from kernels.add import fwd_add


class Add(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape, "Input tensors must have the same shape"
        output = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fwd_add[grid](
            x.flatten(),
            y.flatten(),
            output.flatten(),
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output

    def backward(ctx, dz: torch.Tensor):
        return dz, dz
