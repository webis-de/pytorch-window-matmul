from typing import Tuple

import torch
import window_matmul_kernel


class WindowMatmulFunc(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx, inp: torch.Tensor, other: torch.Tensor, window_size: int
    ) -> torch.Tensor:
        output = window_matmul_kernel.window_matmul_forward(inp, other, window_size)
        ctx.save_for_backward(inp, other)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        window_size = (grad_output.shape[-1]) // 2
        inp, other = ctx.saved_tensors
        inp_grad, other_grad = window_matmul_kernel.window_matmul_backward(
            inp, other, window_size, grad_output
        )
        return inp_grad, other_grad, None


def window_matmul(
    inp: torch.Tensor, other: torch.Tensor, window_size: int
) -> torch.Tensor:
    if not window_size:
        return (inp * other.transpose(-1, -2)).sum(dim=-1, keepdim=True)
    *dims, seq_len, hidden_dim = inp.shape
    inp = inp.reshape(-1, seq_len, hidden_dim)
    other = other.reshape(-1, hidden_dim, seq_len)
    out = WindowMatmulFunc.apply(inp, other, window_size)
    out = out.reshape(*dims, seq_len, window_size * 2 + 1)
    return out


class UnwindowMatmulFunc(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx, inp: torch.Tensor, other: torch.Tensor, window_size: int
    ) -> torch.Tensor:
        output = window_matmul_kernel.unwindow_matmul_forward(inp, other, window_size)
        ctx.save_for_backward(inp, other)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        inp, other = ctx.saved_tensors
        window_size = (inp.shape[-1]) // 2
        inp_grad, other_grad = window_matmul_kernel.unwindow_matmul_backward(
            inp, other, window_size, grad_output
        )
        return inp_grad, other_grad, None


def unwindow_matmul(
    inp: torch.Tensor, other: torch.Tensor, window_size: int
) -> torch.Tensor:
    if not window_size:
        return inp * other
    *dims, seq_len, hidden_dim = other.shape
    inp = inp.reshape(-1, seq_len, window_size * 2 + 1)
    other = other.reshape(-1, seq_len, hidden_dim)
    out = UnwindowMatmulFunc.apply(inp, other, window_size)
    out = out.reshape(*dims, seq_len, hidden_dim)
    return out
