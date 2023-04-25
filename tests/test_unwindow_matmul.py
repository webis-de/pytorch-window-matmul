from functools import partial
from itertools import combinations
from typing import Tuple

import pytest
import torch

from window_matmul import unwindow_matmul

from tests.conftest import DEVICES, DIMS, DTYPES, WINDOW_SIZES
from tests.utils import get_att, get_value, to_windowed


def python_unwindow_matmul(A: torch.Tensor, B: torch.Tensor, window_size: int):
    *shapes, seq_len, hidden_dim = B.shape
    A = A.reshape(-1, *A.shape[-2:])
    B = B.reshape(-1, *B.shape[-2:])
    batch_size = A.shape[0]
    out = torch.zeros_like(B)
    for b in range(batch_size):
        for s in range(seq_len):
            for w in range(max(0, s - window_size), min(seq_len, s + window_size + 1)):
                w_idx = w - min(0, s - window_size) - max(0, s - window_size)
                for h in range(hidden_dim):
                    out[b, s, h] += A[b, s, w_idx] * B[b, w, h]
    out = out.reshape(*shapes, seq_len, hidden_dim)
    return out


def pytorch_unwindow_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    windowed_B = to_windowed(B, window_size * 2 + 1)
    out = torch.matmul(A.unsqueeze(-2), windowed_B).squeeze(-2)
    return out


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("window_size", WINDOW_SIZES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_unwindow_matmul(
    dim: Tuple[int, int, int, int],
    window_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    if device == torch.device("cpu") and dtype == torch.float16:
        return
    funcs = {
        "python": partial(python_unwindow_matmul, window_size=window_size),
        "pytorch": partial(pytorch_unwindow_matmul, window_size=window_size),
        "custom": partial(unwindow_matmul, window_size=window_size),
    }

    att = get_att(*dim[:-1], window_size)
    value = get_value(*dim)

    outs = {}
    att_grads = {}
    value_grads = {}

    for func_name, func in funcs.items():
        _att = att.clone().to(device, dtype).requires_grad_(True)
        _value = value.clone().to(device, dtype).requires_grad_(True)
        out = func(_att, _value)
        (out * torch.arange(out.numel()).to(out).view_as(out)).mean().backward()
        outs[func_name] = out
        att_grads[func_name] = _att.grad
        value_grads[func_name] = _value.grad

    for func_name_1, func_name_2 in combinations(funcs, 2):
        out_1 = outs[func_name_1]
        out_2 = outs[func_name_2]
        assert torch.allclose(out_1, out_2, atol=1e-6)
        # att_grad_1 = att_grads[func_name_1]
        # att_grad_2 = att_grads[func_name_2]
        # assert torch.allclose(att_grad_1, att_grad_2, atol=1e-6)
        # value_grad_1 = value_grads[func_name_1]
        # value_grad_2 = value_grads[func_name_2]
        # assert torch.allclose(value_grad_1, value_grad_2, atol=1e-6)
