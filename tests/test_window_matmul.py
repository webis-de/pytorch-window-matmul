from functools import partial
from itertools import combinations
from typing import Tuple

import pytest
import torch

from window_matmul import window_matmul

from tests.conftest import DEVICES, DIMS, WINDOW_SIZES
from tests.utils import get_key, get_query, to_windowed


def python_window_matmul(
    A: torch.Tensor, B: torch.Tensor, window_size: int
) -> torch.Tensor:
    *shapes, seq_len, hidden_dim = A.shape
    A = A.reshape(-1, *A.shape[-2:])
    B = B.reshape(-1, *B.shape[-2:])
    batch_size = A.shape[0]
    out = torch.zeros(
        (batch_size, seq_len, window_size * 2 + 1),
        device=A.device,
        dtype=A.dtype,
    )
    for b in range(batch_size):
        for s in range(seq_len):
            for w in range(max(0, s - window_size), min(seq_len, s + window_size + 1)):
                w_idx = w - min(0, s - window_size) - max(0, s - window_size)
                for h in range(hidden_dim):
                    out[b, s, w_idx] += A[b, s, h] * B[b, h, w]
    out = out.reshape(*shapes, seq_len, window_size * 2 + 1)
    return out


def pytorch_window_matmul(
    A: torch.Tensor, B: torch.Tensor, window_size: int
) -> torch.Tensor:
    B = B.transpose(-1, -2)
    windowed_B = to_windowed(B, window_size * 2 + 1).transpose(-1, -2)
    out = torch.matmul(A.unsqueeze(-2), windowed_B).squeeze(-2)
    return out


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("window_size", WINDOW_SIZES)
@pytest.mark.parametrize("device", DEVICES)
def test_window_matmul(
    dim: Tuple[int, int, int, int],
    window_size: int,
    device: torch.device,
):
    funcs = {
        "python": partial(python_window_matmul, window_size=window_size),
        "pytorch": partial(pytorch_window_matmul, window_size=window_size),
        "custom": partial(window_matmul, window_size=window_size),
    }

    query = get_query(*dim)
    key = get_key(*dim)

    atts = {}
    query_grads = {}
    key_grads = {}

    for func_name, func in funcs.items():
        _query = query.clone().to(device).requires_grad_(True)
        _key = key.clone().to(device).requires_grad_(True)
        att = func(_query, _key.transpose(-1, -2))
        atts[func_name] = att
        (att * torch.arange(att.numel()).to(att).view_as(att)).mean().backward()
        query_grads[func_name] = _query.grad
        key_grads[func_name] = _key.grad

    for func_name_1, func_name_2 in combinations(funcs, 2):
        att_1 = atts[func_name_1]
        att_2 = atts[func_name_2]
        assert torch.allclose(att_1, att_2, atol=1e-6)
        query_grad_1 = query_grads[func_name_1]
        query_grad_2 = query_grads[func_name_2]
        assert torch.allclose(query_grad_1, query_grad_2, atol=1e-6)
        key_grad_1 = key_grads[func_name_1]
        key_grad_2 = key_grads[func_name_2]
        assert torch.allclose(key_grad_1, key_grad_2, atol=1e-6)
