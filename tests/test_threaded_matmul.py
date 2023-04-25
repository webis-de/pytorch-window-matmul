import math
from functools import partial
from itertools import combinations, product
from typing import Tuple

import pytest
import torch

from tests.conftest import BLOCK_SIZES, DEVICES, DIMS, DTYPES, WINDOW_SIZES
from tests.test_unwindow_matmul import pytorch_unwindow_matmul
from tests.test_window_matmul import pytorch_window_matmul
from tests.utils import get_att, get_key, get_query, get_value


def _load(
    matrix: torch.Tensor,
    shared: torch.Tensor,
    thread_x: int,
    thread_y: int,
    batch_idx: int,
    x: int,
    y: int,
):
    if x < 0 or x >= matrix.shape[1] or y < 0 or y >= matrix.shape[2]:
        shared[thread_x, thread_y] = 0
    else:
        shared[thread_x, thread_y] = matrix[batch_idx, x, y]
    return shared


def _window_matmul_fw_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    window_size: int,
    block_size: int,
    grid: Tuple[int, int, int],
    threads: Tuple[int, int],
):
    # A: b x m x k
    # B: b x k x m
    # C: b x m x 2w + 1
    _C = C.clone().detach()
    for block_z, block_y, block_x in product(
        range(grid[2]), range(grid[1]), range(grid[0])
    ):
        batch_idx = block_z
        a_m_start = block_size * block_x
        b_m_start = block_size * block_y
        a_m_end = a_m_start + block_size
        b_m_end = b_m_start + block_size
        dist = min(abs(a_m_end - b_m_start), abs(b_m_end - a_m_start))
        if block_size < window_size and dist > window_size:
            continue
        c_sub = torch.zeros(block_size, block_size)
        num_blocks = math.ceil(A.shape[2] / block_size)
        for block_idx in range(num_blocks):
            a_shared = torch.zeros(
                block_size, block_size, device=C.device, dtype=C.dtype
            )
            b_shared = torch.zeros(
                block_size, block_size, device=C.device, dtype=C.dtype
            )
            for thread_y, thread_x in product(range(threads[1]), range(threads[0])):
                a_m = a_m_start + thread_x
                b_m = b_m_start + thread_x
                a_k = block_idx * block_size + thread_y
                b_k = block_idx * block_size + thread_y
                a_shared = _load(A, a_shared, thread_x, thread_y, batch_idx, a_m, a_k)
                b_shared = _load(B, b_shared, thread_y, thread_x, batch_idx, b_k, b_m)
            a_shared, b_shared
            for thread_x, thread_y, k_block_idx in product(
                range(threads[0]), range(threads[1]), range(block_size)
            ):
                c_sub[thread_x, thread_y] = c_sub[thread_x, thread_y] + (
                    a_shared[thread_x, k_block_idx] * b_shared[k_block_idx, thread_y]
                )
        for thread_x, thread_y in product(range(block_size), range(block_size)):
            a_m = a_m_start + thread_x
            b_m = b_m_start + thread_y
            w_idx = b_m - a_m + window_size
            if a_m >= C.shape[1] or w_idx < 0 or w_idx >= window_size * 2 + 1:
                continue
            _C[
                batch_idx,
                a_m,
                w_idx,
            ] = c_sub[thread_x, thread_y]
    return C + _C


def window_matmul_fw_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    window_size: int,
    block_size: int,
    grid: Tuple[int, int, int],
    threads: Tuple[int, int],
):
    # A: b x m x k
    # B: b x k x m
    # C: b x m x 2w + 1
    _C = C.clone().detach()
    for block_z, block_y, block_x in product(
        range(grid[2]), range(grid[1]), range(grid[0])
    ):
        batch_idx = block_z
        a_m_start = block_size * block_y
        b_m_start = a_m_start + block_size * block_x - window_size
        c_sub = torch.zeros(block_size, block_size)
        num_blocks = math.ceil(A.shape[2] / block_size)
        for block_idx in range(num_blocks):
            a_shared = torch.zeros(
                block_size, block_size, device=C.device, dtype=C.dtype
            )
            b_shared = torch.zeros(
                block_size, block_size, device=C.device, dtype=C.dtype
            )
            for thread_y, thread_x in product(range(threads[1]), range(threads[0])):
                a_m = a_m_start + thread_y
                b_m = b_m_start + thread_x
                a_k = block_idx * block_size + thread_x
                b_k = block_idx * block_size + thread_y
                a_shared = _load(A, a_shared, thread_y, thread_x, batch_idx, a_m, a_k)
                b_shared = _load(B, b_shared, thread_y, thread_x, batch_idx, b_k, b_m)
            a_shared, b_shared
            for thread_x, thread_y, ab_k in product(
                range(threads[0]), range(threads[1]), range(block_size)
            ):
                c_sub[thread_x, thread_y] = c_sub[thread_x, thread_y] + (
                    a_shared[thread_x, ab_k] * b_shared[ab_k, thread_y]
                )
        for thread_x, thread_y in product(range(block_size), range(block_size)):
            c_x = a_m_start + thread_x
            b_m = b_m_start + thread_y
            c_y = b_m - c_x + window_size
            if c_x >= C.shape[1] or c_y < 0 or c_y >= 2 * window_size + 1:
                continue
            _C[batch_idx, c_x, c_y] = c_sub[thread_x, thread_y]
    return C + _C


def window_matmul_bw_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    grad_C: torch.Tensor,
    grad_A: torch.Tensor,
    grad_B: torch.Tensor,
    window_size: int,
    block_size: int,
    grid: Tuple[int, int, int],
    threads: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # A: b x m x k
    # B: b x k x m
    # C: b x m x 2w + 1
    for block_z, block_y, block_x in product(
        range(grid[2]), range(grid[1]), range(grid[0])
    ):
        batch_idx = block_z
        c_m_start = block_size * block_y
        ab_k_start = block_size * block_x
        a_sub = torch.zeros(block_size, block_size, device=A.device, dtype=A.dtype)
        b_sub = torch.zeros(block_size, block_size, device=B.device, dtype=B.dtype)
        if window_size < block_size and block_size <= A.shape[1]:
            num_blocks = 2
        else:
            num_blocks = math.ceil(grad_C.shape[2] / block_size)
        for block_idx in range(num_blocks):
            a_shared = torch.zeros(
                block_size, block_size, device=A.device, dtype=A.dtype
            )
            b_shared = torch.zeros(
                block_size, block_size, device=B.device, dtype=B.dtype
            )
            ac_shared = torch.zeros(
                block_size, block_size, device=grad_C.device, dtype=grad_C.dtype
            )
            bc_shared = torch.zeros(
                block_size, block_size, device=grad_C.device, dtype=grad_C.dtype
            )
            for thread_y, thread_x in product(range(threads[1]), range(threads[0])):
                aw_idx = block_idx * block_size + thread_y - window_size
                a_k = ab_k_start + thread_x
                a_m = c_m_start + aw_idx

                bw_idx = block_idx * block_size + thread_x - window_size
                b_k = ab_k_start + thread_y
                b_m = c_m_start + bw_idx

                c_w_idx = block_idx * block_size + thread_x
                ac_m = c_m_start + thread_y
                ac_w = c_w_idx - thread_y
                bc_m = ac_m - window_size + block_idx * block_size
                bc_w = ac_w + 2 * window_size - block_idx * block_size * 2

                a_shared = _load(A, a_shared, thread_x, thread_y, batch_idx, a_m, a_k)
                b_shared = _load(B, b_shared, thread_y, thread_x, batch_idx, b_k, b_m)
                ac_shared = _load(
                    grad_C, ac_shared, thread_x, thread_y, batch_idx, ac_m, ac_w
                )
                bc_shared = _load(
                    grad_C, bc_shared, thread_y, thread_x, batch_idx, bc_m, bc_w
                )
            a_shared, b_shared, ac_shared, bc_shared
            for thread_x, thread_y, w_idx in product(
                range(threads[0]), range(threads[1]), range(block_size)
            ):
                a_sub[thread_x, thread_y] += (
                    b_shared[thread_x, w_idx] * ac_shared[w_idx, thread_y]
                )
                b_sub[thread_y, thread_x] += (
                    a_shared[thread_x, w_idx] * bc_shared[w_idx, thread_y]
                )
        for thread_x, thread_y in product(range(block_size), range(block_size)):
            a_m = c_m_start + thread_x
            b_m = c_m_start + thread_x
            ab_k = ab_k_start + thread_y
            if a_m >= 0 and a_m < A.shape[1] and ab_k < A.shape[2]:
                grad_A[batch_idx, a_m, ab_k] = a_sub[thread_y, thread_x]
            if b_m >= 0 and ab_k < B.shape[1] and b_m < B.shape[2]:
                grad_B[batch_idx, ab_k, b_m] = b_sub[thread_x, thread_y]
    return grad_A, grad_B


def _unwindow_matmul_fw_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    window_size: int,
    block_size: int,
    grid: Tuple[int, int, int],
    threads: Tuple[int, int],
):
    # A: b x m x 2w + 1
    # B: b x m x k
    # C: b x m x k
    _C = C.clone().detach()
    for block_z, block_y, block_x in product(
        range(grid[2]), range(grid[1]), range(grid[0])
    ):
        batch_idx = block_z
        a_m_start = block_size * block_y
        b_k_start = block_size * block_x
        c_sub = torch.zeros(block_size, block_size, device=C.device, dtype=C.dtype)
        num_blocks = math.ceil(A.shape[2] / block_size)
        for block_idx in range(num_blocks):
            a_shared = torch.zeros(
                block_size, block_size, device=C.device, dtype=C.dtype
            )
            b_shared = torch.zeros(
                block_size, block_size, device=C.device, dtype=C.dtype
            )
            for thread_y, thread_x in product(range(threads[1]), range(threads[0])):
                a_m = a_m_start + thread_y
                b_k = b_k_start + thread_x
                a_w = block_idx * block_size + thread_x
                b_m = block_idx * block_size + thread_y
                if a_w < 0 or a_m >= A.shape[1] or a_w >= A.shape[2]:
                    a_shared[thread_y, thread_x] = 0
                else:
                    a_shared[thread_y, thread_x] = A[batch_idx, a_m, a_w]
                if b_m < 0 or b_m >= B.shape[1] or b_k >= B.shape[2]:
                    b_shared[thread_y, thread_x] = 0
                else:
                    b_shared[thread_y, thread_x] = B[batch_idx, b_m, b_k]
            a_shared, b_shared
            for thread_x, thread_y, w_block_idx in product(
                range(threads[0]), range(threads[1]), range(block_size)
            ):
                a_w = block_idx * block_size + thread_x
                w_idx = block_idx * block_size + w_block_idx
                b_w = a_w - window_size + w_idx
                bw_block_idx = w_block_idx - window_size + thread_x
                if b_w < 0 or b_w >= B.shape[1] or w_idx >= window_size * 2 + 1:
                    continue
                c_sub[thread_x, thread_y] += (
                    a_shared[thread_x, w_block_idx] * b_shared[bw_block_idx, thread_y]
                )
        for thread_x, thread_y in product(range(block_size), range(block_size)):
            c_m = a_m_start + thread_x
            c_k = b_k_start + thread_y
            if c_m >= C.shape[1] or c_k >= C.shape[2]:
                continue
            _C[batch_idx, c_m, c_k] = c_sub[thread_x, thread_y]
    return C + _C


def unwindow_matmul_fw_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    window_size: int,
    block_size: int,
    grid: Tuple[int, int, int],
    threads: Tuple[int, int],
):
    # A: b x m x 2w + 1
    # B: b x m x k
    # C: b x m x k
    for block_z, block_y, block_x in product(
        range(grid[2]), range(grid[1]), range(grid[0])
    ):
        batch_idx = block_z
        a_m_start = block_size * block_y
        b_k_start = block_size * block_x
        c_sub = torch.zeros(block_size, block_size)
        if window_size < block_size and block_size <= A.shape[1]:
            num_blocks = 2
        else:
            num_blocks = math.ceil(A.shape[2] / block_size)
        for block_idx in range(num_blocks):
            a_shared = torch.zeros(
                block_size, block_size, device=C.device, dtype=C.dtype
            )
            b_shared = torch.zeros(
                block_size, block_size, device=C.device, dtype=C.dtype
            )
            for thread_y, thread_x in product(range(threads[1]), range(threads[0])):
                a_m = a_m_start + thread_y
                b_k = b_k_start + thread_x
                aw_idx = block_idx * block_size + thread_x
                bw_idx = block_idx * block_size + thread_y
                a_w = aw_idx - thread_y
                b_m = a_m_start - window_size + bw_idx
                a_shared = _load(A, a_shared, thread_y, thread_x, batch_idx, a_m, a_w)
                b_shared = _load(B, b_shared, thread_y, thread_x, batch_idx, b_m, b_k)
            a_shared, b_shared
            for thread_x, thread_y, w_block_idx in product(
                range(threads[0]), range(threads[1]), range(block_size)
            ):
                c_sub[thread_x, thread_y] = c_sub[thread_x, thread_y] + (
                    a_shared[thread_x, w_block_idx] * b_shared[w_block_idx, thread_y]
                )
        for thread_x, thread_y in product(range(block_size), range(block_size)):
            c_m = a_m_start + thread_x
            c_k = b_k_start + thread_y
            if c_m >= C.shape[1] or c_k >= C.shape[2]:
                continue
            C[batch_idx, c_m, c_k] = c_sub[thread_x, thread_y]
    return C


def unwindow_matmul_bw_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    grad_C: torch.Tensor,
    grad_A: torch.Tensor,
    grad_B: torch.Tensor,
    window_size: int,
    block_size: int,
    grid: Tuple[int, int, int],
    threads: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # A: b x m x 2w + 1
    # B: b x m x k
    # C: b x m x k
    for block_z, block_y, block_x in product(
        range(grid[2]), range(grid[1]), range(grid[0])
    ):
        batch_idx = block_z
        c_m_start = block_size * block_y
        ab_k_start = block_size * block_x
        a_sub = torch.zeros(block_size, block_size, device=A.device, dtype=A.dtype)
        b_sub = torch.zeros(block_size, block_size, device=B.device, dtype=B.dtype)
        if window_size < block_size and block_size <= A.shape[1]:
            num_blocks = 2
        else:
            num_blocks = math.ceil(grad_C.shape[2] / block_size)
        for block_idx in range(num_blocks):
            a_shared = torch.zeros(
                block_size, block_size, device=A.device, dtype=A.dtype
            )
            b_shared = torch.zeros(
                block_size, block_size, device=B.device, dtype=B.dtype
            )
            ac_shared = torch.zeros(
                block_size, block_size, device=grad_C.device, dtype=grad_C.dtype
            )
            bc_shared = torch.zeros(
                block_size, block_size, device=grad_C.device, dtype=grad_C.dtype
            )
            for thread_y, thread_x in product(range(threads[1]), range(threads[0])):
                aw_idx = block_idx * block_size + thread_y - window_size
                a_k = ab_k_start + thread_x
                a_m = c_m_start + aw_idx

                bw_idx = block_idx * block_size + thread_x - window_size
                b_k = ab_k_start + thread_y
                b_m = c_m_start + bw_idx

                c_w_idx = block_idx * block_size + thread_x
                ac_m = c_m_start + thread_y
                ac_w = c_w_idx - thread_y
                bc_m = ac_m - window_size + block_idx * block_size
                bc_w = ac_w + 2 * window_size - block_idx * block_size * 2

                a_shared = _load(A, a_shared, thread_x, thread_y, batch_idx, a_m, a_k)
                b_shared = _load(B, b_shared, thread_y, thread_x, batch_idx, b_k, b_m)
                ac_shared = _load(
                    grad_C, ac_shared, thread_x, thread_y, batch_idx, ac_m, ac_w
                )
                bc_shared = _load(
                    grad_C, bc_shared, thread_y, thread_x, batch_idx, bc_m, bc_w
                )
            a_shared, b_shared, ac_shared, bc_shared
            for thread_x, thread_y, w_idx in product(
                range(threads[0]), range(threads[1]), range(block_size)
            ):
                a_sub[thread_x, thread_y] += (
                    b_shared[thread_x, w_idx] * ac_shared[w_idx, thread_y]
                )
                b_sub[thread_y, thread_x] += (
                    a_shared[thread_x, w_idx] * bc_shared[w_idx, thread_y]
                )
        for thread_x, thread_y in product(range(block_size), range(block_size)):
            a_m = c_m_start + thread_x
            b_m = c_m_start + thread_x
            ab_k = ab_k_start + thread_y
            if a_m >= 0 and a_m < A.shape[1] and ab_k < A.shape[2]:
                grad_A[batch_idx, a_m, ab_k] = a_sub[thread_y, thread_x]
            if b_m >= 0 and ab_k < B.shape[1] and b_m < B.shape[2]:
                grad_B[batch_idx, ab_k, b_m] = b_sub[thread_x, thread_y]
    return grad_A, grad_B


def threaded_window_matmul_fw(
    A: torch.Tensor, B: torch.Tensor, window_size: int, block_size: int
) -> torch.Tensor:
    full_window_size = window_size * 2 + 1
    *shapes, _ = A.shape
    C = torch.zeros(
        (*shapes, full_window_size), device=A.device, dtype=A.dtype, requires_grad=True
    )
    A = A.reshape(-1, *A.shape[-2:])
    B = B.reshape(-1, *B.shape[-2:])
    C = C.reshape(-1, *C.shape[-2:])

    batch_size = A.shape[0]
    seq_len = A.shape[1]

    if window_size < block_size and block_size <= A.shape[1]:
        num_window_blocks = 2
    else:
        num_window_blocks = math.ceil(full_window_size / block_size)

    grid = (
        num_window_blocks,
        math.ceil(seq_len / block_size),
        batch_size,
    )
    threads = (block_size, block_size)

    C = window_matmul_fw_kernel(A, B, C, window_size, block_size, grid, threads)

    C = C.reshape(*shapes, full_window_size)
    return C


def threaded_window_matmul_bw(
    A: torch.Tensor,
    B: torch.Tensor,
    C_grad: torch.Tensor,
    window_size: int,
    block_size: int,
):
    A_shape = A.shape
    B_shape = B.shape
    A = A.reshape(-1, *A.shape[-2:])
    A_grad = torch.zeros_like(A)
    B = B.reshape(-1, *B.shape[-2:])
    B_grad = torch.zeros_like(B)
    C_grad = C_grad.reshape(-1, *C_grad.shape[-2:])

    batch_size = A.shape[0]
    seq_len = A.shape[1]
    hidden_dim = A.shape[2]

    grid = (
        math.ceil(hidden_dim / block_size),
        math.ceil(seq_len / block_size),
        batch_size,
    )
    threads = (block_size, block_size)

    A_grad, B_grad = window_matmul_bw_kernel(
        A, B, C_grad, A_grad, B_grad, window_size, block_size, grid, threads
    )
    A = A.reshape(*A_shape)
    B = B.reshape(*B_shape)
    return A_grad, B_grad


def threaded_unwindow_matmul_fw(
    A: torch.Tensor, B: torch.Tensor, window_size: int, block_size: int
) -> torch.Tensor:
    *shapes, _ = B.shape
    C = torch.zeros_like(B)
    A = A.reshape(-1, *A.shape[-2:])
    B = B.reshape(-1, *B.shape[-2:])
    C = C.reshape(-1, *C.shape[-2:])

    batch_size = A.shape[0]
    seq_len = A.shape[1]
    full_window_size = C.shape[2]

    grid = (
        math.ceil(full_window_size / block_size),
        math.ceil(seq_len / block_size),
        batch_size,
    )
    threads = (block_size, block_size)

    C = unwindow_matmul_fw_kernel(A, B, C, window_size, block_size, grid, threads)

    C = C.reshape(*shapes, C.shape[-1])
    return C


def threaded_unwindow_matmul_bw(
    A: torch.Tensor,
    B: torch.Tensor,
    C_grad: torch.Tensor,
    window_size: int,
    block_size: int,
    _A_grad: torch.Tensor,
    _B_grad: torch.Tensor,
):
    A_shape = A.shape
    B_shape = B.shape
    A = A.reshape(-1, *A.shape[-2:])
    A_grad = torch.zeros_like(A)
    B = B.reshape(-1, *B.shape[-2:])
    B_grad = torch.zeros_like(B)
    C_grad = C_grad.reshape(-1, *C_grad.shape[-2:])

    batch_size = A.shape[0]
    seq_len = A.shape[1]
    full_window_size = A.shape[2]

    grid = (
        math.ceil(full_window_size / block_size),
        math.ceil(seq_len / block_size),
        batch_size,
    )
    threads = (block_size, block_size)

    A_grad, B_grad = unwindow_matmul_bw_kernel(
        A, B, C_grad, A_grad, B_grad, window_size, block_size, grid, threads
    )
    A = A.reshape(*A_shape)
    B = B.reshape(*B_shape)
    return A_grad, B_grad


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("window_size", WINDOW_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_window_matmul(
    dim: Tuple[int, int, int, int],
    window_size: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    if device == torch.device("cpu") and dtype == torch.float16:
        return
    funcs = {
        "pytorch": partial(pytorch_window_matmul, window_size=window_size),
        "threaded": partial(
            threaded_window_matmul_fw, window_size=window_size, block_size=block_size
        ),
        # "custom": partial(window_matmul, window_size=window_size),
    }

    query = get_query(*dim).to(device, dtype)
    key = get_key(*dim).to(device, dtype)

    atts = {}
    query_grads = {}
    key_grads = {}

    att_grad = None
    for func_name, func in funcs.items():
        _query = query.clone().requires_grad_(True)
        _key = key.clone().requires_grad_(True)
        att = func(_query, _key.transpose(-1, -2))
        atts[func_name] = att

        def _get_grad(grad: torch.Tensor):
            nonlocal att_grad
            att_grad = grad

        att.register_hook(_get_grad)
        (att * torch.arange(att.numel()).to(att).view_as(att)).mean().backward()
        query_grads[func_name] = _query.grad
        key_grads[func_name] = _key.grad

    manual_query_grad, manual_key_grad = threaded_window_matmul_bw(
        query, key.transpose(-1, -2), att_grad, window_size, block_size
    )
    manual_key_grad = manual_key_grad.transpose(-1, -2)

    for func_name_1, func_name_2 in combinations(funcs, 2):
        att_1 = atts[func_name_1]
        att_2 = atts[func_name_2]
        assert torch.allclose(att_1, att_2, atol=1e-6)
        query_grad_1 = query_grads[func_name_1]
        query_grad_2 = query_grads[func_name_2]
        assert torch.allclose(query_grad_1, query_grad_2, atol=1e-6)
        assert torch.allclose(
            manual_query_grad.view_as(query_grad_1), query_grad_1, atol=1e-6
        )
        assert torch.allclose(
            manual_query_grad.view_as(query_grad_2), query_grad_2, atol=1e-6
        )
        key_grad_1 = key_grads[func_name_1]
        key_grad_2 = key_grads[func_name_2]
        assert torch.allclose(key_grad_1, key_grad_2, atol=1e-6)
        assert torch.allclose(
            manual_key_grad.view_as(key_grad_1), key_grad_1, atol=1e-6
        )
        assert torch.allclose(
            manual_key_grad.view_as(key_grad_2), key_grad_2, atol=1e-6
        )


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("window_size", WINDOW_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_unwindow_matmul(
    dim: Tuple[int, int, int, int],
    window_size: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    if device == torch.device("cpu") and dtype == torch.float16:
        return
    funcs = {
        "pytorch": partial(pytorch_unwindow_matmul, window_size=window_size),
        "threaded": partial(
            threaded_unwindow_matmul_fw, window_size=window_size, block_size=block_size
        ),
        # "custom": partial(unwindow_matmul, window_size=window_size),
    }

    att = get_att(*dim[:-1], window_size)
    value = get_value(*dim)

    outs = {}
    att_grads = {}
    value_grads = {}

    out_grad = None
    for func_name, func in funcs.items():
        _att = att.clone().to(device, dtype).requires_grad_(True)
        _value = value.clone().to(device, dtype).requires_grad_(True)
        out = func(_att, _value)
        outs[func_name] = out

        def _get_grad(grad: torch.Tensor):
            nonlocal out_grad
            out_grad = grad

        out.register_hook(_get_grad)
        (out * torch.arange(out.numel()).to(out).view_as(out)).mean().backward()
        att_grads[func_name] = _att.grad
        value_grads[func_name] = _value.grad

    manual_query_grad, manual_key_grad = threaded_unwindow_matmul_bw(
        att,
        value,
        out_grad,
        window_size,
        block_size,
        att_grads["pytorch"],
        value_grads["pytorch"],
    )

    for func_name_1, func_name_2 in combinations(funcs, 2):
        out_1 = outs[func_name_1]
        out_2 = outs[func_name_2]
        assert torch.allclose(out_1, out_2, atol=1e-6)
        att_grad_1 = att_grads[func_name_1]
        att_grad_2 = att_grads[func_name_2]
        assert torch.allclose(att_grad_1, att_grad_2, atol=1e-6)
        assert torch.allclose(
            manual_query_grad.view_as(att_grad_1), att_grad_1, atol=1e-6
        )
        assert torch.allclose(
            manual_query_grad.view_as(att_grad_2), att_grad_2, atol=1e-6
        )
        value_grad_1 = value_grads[func_name_1]
        value_grad_2 = value_grads[func_name_2]
        assert torch.allclose(value_grad_1, value_grad_2, atol=1e-6)
        assert torch.allclose(
            manual_key_grad.view_as(value_grad_1), value_grad_1, atol=1e-6
        )
        assert torch.allclose(
            manual_key_grad.view_as(value_grad_2), value_grad_2, atol=1e-6
        )


def main():
    for dim in DIMS:
        for window_size in WINDOW_SIZES:
            for block_size in BLOCK_SIZES:
                for device in DEVICES:
                    for dtype in DTYPES:
                        test_window_matmul(dim, window_size, block_size, device, dtype)
                        test_unwindow_matmul(
                            dim, window_size, block_size, device, dtype
                        )


if __name__ == "__main__":
    main()
