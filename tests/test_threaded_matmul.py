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
    thread_y: int,
    thread_x: int,
    batch_idx: int,
    x: int,
    y: int,
):
    if x < 0 or x >= matrix.shape[1] or y < 0 or y >= matrix.shape[2]:
        shared[thread_y, thread_x] = 0
    else:
        shared[thread_y, thread_x] = matrix[batch_idx, x, y]
    return shared


def compute_sub(
    threads: Tuple[int, int],
    block_size: int,
    x_shared: torch.Tensor,
    y_shared: torch.Tensor,
    sub: torch.Tensor,
) -> torch.Tensor:
    for thread_x, thread_y, block_idx in product(
        range(threads[0]), range(threads[1]), range(block_size)
    ):
        sub[thread_x, thread_y] = sub[thread_x, thread_y] + (
            x_shared[thread_x, block_idx] * y_shared[block_idx, thread_y]
        )
    return sub


def window_matmul_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    window_size: int,
    block_size: int,
    grid: Tuple[int, int, int],
    threads: Tuple[int, int],
):
    # A: b x m x k
    # B: b x m x k
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
                block = block_idx * block_size
                a_m = a_m_start + thread_y
                b_m = b_m_start + thread_y
                ab_k = block + thread_x
                a_shared = _load(A, a_shared, thread_y, thread_x, batch_idx, a_m, ab_k)
                b_shared = _load(B, b_shared, thread_x, thread_y, batch_idx, b_m, ab_k)
            a_shared, b_shared
            compute_sub(threads, block_size, a_shared, b_shared, c_sub)
        for thread_x, thread_y in product(range(block_size), range(block_size)):
            c_m = a_m_start + thread_x
            c_w = b_m_start + thread_y - c_m + window_size
            if c_m >= C.shape[1] or c_w < 0 or c_w >= C.shape[2]:
                continue
            _C[batch_idx, c_m, c_w] = c_sub[thread_x, thread_y]
    return C + _C


def unwindow_matmul_kernel_A(
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
                block = block_idx * block_size
                a_m = a_m_start + thread_y
                b_k = b_k_start + thread_x
                aw_idx = block + thread_x
                bw_idx = block + thread_y
                a_w = aw_idx - thread_y
                b_m = a_m_start - window_size + bw_idx
                a_shared = _load(A, a_shared, thread_y, thread_x, batch_idx, a_m, a_w)
                b_shared = _load(B, b_shared, thread_y, thread_x, batch_idx, b_m, b_k)
            a_shared, b_shared
            c_sub = compute_sub(threads, block_size, a_shared, b_shared, c_sub)
        for thread_x, thread_y in product(range(block_size), range(block_size)):
            c_m = a_m_start + thread_x
            c_k = b_k_start + thread_y
            if c_m >= C.shape[1] or c_k >= C.shape[2]:
                continue
            C[batch_idx, c_m, c_k] = c_sub[thread_x, thread_y]
    return C


def unwindow_matmul_kernel_B(
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
                block = block_idx * block_size
                a_m = a_m_start + thread_y - window_size + block
                b_k = b_k_start + thread_x
                aw_idx = block + thread_x
                bw_idx = block + thread_y
                a_w = aw_idx - thread_y + 2 * window_size - block * 2
                b_m = a_m_start + bw_idx - window_size
                a_shared = _load(A, a_shared, thread_x, thread_y, batch_idx, a_m, a_w)
                b_shared = _load(B, b_shared, thread_y, thread_x, batch_idx, b_m, b_k)
            a_shared, b_shared
            c_sub = compute_sub(threads, block_size, a_shared, b_shared, c_sub)
        for thread_x, thread_y in product(range(block_size), range(block_size)):
            c_m = a_m_start + thread_x
            c_k = b_k_start + thread_y
            if c_m >= C.shape[1] or c_k >= C.shape[2]:
                continue
            C[batch_idx, c_m, c_k] = c_sub[thread_x, thread_y]
    return C


def threaded_window_matmul_fw(
    A: torch.Tensor, B: torch.Tensor, window_size: int, block_size: int
) -> torch.Tensor:
    B = B.transpose(-1, -2)
    full_window_size = window_size * 2 + 1
    *shapes, _ = A.shape
    C = torch.zeros(
        (*shapes, full_window_size), device=A.device, dtype=A.dtype, requires_grad=True
    )
    A = A.reshape(-1, *A.shape[-2:])
    B = B.reshape(-1, *B.shape[-2:])
    C = C.reshape(-1, *C.shape[-2:])

    grid = get_grid(A.shape[0], A.shape[1], window_size, block_size)
    threads = (block_size, block_size)

    C = window_matmul_kernel(A, B, C, window_size, block_size, grid, threads)

    C = C.reshape(*shapes, full_window_size)
    return C


def threaded_window_matmul_bw(
    A: torch.Tensor,
    B: torch.Tensor,
    C_grad: torch.Tensor,
    window_size: int,
    block_size: int,
):
    B = B.transpose(-1, -2)
    A_shape = A.shape
    B_shape = B.shape
    A = A.reshape(-1, *A.shape[-2:])
    A_grad = torch.zeros_like(A)
    B = B.reshape(-1, *B.shape[-2:])
    B_grad = torch.zeros_like(B)
    C_grad = C_grad.reshape(-1, *C_grad.shape[-2:])

    grid = get_grid(A.shape[0], A.shape[1], window_size, block_size)
    threads = (block_size, block_size)

    A_grad = unwindow_matmul_kernel_A(
        C_grad, B, A_grad, window_size, block_size, grid, threads
    )
    B_grad = unwindow_matmul_kernel_B(
        C_grad, A, B_grad, window_size, block_size, grid, threads
    )
    A_grad = A_grad.reshape(*A_shape)
    B_grad = B_grad.reshape(*B_shape)
    return A_grad, B_grad.transpose(-1, -2)


def threaded_unwindow_matmul_fw(
    A: torch.Tensor, B: torch.Tensor, window_size: int, block_size: int
) -> torch.Tensor:
    *shapes, _ = B.shape
    C = torch.zeros_like(B)
    A = A.reshape(-1, *A.shape[-2:])
    B = B.reshape(-1, *B.shape[-2:])
    C = C.reshape(-1, *C.shape[-2:])

    grid = get_grid(A.shape[0], A.shape[1], window_size, block_size)
    threads = (block_size, block_size)

    C = unwindow_matmul_kernel_A(A, B, C, window_size, block_size, grid, threads)

    C = C.reshape(*shapes, C.shape[-1])
    return C


def threaded_unwindow_matmul_bw(
    A: torch.Tensor,
    B: torch.Tensor,
    C_grad: torch.Tensor,
    window_size: int,
    block_size: int,
):
    A_shape = A.shape
    B_shape = B.shape
    A = A.reshape(-1, *A.shape[-2:])
    A_grad = torch.zeros_like(A, dtype=C_grad.dtype)
    B = B.reshape(-1, *B.shape[-2:])
    B_grad = torch.zeros_like(B, dtype=C_grad.dtype)
    C_grad = C_grad.reshape(-1, *C_grad.shape[-2:])

    grid = get_grid(A.shape[0], A.shape[1], window_size, block_size)
    threads = (block_size, block_size)

    A_grad = window_matmul_kernel(
        C_grad, B, A_grad, window_size, block_size, grid, threads
    )
    B_grad = unwindow_matmul_kernel_B(
        A, C_grad, B_grad, window_size, block_size, grid, threads
    )
    A_grad = A_grad.reshape(*A_shape)
    B_grad = B_grad.reshape(*B_shape)
    return A_grad, B_grad


def get_grid(b: int, m: int, window_size: int, block_size: int) -> Tuple[int, int, int]:
    full_window_size = window_size * 2 + 1
    if window_size < block_size and block_size <= m:
        num_w_blocks = 2
    else:
        num_w_blocks = math.ceil(full_window_size / block_size)
    num_m_blocks = math.ceil(m / block_size)
    return (num_w_blocks, num_m_blocks, b)


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

    manual_att_grad, manual_value_grad = threaded_unwindow_matmul_bw(
        att,
        value,
        out_grad,
        window_size,
        block_size,
    )

    for func_name_1, func_name_2 in combinations(funcs, 2):
        out_1 = outs[func_name_1]
        out_2 = outs[func_name_2]
        assert torch.allclose(out_1, out_2, atol=1e-6)
        att_grad_1 = att_grads[func_name_1]
        att_grad_2 = att_grads[func_name_2]
        assert torch.allclose(att_grad_1, att_grad_2, atol=1e-6)
        assert torch.allclose(
            manual_att_grad.view_as(att_grad_1), att_grad_1, atol=1e-6
        )
        assert torch.allclose(
            manual_att_grad.view_as(att_grad_2), att_grad_2, atol=1e-6
        )
        value_grad_1 = value_grads[func_name_1]
        value_grad_2 = value_grads[func_name_2]
        assert torch.allclose(value_grad_1, value_grad_2, atol=1e-6)
        assert torch.allclose(
            manual_value_grad.view_as(value_grad_1), value_grad_1, atol=1e-6
        )
        assert torch.allclose(
            manual_value_grad.view_as(value_grad_2), value_grad_2, atol=1e-6
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
