import math
from itertools import combinations
from typing import Tuple

import pytest
import torch

from window_matmul import unwindow_matmul, window_matmul

from tests.conftest import DEVICES, DIMS, NUM_LAYERS, WINDOW_SIZES
from tests.test_unwindow_matmul import python_unwindow_matmul, pytorch_unwindow_matmul
from tests.test_window_matmul import python_window_matmul, pytorch_window_matmul
from tests.utils import get_key, get_query, get_value


class Network(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        window_size: int,
        func_type: str,
        linear: torch.nn.Linear,
    ) -> None:
        super().__init__()
        self.layers = [
            AttentionLayer(window_size, func_type, linear) for _ in range(num_layers)
        ]
        self.linear = linear

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        for layer in self.layers:
            out = layer(query, key, value)
            query = self.linear(out)
            key = self.linear(out)
            value = self.linear(out)
        return out


class AttentionLayer(torch.nn.Module):
    def __init__(
        self, window_size: int, func_type: str, linear: torch.nn.Linear
    ) -> None:
        super().__init__()
        self.linear = linear
        self.window_size = window_size
        self.func_type = func_type

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        key = key.transpose(-1, -2)
        if self.func_type == "python":
            att_scores = python_window_matmul(query, key, self.window_size)
        elif self.func_type == "pytorch":
            att_scores = pytorch_window_matmul(query, key, self.window_size)
        elif self.func_type == "custom":
            att_scores = window_matmul(query, key, self.window_size)
        else:
            raise ValueError("unknown func type")
        att_scores = att_scores / math.sqrt(query.shape[-1])
        att_probs = torch.nn.functional.softmax(att_scores, dim=-1)
        if self.func_type == "python":
            out = python_unwindow_matmul(att_probs, value, self.window_size)
        elif self.func_type == "pytorch":
            out = pytorch_unwindow_matmul(att_probs, value, self.window_size)
        elif self.func_type == "custom":
            out = unwindow_matmul(att_probs, value, self.window_size)
        else:
            raise ValueError("unknown func type")
        out = self.linear(out)
        out = torch.sigmoid(out)
        return out


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("device", DEVICES)
def test_window_size_0(dim: Tuple[int, int, int, int], device: torch.device):
    query = get_query(*dim, device)
    key = get_key(*dim, device)
    value = get_value(*dim, device)

    att = window_matmul(query, key, 0)
    att_prob = att.softmax(-1)
    out = unwindow_matmul(att_prob, value, 0)

    assert att.shape[-1] == 1
    assert torch.allclose(out, value)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("window_size", WINDOW_SIZES)
@pytest.mark.parametrize("device", DEVICES)
def test_window_attention(
    dim: Tuple[int, int, int, int],
    window_size: int,
    device: torch.device,
):
    linear = torch.nn.Linear(dim[-1], dim[-1])

    python_network = Network(NUM_LAYERS, window_size, "python", linear).to(device)
    pytorch_network = Network(NUM_LAYERS, window_size, "pytorch", linear).to(device)
    custom_network = Network(NUM_LAYERS, window_size, "custom", linear).to(device)
    networks = {
        "python": python_network,
        "pytorch": pytorch_network,
        "custom": custom_network,
    }

    query = get_query(*dim, device)
    key = get_key(*dim, device)
    value = get_value(*dim, device)

    outs = {}
    query_grads = {}
    key_grads = {}
    value_grads = {}
    for network_name, network in networks.items():
        query = query.detach().clone().requires_grad_(True)
        key = key.detach().clone().requires_grad_(True)
        value = value.detach().clone().requires_grad_(True)
        out = network(query, key, value)
        out.mean().backward()
        outs[network_name] = out
        query_grads[network_name] = query.grad
        key_grads[network_name] = key.grad
        value_grads[network_name] = value.grad

    for network_name_1, network_name_2 in combinations(networks, 2):
        out_1 = outs[network_name_1]
        out_2 = outs[network_name_2]
        assert torch.allclose(out_1, out_2, atol=1e-6)
        query_grad_1 = query_grads[network_name_1]
        query_grad_2 = query_grads[network_name_2]
        assert torch.allclose(query_grad_1, query_grad_2, atol=1e-6)
        key_grad_1 = key_grads[network_name_1]
        key_grad_2 = key_grads[network_name_2]
        assert torch.allclose(key_grad_1, key_grad_2, atol=1e-6)
        value_grad_1 = value_grads[network_name_1]
        value_grad_2 = value_grads[network_name_2]
        assert torch.allclose(value_grad_1, value_grad_2, atol=1e-6)
