import torch


def to_windowed(
    hidden_states: torch.Tensor,
    attention_window_size: int,
    pad: bool = True,
) -> torch.Tensor:
    if not attention_window_size:
        return hidden_states.unsqueeze(2)
    if pad:
        padding = (attention_window_size - 1) // 2
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, padding, padding))
    *sizes, seq_len, hidden_dim = hidden_states.size()
    num_windows = max(seq_len, attention_window_size) - attention_window_size + 1
    new_stride = tuple(list(hidden_states.stride()[:-1]) + [hidden_dim, 1])
    new_shape = (
        *sizes,
        num_windows,
        attention_window_size,
        hidden_dim,
    )
    return hidden_states.as_strided(new_shape, new_stride)


def memoize(func):
    cache = {}

    def memoized_func(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized_func


@memoize
def get_query(
    batch_size: int, num_heads: int, seq_len: int, hidden_dim: int
) -> torch.Tensor:
    tensor = torch.randn(batch_size, seq_len, hidden_dim * num_heads)
    tensor = torch.arange(batch_size * seq_len * hidden_dim * num_heads) + 1
    tensor = tensor.view(batch_size, seq_len, num_heads, -1).transpose(-2, -3)
    return tensor


@memoize
def get_key(
    batch_size: int, num_heads: int, seq_len: int, hidden_dim: int
) -> torch.Tensor:
    tensor = torch.randn(batch_size, seq_len, hidden_dim * num_heads)
    tensor = torch.arange(batch_size * seq_len * hidden_dim * num_heads) + 1
    tensor = tensor.view(batch_size, seq_len, num_heads, -1).transpose(-2, -3)
    return tensor


@memoize
def get_value(
    batch_size: int, num_heads: int, seq_len: int, hidden_dim: int
) -> torch.Tensor:
    tensor = torch.randn(batch_size, seq_len, hidden_dim * num_heads)
    tensor = torch.arange(batch_size * seq_len * hidden_dim * num_heads) + 1
    tensor = tensor.view(batch_size, seq_len, num_heads, -1).transpose(-2, -3)
    return tensor


@memoize
def get_att(
    batch_size: int, num_heads: int, seq_len: int, window_size: int
) -> torch.Tensor:
    tensor = torch.randn(batch_size, num_heads, seq_len, window_size * 2 + 1)
    tensor = torch.arange(batch_size * num_heads * seq_len * (window_size * 2 + 1))
    tensor = tensor.view(batch_size, num_heads, seq_len, window_size * 2 + 1)
    for i in range(seq_len):
        tensor[:, :, i, : max(window_size - i, 0)] = 0
        tensor[:, :, i, seq_len + window_size - i :] = 0
    return tensor
