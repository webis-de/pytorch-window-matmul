import torch
from typing import List, Tuple

NUM_LAYERS: int = 3
DIMS: List[Tuple[int, int, int, int]] = [
    (1, 1, 2, 3),
    (2, 2, 2, 3),
    (2, 2, 4, 3),
    (2, 2, 16, 3),
    (2, 2, 32, 3),
]
BLOCK_SIZES: List[int] = [8, 16]
WINDOW_SIZES: List[int] = [2, 4, 8, 16, 32]
DEVICES: List[torch.device] = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

torch.manual_seed(42)
