import torch
from typing import List, Tuple

NUM_LAYERS: int = 3
DIMS: List[Tuple[int, int, int, int]] = [
    (2, 1, 1, 3),
    (2, 1, 4, 3),
    (2, 1, 16, 3),
    (2, 1, 32, 3),
    (2, 1, 16, 32),
]
BLOCK_SIZES: List[int] = [4]
WINDOW_SIZES: List[int] = [0, 1, 4, 16]
DEVICES: List[torch.device] = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

torch.manual_seed(42)
