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
# BLOCK_SIZES: List[int] = [8]
WINDOW_SIZES: List[int] = [2, 4, 8, 16, 32]
# WINDOW_SIZES: List[int] = [2]
# DEVICES: List[torch.device] = [torch.device("cpu")]
DEVICES: List[torch.device] = [torch.device("cuda")]
DTYPES: List[torch.dtype] = [torch.float32]
# if torch.cuda.is_available():
#     DEVICES.append(torch.device("cuda"))
#     DTYPES.append(torch.float16)

torch.manual_seed(42)
