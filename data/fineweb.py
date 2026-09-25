"""FineWeb-Edu Data Pipeline (2.5B Token Dataset Streamer).

Provides high-throughput memory-mapped token streaming for pretraining
on 2.5B tokens of FineWeb-Edu. Features hermetic synthetic fallback for
testing without filesystem dependencies.
"""

import os
from pathlib import Path

import numpy as np
import torch

DEFAULT_DATA_PATH = Path("/home/tasma/algebraic-intelligence/data")
DATA_DIR = Path(
    os.environ.get(
        "FINEWEB_DATA_DIR",
        str(DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else Path(__file__).resolve().parent),
    )
)
TRAIN_NPY = DATA_DIR / "fineweb_train_2_5B.npy"
VALID_NPY = DATA_DIR / "fineweb_valid.npy"


class FineWebDataset:
    """Memory-mapped dataset reader for 2.5B tokens of FineWeb-Edu."""

    def __init__(self, data_path: Path | None = None, is_train: bool = True):
        self.is_train = is_train
        if data_path is None:
            data_path = TRAIN_NPY if is_train else VALID_NPY

        if data_path.exists():
            self.data = np.load(str(data_path), mmap_mode="r")
            self.num_tokens = len(self.data)
            self.synthetic = False
        else:
            self.num_tokens = 2_500_000_000 if is_train else 10_000_000
            self.data = None
            self.synthetic = True

    def get_batch(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch a contiguous batch of (inputs, targets) of shape (B, T)."""
        if self.synthetic:
            gen = torch.Generator(device=device)
            if seed is not None:
                gen.manual_seed(seed)
            x = torch.randint(
                0,
                50257,
                (batch_size, block_size),
                generator=gen,
                device=device,
                dtype=torch.long,
            )
            y = torch.roll(x, -1, dims=-1)
            y[:, -1] = torch.randint(0, 50257, (batch_size,), generator=gen, device=device, dtype=torch.long)
            return x, y

        rng = np.random.default_rng(seed)
        max_idx = len(self.data) - block_size - 1
        indices = rng.integers(0, max_idx, size=batch_size)

        x_chunks = [self.data[i : i + block_size].astype(np.int64) for i in indices]
        y_chunks = [self.data[i + 1 : i + block_size + 1].astype(np.int64) for i in indices]

        x = torch.from_numpy(np.stack(x_chunks)).to(device)
        y = torch.from_numpy(np.stack(y_chunks)).to(device)
        return x, y


def load_dataset(is_train: bool = True) -> FineWebDataset:
    return FineWebDataset(is_train=is_train)
