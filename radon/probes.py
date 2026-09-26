"""Probe machinery for RADON: Hadamard codes, Latin-square coloring, and random sign flips.

Implements theory.md §4-§5 (Lean: master_identity, cycle_identity,
diag_recovery_exact, coded_unbiased, coded_variance):

* Each coordinate receives a code index corresponding to a column of a Sylvester-Hadamard matrix H_m.
* Weight matrices utilize Latin-square coloring: code(a, b) = (a + b) mod m. Coordinates in the
  same row or same column (the dominant cross-talk interference pairs) receive distinct,
  hence mutually orthogonal, codes for offsets < m.
* 1D parameter vectors use modular band coloring: code(i) = i mod m.
* Probe r of a cycle: v = s ⊙ H_m[r, code], where s is an independent uniform Rademacher
  global sign flip fixed for the entire cycle.
"""

from collections.abc import Sequence

import torch


def hadamard(m: int) -> torch.Tensor:
    """Generate a Sylvester-Hadamard matrix of order m (power of 2). Elements in {-1, +1}."""
    assert m > 0 and (m & (m - 1)) == 0, f"m must be a power of 2, got {m}"
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < m:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H


def code_tensor(shape: torch.Size, m: int, device: torch.device) -> torch.Tensor:
    """Assign Latin-square or band codes to a parameter tensor of given shape."""
    if len(shape) >= 2:
        rows = shape[0]
        cols = int(torch.tensor(shape[1:]).prod().item())
        a = torch.arange(rows, device=device).unsqueeze(1)
        b = torch.arange(cols, device=device).unsqueeze(0)
        return ((a + b) % m).reshape(shape)
    idx = torch.arange(shape[0] if len(shape) == 1 else 1, device=device)
    return (idx % m).reshape(shape)


def flip_signs(shape: torch.Size, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate a deterministic Rademacher ±1 sign flip tensor from a cycle seed."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    raw = torch.randint(0, 2, shape, generator=generator, device=device, dtype=torch.int8)
    return raw.to(dtype) * 2 - 1


class ProbeGenerator:
    """Generates directional coded probe tensors for model parameters cycling Hadamard rows."""

    def __init__(self, params: Sequence[torch.Tensor], m: int = 16, seed: int = 1234):
        self.m = m
        self.seed = seed
        self.params = list(params)
        self.had = {}
        self.codes = {}
        for i, p in enumerate(self.params):
            dev = p.device
            if dev not in self.had:
                self.had[dev] = hadamard(m).to(dev)
            self.codes[i] = code_tensor(p.shape, m, dev)

    def probe(self, cycle_idx: int, r: int) -> list[torch.Tensor]:
        """Generate probe r in [0, m) of cycle cycle_idx (one +/-1 tensor per parameter)."""
        assert 0 <= r < self.m, f"r must be in [0, {self.m}), got {r}"
        out = []
        for i, p in enumerate(self.params):
            row = self.had[p.device][r]
            d = flip_signs(
                p.shape,
                self.seed + 1_000_003 * cycle_idx + 7919 * i,
                p.device,
                torch.float32,
            )
            out.append(d * row[self.codes[i]])
        return out

    def scheduled_probe(self, step: int, frequency: int = 4) -> list[torch.Tensor]:
        """Probe on a scheduled step, keeping sign flips fixed for a full code cycle."""
        if step < 0 or frequency < 1 or step % frequency:
            raise ValueError("step must be a nonnegative multiple of frequency")
        probe_index = step // frequency
        return self.probe(cycle_idx=probe_index // self.m, r=probe_index % self.m)
