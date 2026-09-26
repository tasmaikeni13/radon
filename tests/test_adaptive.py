"""Exact small-matrix checks for adaptive projection and probe accounting."""

import itertools
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from radon.adaptive import AdaptiveProbeConfig, adaptive_residual_diagonal


def test_conditional_projection_variance_identity():
    matrix = torch.tensor(
        [[2.0, 1.0, -0.5, 0.0],
         [1.0, -1.0, 0.25, 0.75],
         [-0.5, 0.25, 3.0, -1.5],
         [0.0, 0.75, -1.5, 1.0]],
        dtype=torch.float64,
    )
    q = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float64)
    q /= q.norm()
    projector = torch.eye(4, dtype=torch.float64) - torch.outer(q, q)
    remainder = matrix @ projector
    low_rank = (matrix @ q) * q
    assert torch.allclose(low_rank + remainder.diag(), matrix.diag(), atol=1e-14)

    signs = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=4)), dtype=torch.float64
    )
    samples = signs * (signs @ remainder.T)
    empirical_mean = samples.mean(dim=0)
    empirical_variance = samples.var(dim=0, unbiased=False)
    expected_variance = remainder.square().sum(dim=1) - remainder.diag().square()
    assert torch.allclose(empirical_mean, remainder.diag(), atol=1e-14)
    assert torch.allclose(empirical_variance, expected_variance, atol=1e-14)


def test_rank_one_direction_is_recovered_and_all_calls_counted():
    direction = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    matrix = torch.outer(direction, direction)
    calls = 0

    def matvec(vectors):
        nonlocal calls
        calls += 1
        return [matrix @ vectors[0]]

    estimate = adaptive_residual_diagonal(
        matvec, [torch.zeros(4, dtype=torch.float64)],
        AdaptiveProbeConfig(target_relative_rms=0.1, max_final_probes=4),
        seed=12,
    )
    assert estimate.projection_rank == 1
    assert estimate.final_probes == 1
    assert estimate.hvp_calls == calls == 5
    assert torch.allclose(estimate.diagonal[0], matrix.diag(), atol=1e-12)
