"""Matrix-free adaptive residual diagonal probing at a frozen parameter state.

The rank-one direction is learned from one pilot residual HVP. Two independent
calibration probes choose whether to use it and how many final probes to draw.
Only fresh final probes enter the estimate, so selection does not bias it at a
fixed Hessian.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

Vector = list[torch.Tensor]
Matvec = Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]]


@dataclass(frozen=True)
class AdaptiveProbeConfig:
    """Cost and accuracy controls for one frozen-state estimate."""

    target_relative_rms: float = 0.25
    min_final_probes: int = 1
    max_final_probes: int = 4
    scale_floor: float = 1e-12

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_relative_rms) or self.target_relative_rms <= 0:
            raise ValueError("target_relative_rms must be finite and positive")
        if not 1 <= self.min_final_probes <= self.max_final_probes:
            raise ValueError("final probe limits must satisfy 1 <= min <= max")
        if not math.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("scale_floor must be finite and positive")


@dataclass(frozen=True)
class ProbeEstimate:
    """Estimated diagonal and actual matrix-vector call accounting."""

    diagonal: Vector
    hvp_calls: int
    final_probes: int
    projection_rank: int
    variance_proxy: float


def _signs(templates: Sequence[torch.Tensor], seed: int) -> Vector:
    vectors = []
    for index, template in enumerate(templates):
        generator = torch.Generator(device=template.device)
        generator.manual_seed(seed + 1_000_003 * index)
        bits = torch.randint(
            0, 2, template.shape, generator=generator,
            device=template.device, dtype=torch.int8,
        )
        vectors.append(bits.to(template.dtype).mul_(2).sub_(1))
    return vectors


def _dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum((a.double() * b.double()).sum() for a, b in zip(left, right))


def _weighted_square(
    values: Sequence[torch.Tensor], weights: Sequence[torch.Tensor] | None,
) -> torch.Tensor:
    if weights is None:
        return sum(value.double().square().sum() for value in values)
    return sum(
        (value.double().square() * weight.double()).sum()
        for value, weight in zip(values, weights)
    )


def _checked_matvec(matvec: Matvec, templates: Sequence[torch.Tensor], vector: Vector) -> Vector:
    product = list(matvec(vector))
    if len(product) != len(templates) or any(
        item.shape != template.shape for item, template in zip(product, templates)
    ):
        raise ValueError("matvec returned tensors with incompatible shapes")
    if any(not torch.isfinite(item).all().item() for item in product):
        raise FloatingPointError("matvec returned a non-finite product")
    return [item.detach() for item in product]


def _validate_inputs(
    templates: Sequence[torch.Tensor], weights: Sequence[torch.Tensor] | None,
) -> None:
    if not templates:
        raise ValueError("at least one parameter tensor is required")
    if any(not template.is_floating_point() for template in templates):
        raise ValueError("parameter templates must be floating point")
    if weights is None:
        return
    if len(weights) != len(templates) or any(
        weight.shape != template.shape for weight, template in zip(weights, templates)
    ):
        raise ValueError("weights must match parameter tensor shapes")
    if any((not torch.isfinite(weight).all().item()) or (weight < 0).any().item() for weight in weights):
        raise ValueError("weights must be finite and nonnegative")


@torch.no_grad()
def adaptive_residual_diagonal(
    matvec: Matvec,
    templates: Sequence[torch.Tensor],
    config: AdaptiveProbeConfig = AdaptiveProbeConfig(),
    *,
    seed: int = 0,
    weights: Sequence[torch.Tensor] | None = None,
) -> ProbeEstimate:
    """Estimate diag(R) with adaptive direction and probe count.

    One Rademacher pilot forms q = R omega / ||R omega||. The exact diagonal of
    R q q^T is computed using a second HVP. Two separate Rademacher probes
    compare projected and full residual noise, then choose s. The final s
    probes are fresh and independent of pilot and calibration. Every HVP is
    included in hvp_calls.
    """
    _validate_inputs(templates, weights)
    pilot = _signs(templates, seed)
    y = _checked_matvec(matvec, templates, pilot)
    calls = 1
    y_norm = _dot(y, y).sqrt().item()
    if y_norm > 0:
        q = [item / y_norm for item in y]
        rq = _checked_matvec(matvec, templates, q)
        calls += 1
        low_rank_diagonal = [a * b for a, b in zip(q, rq)]
        rank = 1
    else:
        q = None
        low_rank_diagonal = [torch.zeros_like(item) for item in templates]
        rank = 0

    def draw_sample(sample_seed: int, project: bool) -> Vector:
        z = _signs(templates, sample_seed)
        if q is None or not project:
            direction = z
        else:
            coefficient = _dot(q, z).item()
            direction = [zi - coefficient * qi for zi, qi in zip(z, q)]
        product = _checked_matvec(matvec, templates, direction)
        return [zi * bi for zi, bi in zip(z, product)]

    def calibration_sample(sample_seed: int) -> tuple[Vector, Vector]:
        z = _signs(templates, sample_seed)
        if q is None:
            full = _checked_matvec(matvec, templates, z)
            sample = [zi * item for zi, item in zip(z, full)]
            return sample, sample
        coefficient = _dot(q, z).item()
        direction = [zi - coefficient * qi for zi, qi in zip(z, q)]
        remainder = _checked_matvec(matvec, templates, direction)
        projected = [zi * item for zi, item in zip(z, remainder)]
        full = [
            zi * (item + coefficient * rqi)
            for zi, item, rqi in zip(z, remainder, rq)
        ]
        return projected, full

    projected_first, full_first = calibration_sample(seed + 10_000_019)
    projected_second, full_second = calibration_sample(seed + 20_000_033)
    calls += 2
    projected_difference = [
        a - b for a, b in zip(projected_first, projected_second)
    ]
    full_difference = [a - b for a, b in zip(full_first, full_second)]
    projected_proxy = 0.5 * _weighted_square(projected_difference, weights).item()
    full_proxy = 0.5 * _weighted_square(full_difference, weights).item()
    use_projection = q is not None and projected_proxy < full_proxy
    rank = int(use_projection)
    first, second = (
        (projected_first, projected_second) if use_projection
        else (full_first, full_second)
    )
    if not use_projection:
        low_rank_diagonal = [torch.zeros_like(item) for item in templates]
    variance_proxy = projected_proxy if use_projection else full_proxy
    calibration_diagonal = [
        low + 0.5 * (a + b)
        for low, a, b in zip(low_rank_diagonal, first, second)
    ]
    diagonal_scale = max(
        _weighted_square(calibration_diagonal, weights).item(), config.scale_floor
    )
    target_squared = config.target_relative_rms**2 * diagonal_scale
    requested = math.ceil(variance_proxy / target_squared)
    count = min(config.max_final_probes, max(config.min_final_probes, requested))

    accumulated = [torch.zeros_like(item) for item in templates]
    for index in range(count):
        sample = draw_sample(seed + 30_000_049 + 7_919 * index, use_projection)
        for total, value in zip(accumulated, sample):
            total.add_(value)
    calls += count
    diagonal = [
        low + total / count
        for low, total in zip(low_rank_diagonal, accumulated)
    ]
    return ProbeEstimate(diagonal, calls, count, rank, variance_proxy)


@torch.no_grad()
def fixed_rademacher_diagonal(
    matvec: Matvec,
    templates: Sequence[torch.Tensor],
    probes: int,
    *,
    seed: int = 0,
) -> ProbeEstimate:
    """Fixed-count Hutchinson diagonal estimate for matched-cost comparisons."""
    _validate_inputs(templates, None)
    if probes < 1:
        raise ValueError("probes must be positive")
    accumulated = [torch.zeros_like(item) for item in templates]
    for index in range(probes):
        signs = _signs(templates, seed + 7_919 * index)
        product = _checked_matvec(matvec, templates, signs)
        for total, sign, value in zip(accumulated, signs, product):
            total.add_(sign * value)
    diagonal = [total / probes for total in accumulated]
    return ProbeEstimate(diagonal, probes, probes, 0, float("nan"))
