"""Small matrix-free Monte Carlo study of adaptive and fixed residual probes.

This measures diagonal error against exact dense ground truth on synthetic
matrices. It is a numerical diagnostic, not a neural training result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import mean

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from radon.adaptive import (
    AdaptiveProbeConfig,
    adaptive_residual_diagonal,
    fixed_rademacher_diagonal,
)
from radon.probes import ProbeGenerator


def matrices(size: int) -> dict[str, torch.Tensor]:
    """Three distinct symmetric residual structures, all with known diagonals."""
    generator = torch.Generator().manual_seed(2026)
    direction = torch.randn(size, generator=generator, dtype=torch.float64)
    rank_one = torch.outer(direction, direction) / size
    identity = torch.eye(size, dtype=torch.float64)
    banded = identity.clone()
    for offset, magnitude in ((1, 0.5), (2, 0.25)):
        banded += magnitude * torch.diag(torch.ones(size - offset), diagonal=offset)
        banded += magnitude * torch.diag(torch.ones(size - offset), diagonal=-offset)
    noise = torch.randn(size, size, generator=generator, dtype=torch.float64)
    flat = identity + (noise + noise.T) / math.sqrt(2 * size)
    return {"rank_one": rank_one, "banded": banded, "flat": flat}


def coded_estimate(matrix: torch.Tensor, seed: int, cycle_m: int) -> torch.Tensor:
    """Use the original fixed Hadamard code at one frozen matrix."""
    template = torch.zeros(matrix.shape[0], dtype=torch.float64)
    prober = ProbeGenerator([template], m=cycle_m, seed=seed)
    total = torch.zeros_like(template)
    for row in range(cycle_m):
        vector = prober.probe(0, row)[0].double()
        total += vector * (matrix @ vector)
    return total / cycle_m


def relative_squared_error(estimate: torch.Tensor, target: torch.Tensor) -> float:
    return ((estimate - target).square().sum() / target.square().sum().clamp_min(1e-20)).item()


def run_study(size: int, trials: int) -> dict:
    if size < 4 or trials < 1:
        raise ValueError("size must be at least four and trials must be positive")
    config = AdaptiveProbeConfig(target_relative_rms=0.25, max_final_probes=4)
    results = {}
    for name, matrix in matrices(size).items():
        target = matrix.diag()
        template = torch.zeros(size, dtype=torch.float64)
        def matvec(vectors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
            return [matrix @ vectors[0]]
        rows = []
        for trial in range(trials):
            seed = 100_003 + 7_919 * trial
            adaptive = adaptive_residual_diagonal(
                matvec, [template], config, seed=seed
            )
            matched = fixed_rademacher_diagonal(
                matvec, [template], adaptive.hvp_calls, seed=seed + 1_000_003
            )
            coded = coded_estimate(matrix, seed + 2_000_003, cycle_m=4)
            rows.append({
                "seed": seed,
                "adaptive_hvp_calls": adaptive.hvp_calls,
                "adaptive_final_probes": adaptive.final_probes,
                "adaptive_projection_rank": adaptive.projection_rank,
                "adaptive_relative_squared_error": relative_squared_error(
                    adaptive.diagonal[0], target
                ),
                "matched_random_relative_squared_error": relative_squared_error(
                    matched.diagonal[0], target
                ),
                "fixed_coded_m4_relative_squared_error": relative_squared_error(
                    coded, target
                ),
            })
        results[name] = {
            "trials": rows,
            "mean_hvp_calls_adaptive": mean(row["adaptive_hvp_calls"] for row in rows),
            "mean_final_probes_adaptive": mean(row["adaptive_final_probes"] for row in rows),
            "projection_selection_rate": mean(row["adaptive_projection_rank"] for row in rows),
            "rms_relative_error_adaptive": math.sqrt(mean(
                row["adaptive_relative_squared_error"] for row in rows
            )),
            "rms_relative_error_matched_random": math.sqrt(mean(
                row["matched_random_relative_squared_error"] for row in rows
            )),
            "rms_relative_error_fixed_coded_m4": math.sqrt(mean(
                row["fixed_coded_m4_relative_squared_error"] for row in rows
            )),
        }
    return {
        "study": "fixed_synthetic_residual_matrices",
        "size": size,
        "trials_per_matrix": trials,
        "adaptive_config": {
            "target_relative_rms": config.target_relative_rms,
            "min_final_probes": config.min_final_probes,
            "max_final_probes": config.max_final_probes,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use a tiny deterministic study")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--output", type=Path, help="Write raw trials to a new JSON file")
    args = parser.parse_args()
    if args.smoke and args.output is not None:
        parser.error("smoke mode does not write results")
    result = run_study(16 if args.smoke else args.size, 16 if args.smoke else args.trials)
    for name, case in result["results"].items():
        print(
            f"{name:<10s} HVPs={case['mean_hvp_calls_adaptive']:.2f} "
            f"projection={case['projection_selection_rate']:.2f} "
            f"relative RMS: adaptive={case['rms_relative_error_adaptive']:.4f}, "
            f"random matched={case['rms_relative_error_matched_random']:.4f}, "
            f"coded m4={case['rms_relative_error_fixed_coded_m4']:.4f}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(f"Wrote raw Monte Carlo trials to {args.output}")
    else:
        print("Monte Carlo check complete; no result file was written")


if __name__ == "__main__":
    main()
