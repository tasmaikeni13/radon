"""Phase 6: measured hyperparameter search and small CPU smoke checks.

Full mode is intentionally expensive and is reserved for the repository owner.
Each candidate is trained on exactly 600M token positions for each of three seeds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.fineweb import DATA_DIR, TRAIN_NPY, VALID_NPY
from experiments.training import TrainSpec, run_training
from radon.tpu import get_device, get_world_size, is_master

SEEDS = (42, 1337, 2024)
SWEEP_TOKENS = 600_000_000
SWEEP_STEPS = 2500
SWEEP_DATA = DATA_DIR / "fineweb_sweep_600M.npy"
OUTPUT_DIR = REPO_ROOT / "runs" / "phase6"
REPORT_PATH = REPO_ROOT / "runs" / "hpo_sweep_report.json"
SMOKE_CONFIGS = {
    "radon": {"lr": 4e-4, "gamma": 0.02, "cycle_m": 16, "probe_freq": 4},
    "adamw": {"lr": 1e-3, "weight_decay": 0.1, "betas": [0.9, 0.95]},
    "sophia": {"lr": 6e-4, "rho": 0.04},
    "adahessian": {"lr": 1e-3, "hessian_power": 1.0},
    "shampoo": {"lr": 1e-3, "update_freq": 10, "block_size": 128},
}


def candidate_grid() -> dict[str, list[dict[str, Any]]]:
    """Registered search space; candidate order is stable for resumable runs."""
    return {
        "radon": [
            {"lr": lr, "gamma": gamma, "cycle_m": m, "probe_freq": freq}
            for lr, gamma, m, freq in itertools.product(
                (2e-4, 4e-4, 6e-4), (0.01, 0.02, 0.05), (8, 16), (4, 8)
            )
        ],
        "adamw": [
            {"lr": lr, "weight_decay": decay, "betas": [0.9, beta2]}
            for lr, decay, beta2 in itertools.product(
                (5e-4, 1e-3, 2e-3), (0.01, 0.1), (0.95, 0.999)
            )
        ],
        "sophia": [
            {"lr": lr, "rho": rho}
            for lr, rho in itertools.product((3e-4, 6e-4, 1e-3), (0.02, 0.04, 0.08))
        ],
        "adahessian": [
            {"lr": lr, "hessian_power": power}
            for lr, power in itertools.product((5e-4, 1e-3, 2e-3), (0.5, 1.0))
        ],
        "shampoo": [
            {"lr": lr, "update_freq": frequency, "block_size": block_size}
            for lr, frequency, block_size in itertools.product(
                (5e-4, 1e-3, 2e-3), (10, 20), (64, 128)
            )
        ],
    }


def raw_path(optimizer: str, candidate: int, seed: int) -> Path:
    return OUTPUT_DIR / f"{optimizer}_candidate{candidate:02d}_seed{seed}.json"


def validate_raw(
    result: dict[str, Any], optimizer: str, hp: dict[str, Any], seed: int
) -> None:
    expected = {
        "optimizer": optimizer,
        "seed": seed,
        "hyperparameters": hp,
        "steps": SWEEP_STEPS,
        "tokens_total": SWEEP_TOKENS,
        "smoke": False,
        "synthetic_train": False,
        "synthetic_valid": False,
        "model_params": 125_160_192,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Invalid Phase 6 raw result: {key}={result.get(key)!r}")
    if not math.isfinite(result["val_loss"]):
        raise ValueError("Non-finite held-out validation loss")


def run_smoke(steps: int) -> None:
    device = get_device()
    world_size = get_world_size()
    for optimizer, hp in SMOKE_CONFIGS.items():
        losses = []
        for seed in SEEDS:
            spec = TrainSpec(
                optimizer=optimizer, seed=seed, hyperparameters=hp,
                steps=steps, tokens_total=steps * world_size * 350,
                block_size=128, microbatch_size=2,
                train_path=TRAIN_NPY, valid_path=VALID_NPY,
                smoke=True, eval_batches=1,
            )
            result = run_training(spec, device)
            losses.append(result["final_train_loss"])
        if is_master():
            print(f"{optimizer:<12s} three seeds: mean final training loss {mean(losses):.4f}")
    if is_master():
        print("SMOKE PASS: small-model paths executed; no sweep result was written")


def compile_report() -> bool:
    """Write a measured selection report only when every raw candidate exists."""
    selected: dict[str, Any] = {}
    grid = candidate_grid()
    for optimizer, candidates in grid.items():
        summaries = []
        for index, hp in enumerate(candidates):
            results = []
            for seed in SEEDS:
                path = raw_path(optimizer, index, seed)
                if not path.exists():
                    return False
                result = json.loads(path.read_text(encoding="utf-8"))
                validate_raw(result, optimizer, hp, seed)
                results.append(result)
            summaries.append({
                "candidate": index,
                "hyperparameters": hp,
                "mean_val_loss": mean(result["val_loss"] for result in results),
                "mean_val_ppl": mean(result["val_ppl"] for result in results),
                "raw_files": [str(raw_path(optimizer, index, seed).relative_to(REPO_ROOT)) for seed in SEEDS],
            })
        best = min(summaries, key=lambda item: item["mean_val_loss"])
        selected[optimizer] = {"best": best, "candidates": summaries}
    report = {
        "phase": 6,
        "status": "measured",
        "tokens_per_candidate_seed": SWEEP_TOKENS,
        "steps_per_run": SWEEP_STEPS,
        "seeds": list(SEEDS),
        "train_file": str(SWEEP_DATA),
        "valid_file": str(VALID_NPY),
        "optimizers": selected,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return True


def run_full(optimizer_filter: str | None, candidate_filter: int | None) -> None:
    device = get_device()
    grid = candidate_grid()
    if optimizer_filter is None and candidate_filter is not None:
        raise ValueError("--candidate requires --optimizer")
    for optimizer, candidates in grid.items():
        if optimizer_filter is not None and optimizer != optimizer_filter:
            continue
        for index, hp in enumerate(candidates):
            if candidate_filter is not None and index != candidate_filter:
                continue
            for seed in SEEDS:
                path = raw_path(optimizer, index, seed)
                if path.exists():
                    validate_raw(json.loads(path.read_text(encoding="utf-8")), optimizer, hp, seed)
                    continue
                spec = TrainSpec(
                    optimizer=optimizer, seed=seed, hyperparameters=hp,
                    steps=SWEEP_STEPS, tokens_total=SWEEP_TOKENS,
                    block_size=512, microbatch_size=4,
                    train_path=SWEEP_DATA, valid_path=VALID_NPY,
                    eval_batches=32,
                )
                result = run_training(spec, device)
                if is_master():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                    print(f"Measured {optimizer} candidate {index} seed {seed}: val PPL {result['val_ppl']:.3f}")
    if is_master() and compile_report():
        print(f"Phase 6 selection report: {REPORT_PATH}")


def _worker(_index: int, args: argparse.Namespace) -> None:
    if args.smoke:
        run_smoke(args.steps)
    else:
        run_full(args.optimizer, args.candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Small-model path check; writes no results")
    mode.add_argument("--full", action="store_true", help="Owner-run 600M-token candidate sweep")
    parser.add_argument("--steps", type=int, default=1, help="Smoke steps (default: 1)")
    parser.add_argument("--optimizer", choices=tuple(SMOKE_CONFIGS), help="Full-run optimizer filter")
    parser.add_argument("--candidate", type=int, help="Full-run candidate index")
    parser.add_argument("--tpu", action="store_true", help="Launch one worker per TPU device")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.tpu:
        import torch_xla

        torch_xla.launch(_worker, args=(args,))
    else:
        _worker(0, args)


if __name__ == "__main__":
    main()
