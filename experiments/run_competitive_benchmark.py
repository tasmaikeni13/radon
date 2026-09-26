"""Phase 7: measured 2.5B-token comparison and small CPU smoke checks.

Full mode is reserved for the owner. Each optimizer and seed receives exactly
2.5B training token positions and a separate held-out validation evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.fineweb import TRAIN_NPY, VALID_NPY
from experiments.sweep_hparams import REPORT_PATH as SWEEP_REPORT_PATH
from experiments.sweep_hparams import (
    SEEDS as SWEEP_SEEDS,
)
from experiments.sweep_hparams import (
    SMOKE_CONFIGS,
    SWEEP_STEPS,
    candidate_grid,
)
from experiments.sweep_hparams import (
    raw_path as sweep_raw_path,
)
from experiments.sweep_hparams import (
    validate_raw as validate_sweep_raw,
)
from experiments.training import TrainSpec, run_training, validate_measured_result
from radon.tpu import get_device, get_world_size, is_master

SEEDS = (42, 43, 44)
TOKENS_PER_RUN = 2_500_000_000
DEFAULT_STEPS = 10_000
OUTPUT_DIR = REPO_ROOT / "runs" / "competitive_benchmark"
RESULTS_PATH = OUTPUT_DIR / "results.json"


def raw_path(optimizer: str, seed: int) -> Path:
    return OUTPUT_DIR / f"{optimizer}_seed{seed}.json"


def selected_configs() -> dict[str, dict[str, Any]]:
    if not SWEEP_REPORT_PATH.exists():
        raise FileNotFoundError("Phase 6 selection report is required for the full benchmark")
    report = json.loads(SWEEP_REPORT_PATH.read_text(encoding="utf-8"))
    if report.get("status") != "measured" or report.get("tokens_per_candidate_seed") != 600_000_000:
        raise ValueError("Phase 6 report is not a measured 600M-token sweep")
    if report.get("seeds") != list(SWEEP_SEEDS) or report.get("steps_per_run") != SWEEP_STEPS:
        raise ValueError("Phase 6 report has incomplete seed coverage")
    selected = {}
    for name, candidates in candidate_grid().items():
        reported = report["optimizers"][name]
        if len(reported["candidates"]) != len(candidates):
            raise ValueError(f"Incomplete Phase 6 candidate coverage for {name}")
        measured = []
        for index, hp in enumerate(candidates):
            runs = []
            for seed in SWEEP_SEEDS:
                path = sweep_raw_path(name, index, seed)
                if not path.exists():
                    raise FileNotFoundError(f"Missing Phase 6 raw run: {path}")
                raw = json.loads(path.read_text(encoding="utf-8"))
                validate_sweep_raw(raw, name, hp, seed)
                runs.append(raw)
            loss = mean(run["val_loss"] for run in runs)
            summary = reported["candidates"][index]
            if summary["candidate"] != index or summary["hyperparameters"] != hp:
                raise ValueError(f"Phase 6 candidate metadata differs from raw runs for {name}")
            expected_files = [
                str(sweep_raw_path(name, index, seed).relative_to(REPO_ROOT))
                for seed in SWEEP_SEEDS
            ]
            if summary["raw_files"] != expected_files:
                raise ValueError(f"Phase 6 candidate file list differs for {name}")
            if not math.isclose(summary["mean_val_loss"], loss, rel_tol=1e-12):
                raise ValueError(f"Phase 6 candidate loss differs from raw runs for {name}")
            if not math.isclose(
                summary["mean_val_ppl"], mean(run["val_ppl"] for run in runs), rel_tol=1e-12
            ):
                raise ValueError(f"Phase 6 candidate perplexity differs from raw runs for {name}")
            measured.append(loss)
        winner = min(range(len(candidates)), key=lambda index: measured[index])
        best = reported["best"]
        if best != reported["candidates"][winner]:
            raise ValueError(f"Phase 6 selection differs from raw runs for {name}")
        selected[name] = candidates[winner]
    return selected


def validate_raw(
    result: dict[str, Any], optimizer: str, seed: int,
    hp: dict[str, Any], steps: int,
) -> None:
    expected = {
        "optimizer": optimizer,
        "seed": seed,
        "hyperparameters": hp,
        "steps": steps,
        "tokens_total": TOKENS_PER_RUN,
        "model_params": 125_160_192,
        "smoke": False,
        "synthetic_train": False,
        "synthetic_valid": False,
        "block_size": 2048,
        "train_file": str(TRAIN_NPY),
        "valid_file": str(VALID_NPY),
    }
    validate_measured_result(result, expected)


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
        print("SMOKE PASS: small-model paths executed; no benchmark result was written")


def compile_results(steps: int) -> bool:
    configs = selected_configs()
    summaries = {}
    for optimizer, hp in configs.items():
        runs = []
        for seed in SEEDS:
            path = raw_path(optimizer, seed)
            if not path.exists():
                return False
            result = json.loads(path.read_text(encoding="utf-8"))
            validate_raw(result, optimizer, seed, hp, steps)
            runs.append(result)
        summaries[optimizer] = {
            "hyperparameters": hp,
            "mean_val_loss": mean(run["val_loss"] for run in runs),
            "std_val_loss": stdev(run["val_loss"] for run in runs),
            "mean_val_ppl": mean(run["val_ppl"] for run in runs),
            "std_val_ppl": stdev(run["val_ppl"] for run in runs),
            "mean_step_time_ms": mean(run["mean_step_time_ms"] for run in runs),
            "raw_files": [str(raw_path(optimizer, seed).relative_to(REPO_ROOT)) for seed in SEEDS],
        }
    report = {
        "phase": 7,
        "status": "measured",
        "tokens_per_optimizer_seed": TOKENS_PER_RUN,
        "steps_per_run": steps,
        "seeds": list(SEEDS),
        "train_file": str(TRAIN_NPY),
        "valid_file": str(VALID_NPY),
        "sweep_report": str(SWEEP_REPORT_PATH.relative_to(REPO_ROOT)),
        "results": summaries,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return True


def run_full(optimizer_filter: str | None, steps: int) -> None:
    configs = selected_configs()
    device = get_device()
    for optimizer, hp in configs.items():
        if optimizer_filter is not None and optimizer != optimizer_filter:
            continue
        for seed in SEEDS:
            path = raw_path(optimizer, seed)
            if path.exists():
                validate_raw(json.loads(path.read_text(encoding="utf-8")), optimizer, seed, hp, steps)
                continue
            spec = TrainSpec(
                optimizer=optimizer, seed=seed, hyperparameters=hp,
                steps=steps, tokens_total=TOKENS_PER_RUN,
                block_size=2048, microbatch_size=2,
                train_path=TRAIN_NPY, valid_path=VALID_NPY,
                eval_batches=32,
            )
            result = run_training(spec, device)
            if is_master():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(f"Measured {optimizer} seed {seed}: val PPL {result['val_ppl']:.3f}")
    if is_master() and compile_results(steps):
        print(f"Phase 7 measured benchmark report: {RESULTS_PATH}")


def verify_all(verbose: bool = True) -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("No measured Phase 7 result report exists")
    report = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if report.get("status") != "measured" or report.get("tokens_per_optimizer_seed") != TOKENS_PER_RUN:
        raise ValueError("Result report is not a measured 2.5B-token benchmark")
    steps = report["steps_per_run"]
    configs = selected_configs()
    for optimizer, hp in configs.items():
        runs = []
        for seed in SEEDS:
            path = raw_path(optimizer, seed)
            raw = json.loads(path.read_text(encoding="utf-8"))
            validate_raw(raw, optimizer, seed, hp, steps)
            runs.append(raw)
        summary = report["results"][optimizer]
        if summary["hyperparameters"] != hp:
            raise ValueError(f"Phase 7 summary differs from raw runs for {optimizer}")
        summary_metrics = {
            "mean_val_loss": mean(run["val_loss"] for run in runs),
            "std_val_loss": stdev(run["val_loss"] for run in runs),
            "mean_val_ppl": mean(run["val_ppl"] for run in runs),
            "std_val_ppl": stdev(run["val_ppl"] for run in runs),
            "mean_step_time_ms": mean(run["mean_step_time_ms"] for run in runs),
        }
        for key, value in summary_metrics.items():
            if not math.isclose(summary[key], value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Phase 7 {key} differs from raw runs for {optimizer}")
        expected_files = [str(raw_path(optimizer, seed).relative_to(REPO_ROOT)) for seed in SEEDS]
        if summary["raw_files"] != expected_files:
            raise ValueError(f"Phase 7 raw file list differs for {optimizer}")
        if verbose:
            print(f"{optimizer:<12s} validation PPL {summary['mean_val_ppl']:.3f} ± {summary['std_val_ppl']:.3f}")
    if verbose:
        print("Measured run coverage verified; relative performance is reported without a required winner")


def _worker(_index: int, args: argparse.Namespace) -> None:
    if args.smoke:
        run_smoke(args.steps)
    elif args.full:
        run_full(args.optimizer, args.full_steps)
    elif is_master():
        verify_all()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Small-model path check; writes no results")
    mode.add_argument("--full", action="store_true", help="Owner-run 2.5B-token comparison")
    mode.add_argument("--verify-all", action="store_true", help="Verify existing measured raw runs")
    parser.add_argument("--steps", type=int, default=1, help="Smoke steps (default: 1)")
    parser.add_argument("--full-steps", type=int, default=DEFAULT_STEPS, help="Full-run steps (default: 10000)")
    parser.add_argument("--optimizer", choices=tuple(SMOKE_CONFIGS), help="Full-run optimizer filter")
    parser.add_argument("--tpu", action="store_true", help="Launch one worker per TPU device")
    args = parser.parse_args()
    if args.steps < 1 or args.full_steps < 1:
        parser.error("Step counts must be positive")
    if args.tpu:
        import torch_xla

        torch_xla.launch(_worker, args=(args,))
    else:
        _worker(0, args)


if __name__ == "__main__":
    main()
