"""Phase 8: measured RADON ablations and small CPU smoke checks.

Full mode is reserved for the owner and uses the same exact token budget and
validation path as the competitive benchmark for each variant and seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.fineweb import TRAIN_NPY, VALID_NPY
from experiments.run_competitive_benchmark import (
    DEFAULT_STEPS,
    RESULTS_PATH,
    TOKENS_PER_RUN,
    selected_configs,
)
from experiments.run_competitive_benchmark import (
    verify_all as verify_benchmark,
)
from experiments.sweep_hparams import SMOKE_CONFIGS
from experiments.training import TrainSpec, run_training, validate_measured_result
from radon.tpu import get_device, get_world_size, is_master

SEEDS = (42, 43, 44)
OUTPUT_DIR = REPO_ROOT / "runs" / "phase8"
REPORT_PATH = REPO_ROOT / "runs" / "ablations_report.json"
VARIANTS = (
    "cycle_m2", "cycle_m4", "cycle_m8", "cycle_m16", "cycle_m32",
    "isotropic", "full_hessian", "adaptive_projection",
    "fixed_rademacher_4", "fixed_rademacher_8",
)


def variant_config(name: str, base_hp: dict[str, Any]) -> tuple[dict[str, Any], str]:
    hp = dict(base_hp)
    if name.startswith("cycle_m"):
        hp["cycle_m"] = int(name.removeprefix("cycle_m"))
        return hp, "standard"
    if name == "isotropic":
        hp["cycle_m"] = 16
        return hp, "isotropic"
    if name == "full_hessian":
        hp["cycle_m"] = 16
        return hp, "full_hessian"
    if name == "adaptive_projection":
        hp.update({
            "cycle_m": 1, "adaptive_target": 0.25,
            "adaptive_min_probes": 1, "adaptive_max_probes": 4,
        })
        return hp, "adaptive_projection"
    if name.startswith("fixed_rademacher_"):
        hp.update({"cycle_m": 1, "probe_count": int(name.removeprefix("fixed_rademacher_"))})
        return hp, "fixed_rademacher"
    raise ValueError(f"Unknown ablation: {name}")


def raw_path(variant: str, seed: int) -> Path:
    return OUTPUT_DIR / f"{variant}_seed{seed}.json"


def validate_raw(
    result: dict[str, Any], variant_name: str, seed: int,
    hp: dict[str, Any], internal_variant: str, steps: int,
) -> None:
    expected = {
        "optimizer": "radon",
        "variant": internal_variant,
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
    base_hp = SMOKE_CONFIGS["radon"]
    for name in (
        "cycle_m2", "cycle_m4", "cycle_m16", "isotropic",
        "full_hessian", "adaptive_projection", "fixed_rademacher_4",
    ):
        hp, variant = variant_config(name, base_hp)
        spec = TrainSpec(
            optimizer="radon", seed=42, hyperparameters=hp,
            steps=steps, tokens_total=steps * world_size * 350,
            block_size=128, microbatch_size=2,
            train_path=TRAIN_NPY, valid_path=VALID_NPY,
            smoke=True, eval_batches=1, variant=variant,
        )
        result = run_training(spec, device)
        if is_master():
            print(f"{name:<16s} final training loss {result['final_train_loss']:.4f}")
    if is_master():
        print("SMOKE PASS: small-model ablation paths executed; no result report was written")


def compile_report(base_hp: dict[str, Any], steps: int) -> bool:
    summaries = {}
    for name in VARIANTS:
        hp, variant = variant_config(name, base_hp)
        runs = []
        for seed in SEEDS:
            path = raw_path(name, seed)
            if not path.exists():
                return False
            result = json.loads(path.read_text(encoding="utf-8"))
            validate_raw(result, name, seed, hp, variant, steps)
            runs.append(result)
        summaries[name] = {
            "hyperparameters": hp,
            "variant": variant,
            "mean_val_loss": mean(run["val_loss"] for run in runs),
            "std_val_loss": stdev(run["val_loss"] for run in runs),
            "mean_val_ppl": mean(run["val_ppl"] for run in runs),
            "std_val_ppl": stdev(run["val_ppl"] for run in runs),
            "mean_step_time_ms": mean(run["mean_step_time_ms"] for run in runs),
            "mean_hvp_calls_per_rank": mean(run["hvp_calls_per_rank"] for run in runs),
            "hvp_calls_per_rank_by_seed": [run["hvp_calls_per_rank"] for run in runs],
            "mean_hvp_calls_across_ranks": mean(run["hvp_calls_mean_per_rank"] for run in runs),
            "max_hvp_calls_per_rank": max(run["hvp_calls_max_per_rank"] for run in runs),
            "mean_adaptive_final_probes": (
                mean(
                    decision["final_probes"]
                    for run in runs for decision in run["adaptive_decisions"]
                ) if variant == "adaptive_projection" else None
            ),
            "adaptive_projection_fraction": (
                mean(
                    decision["projection_rank"]
                    for run in runs for decision in run["adaptive_decisions"]
                ) if variant == "adaptive_projection" else None
            ),
            "raw_files": [str(raw_path(name, seed).relative_to(REPO_ROOT)) for seed in SEEDS],
        }
    report = {
        "phase": 8,
        "status": "measured",
        "tokens_per_variant_seed": TOKENS_PER_RUN,
        "steps_per_run": steps,
        "seeds": list(SEEDS),
        "train_file": str(TRAIN_NPY),
        "valid_file": str(VALID_NPY),
        "phase7_report": str(RESULTS_PATH.relative_to(REPO_ROOT)),
        "variants": summaries,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return True


def run_full(variant_filter: str | None, steps: int) -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("Measured Phase 7 results are required before Phase 8")
    benchmark = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if benchmark.get("status") != "measured":
        raise ValueError("Phase 7 benchmark is not marked measured")
    verify_benchmark(verbose=is_master())
    base_hp = selected_configs()["radon"]
    device = get_device()
    for name in VARIANTS:
        if variant_filter is not None and name != variant_filter:
            continue
        hp, variant = variant_config(name, base_hp)
        for seed in SEEDS:
            path = raw_path(name, seed)
            if path.exists():
                validate_raw(json.loads(path.read_text(encoding="utf-8")), name, seed, hp, variant, steps)
                continue
            spec = TrainSpec(
                optimizer="radon", seed=seed, hyperparameters=hp,
                steps=steps, tokens_total=TOKENS_PER_RUN,
                block_size=2048, microbatch_size=2,
                train_path=TRAIN_NPY, valid_path=VALID_NPY,
                eval_batches=32, variant=variant,
            )
            result = run_training(spec, device)
            if is_master():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(f"Measured {name} seed {seed}: val PPL {result['val_ppl']:.3f}")
    if is_master() and compile_report(base_hp, steps):
        print(f"Phase 8 measured ablation report: {REPORT_PATH}")


def _worker(_index: int, args: argparse.Namespace) -> None:
    if args.smoke:
        run_smoke(args.steps)
    else:
        run_full(args.variant, args.full_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Small-model path check; writes no results")
    mode.add_argument("--full", action="store_true", help="Owner-run 2.5B-token ablations")
    parser.add_argument("--steps", type=int, default=1, help="Smoke steps (default: 1)")
    parser.add_argument("--full-steps", type=int, default=DEFAULT_STEPS, help="Full-run steps (default: 10000)")
    parser.add_argument("--variant", choices=VARIANTS, help="Full-run variant filter")
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
