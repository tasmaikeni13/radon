# RADON

RADON is the **Residual-aware Antithetic Decoupled Orthogonal Newton Optimizer**, an experimental PyTorch optimizer for diagonal curvature estimation. Its loss Hessian decomposition is \(H = S + R\), where \(S = J^\top (\nabla^2\ell)J\) and \(R\) is the residual curvature. The code estimates the structural diagonal with sampled labels and probes the residual with coded directions.

## Central research question

For a specified accuracy target and compute budget, how should the optimizer **adaptively choose both the number and directions** of Hessian-vector products used to estimate its curvature diagonal as the model and data change? When should it stop probing and update? One Hessian-vector product returns a vector; reconstructing the entire Hessian is a different, more expensive objective.

The exact coded-probe variance identity in [theory.md](theory.md) is for a fixed matrix and a preset code. It does not establish an optimal adaptive policy for a moving neural-network Hessian. The current heavy-run drivers can compare selected configurations on validation loss and time, but they do not yet measure a curvature-error-versus-cost frontier or implement an adaptive policy. The required diagnostic protocol and evidence boundary are in [Phase 8](phases/phase8.md). No optimality or performance result is claimed.

## Evidence status

The repository contains source code, Lean proofs, numerical checks, and lightweight CPU smoke tests. Previous result JSON files, plots, and a compiled PDF were removed because their reported training numbers were embedded in source code rather than produced by the runs. No 600M-token sweep, 2.5B-token benchmark, TPU performance result, or ablation result is currently certified here. Phases 6–8 are reserved for the owner's heavy runs; the smoke tests exercise small models and do not measure competitive performance.

See [phase status](phases/state.json) for the current gate status. A numerical gate or smoke pass establishes only what that gate checks.

## Local verification

Use a Python environment with PyTorch, NumPy, SciPy, and pytest. On a host without a TPU, set:

```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
python3 -m verify.numerical_gate
pytest tests/
python3 -m verify.verify_kernels
python3 -m verify.verify_models
cd proofs/RadonCert && ~/.elan/bin/lake build
```

The 10 numerical gates use small fp64 examples. The kernel and model checks use CPU-sized workloads; they do not verify TPU collectives or training throughput.

## Phases 6–8 smoke tests

```bash
python3 experiments/sweep_hparams.py --smoke --steps 1
python3 experiments/run_competitive_benchmark.py --smoke --steps 1
python3 experiments/run_ablations.py --smoke --steps 1
```

Smoke tests use a small transformer and synthetic tokens when local arrays are absent. They write no benchmark results. The full drivers are implemented for the owner's later runs: they enforce real, separate train and validation arrays, count processed token positions exactly, evaluate held-out validation loss, and write raw measured runs before aggregate reports. Their TPU launch option is `--full --tpu`; the TPU path has not been tested on physical hardware.

Phase 6 evaluates each registered candidate for 600M processed token positions **per seed** at 2,500 steps. Phase 7 uses the measured Phase 6 selections for 2.5B positions per optimizer and seed. Phase 8 uses measured Phase 7 evidence for 2.5B positions per variant and seed. See the phase documents for the expected file names and owner-run commands. These full modes are expensive and have not been invoked here.

## Source map

- `radon/`: optimizer, Hessian-vector products, probe construction, curvature split, TPU helpers, and peer baselines.
- `models/` and `data/`: transformer and ViT models, plus memory-mapped token loading with a deterministic synthetic fallback for offline tests.
- `proofs/RadonCert/`: Lean 4 formal development.
- `verify/` and `tests/`: numerical and software checks.
- `phases/`: phase specifications and current status.
- `experiments/`: shared training loop, smoke tests, and full-run drivers.

## Data policy

The synthetic fallback is intended only for offline tests. Any later full-run result must record the real dataset path, token count, seed, model configuration, optimizer configuration, validation metric, and hardware. Generated `runs/`, `figures/`, and `paper/*.pdf` artifacts are ignored by Git until reviewed.
