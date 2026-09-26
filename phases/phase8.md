# Phase 8: Robustness and ablations

## Target

Measure how the Hadamard cycle length, adaptive projection, probe count, and curvature split affect validation loss and wall time under matched training conditions. Compare coded residual probes, independent fixed-count Rademacher probes, adaptive residual projection, isotropic probes, and a direct full-Hessian probe. Keep seed, model, data, and schedule matched. Each raw run records local and distributed Hessian-vector product counts, including adaptive pilot and calibration calls, and the rank-zero choices of direction and final probe count at every curvature update.

Changing the coded cycle length \(m\) at a fixed probe frequency changes code collisions and how long a code cycle spans training, **not** the number of Hessian-vector products. The adaptive and fixed-count Rademacher variants use multiple HVPs at one frozen step, so their recorded cost differs. Compare quality against both HVP calls and wall time; identical token budgets alone do not make operator costs equal.

## Required probe-count diagnostic

At several frozen checkpoints, hold the model, data batch, and precision fixed. Estimate \(\operatorname{diag}(H)\) and separately \(\operatorname{diag}(R)\) with matched counts of actual operator calls, varying both the number of probes and their geometry. Compare coded directions, independent Rademacher probes, a structured-probing reference, and adaptive projection. Charge the adaptive policy for pilot and calibration probes. Obtain exact dense diagonals on small models, and exact diagonals for prespecified sampled coordinates on large models via coordinate Hessian-vector products. Measure normalized diagonal error, error in the preconditioned update, operator calls, and wall time across repeated random signs and batches. Measure structural-core sampling noise separately; do not attribute it to residual probes.

Across training runs, vary probe frequency to change actual query cost and report held-out loss against HVP calls and wall time. Evaluate the adaptive policy on held-out checkpoints rather than those used to tune its target and probe bounds. A conclusion should name the model, checkpoint distribution, data, hardware, error threshold, and confidence interval. Select the smallest measured cost meeting the preset error or quality target, or report the measured Pareto frontier if there is no single winner. A finite grid does not prove a universal optimum. Account for curvature drift when a code cycle spans multiple optimization steps.

## Current implementation status

`experiments/run_ablations.py --smoke --steps 1` checks small-model cycle-length, geometry, adaptive, fixed-count random, and full-Hessian HVP paths. `experiments/monte_carlo_adaptive.py --smoke` compares methods on frozen synthetic matrices with exact diagonals and HVP-matched random controls; it does not measure neural training. The full driver requires measured Phase 7 results, uses real train and validation arrays and exact 2.5B processed token positions per variant and seed, and writes raw and aggregate reports after measurement. Frozen neural-checkpoint diagnostics and a dedicated frequency comparison are still required for a cost-accuracy claim. No full ablations have run here.

The owner's later TPU command is `python3 experiments/run_ablations.py --full --tpu`; `--variant` can split runs. Physical TPU execution remains unverified.
