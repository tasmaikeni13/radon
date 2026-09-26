# Phase 8: Robustness and ablations

## Target

Measure how the Hadamard cycle length, code geometry, and curvature split affect validation loss and wall time under matched training conditions. Compare a coded residual probe against an isotropic residual probe and a direct full-Hessian probe. Keep seed, model, data, and schedule matched. Each raw run records the number of Hessian-vector product calls per rank.

Changing the cycle length \(m\) at a fixed probe frequency changes code collisions and how long a code cycle spans training, **not** the number of Hessian-vector products. Consequently, these training ablations alone cannot answer RADON's central probe-count question. The current full driver implements these training ablations; the diagnostic below is a required extension before making a cost-accuracy or optimality claim.

## Required probe-count diagnostic

At several frozen checkpoints, hold the model, data batch, and precision fixed. Estimate \(\operatorname{diag}(H)\) and separately \(\operatorname{diag}(R)\) with matched counts of actual operator calls, varying both the number of probes and their geometry. Compare coded directions with independent Rademacher probes, a structured-probing reference, and an adaptive direction-and-stopping policy. Charge the adaptive policy for pilot probes. Obtain exact dense diagonals on small models, and exact diagonals for prespecified sampled coordinates on large models via coordinate Hessian-vector products. Measure normalized diagonal error, error in the preconditioned update, operator calls, and wall time across repeated random signs and batches. Measure structural-core sampling noise separately; do not attribute it to residual probes.

Across training runs, vary probe frequency to change actual query cost and report held-out loss against HVP calls and wall time. Test an adaptive policy only after its estimator, fresh-randomness conditions, and stopping rule are specified; evaluate on held-out checkpoints rather than those used to tune the rule. A conclusion should name the model, checkpoint distribution, data, hardware, error threshold, and confidence interval. Select the smallest measured cost meeting the preset error or quality target, or report the measured Pareto frontier if there is no single winner. A finite grid does not prove a universal optimum. Account for curvature drift when a code cycle spans multiple optimization steps.

## Current implementation status

`experiments/run_ablations.py --smoke --steps 1` checks small-model cycle-length, geometry, and channel paths, including an actual full-Hessian HVP control. The full driver requires measured Phase 7 results, uses the same real train and validation arrays and exact 2.5B processed token positions per variant and seed, and writes raw and aggregate reports after measurement. The frozen-checkpoint diagnostic, training frequency comparison, and adaptive policy above are not yet implemented. No full ablations have run here.

The owner's later TPU command is `python3 experiments/run_ablations.py --full --tpu`; `--variant` can split runs. Physical TPU execution remains unverified.
