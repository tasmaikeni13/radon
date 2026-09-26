# Phase 8: Robustness and ablations

## Target

Measure how the probe budget, code geometry, and curvature split affect validation loss and wall time under matched training conditions. Compare a coded residual probe against an isotropic residual probe and a direct full-Hessian probe. Keep seed, model, data, schedule, and probe budget matched.

## Current implementation status

`experiments/run_ablations.py --smoke --steps 1` checks small-model budget, geometry, and channel paths, including an actual full-Hessian HVP control. The full driver requires measured Phase 7 results, uses the same real train and validation arrays and exact 2.5B processed token positions per variant and seed, and writes raw and aggregate reports after measurement. No full ablations have run here.

The owner's later TPU command is `python3 experiments/run_ablations.py --full --tpu`; `--variant` can split runs. Physical TPU execution remains unverified.
