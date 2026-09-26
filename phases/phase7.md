# Phase 7: Competitive FineWeb-Edu benchmark

## Target

After Phase 6 has measured and selected hyperparameters, the owner will run RADON, AdamW, Sophia-H, AdaHessian, and Shampoo on the same 125M transformer and FineWeb-Edu splits for 2.5B tokens per optimizer and seed. The benchmark needs three seeds, held-out validation perplexity, complete configurations, actual token counts, and timing from real hardware.

## Current implementation status

`experiments/run_competitive_benchmark.py --smoke --steps 1` exercises the five training paths on a small model and writes no results. The full driver requires a complete measured Phase 6 selection report, real `fineweb_train_2_5B.npy` and `fineweb_valid.npy` arrays, and trains each selected configuration for exactly 2.5B processed token positions per seed. Sophia-H and AdaHessian receive autodiff Hutchinson curvature samples; Shampoo uses blocked matrix preconditioning without distributed state sharding. Raw and aggregate reports are written only from measured runs. No full benchmark has run here.

The owner's later TPU command is `python3 experiments/run_competitive_benchmark.py --full --tpu`. `--optimizer` can split runs. Physical TPU execution and throughput remain unverified.
