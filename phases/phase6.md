# Phase 6: FineWeb-Edu hyperparameter sweep

## Target

The owner will run 3 seeds (42, 1337, 2024) for each optimizer on the 125M causal transformer, using the 600M-token FineWeb-Edu sweep dataset and a 2,500-step horizon. Candidate hyperparameters must be evaluated on held-out validation data, with actual token counts and all configurations recorded. No winning configuration has been established.

## Current implementation status

`experiments/sweep_hparams.py --smoke --steps 1` exercises five optimizer paths on a small model across three seeds and writes no result. The full driver registers 75 candidates, trains each candidate and seed for exactly 600M processed token positions, evaluates a separate validation array, writes one raw JSON per run, and selects by mean validation loss only after all 225 runs exist. Existing raw files are checked before they are reused. No candidate has been measured or selected here.

For the owner's later TPU run, provide `fineweb_sweep_600M.npy` and `fineweb_valid.npy` in `FINEWEB_DATA_DIR`, then use `python3 experiments/sweep_hparams.py --full --tpu`. `--optimizer` and `--candidate` can split the work. The TPU path needs physical v4-32 verification.
